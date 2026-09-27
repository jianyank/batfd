"""论文 Eq.(1)–(4) 的三项损失，以及目标归一化。

对应关系（论文 → 本实现）::

    L_total = λ1·L_recon + λ2·L_late + λ3·L_reg          Eq.(1)
    L_recon = MSE(实测电压, 重建电压)                       Eq.(2)  只算 8 路电压
    L_late  = ‖z_model − s_model‖²                        Eq.(3)  **改成逐单体对齐**
    L_reg   = KL(q(z|x) ‖ N(0,I))                          Eq.(4)  变分潜空间

两处实质差异（均为修补论文缺陷）：

1. **Eq.(3) 的 z_model 从未被指明对应哪个物理量**（论文只说「a subset of latent
   variables」），论文 §4.2 承认因此抓不住单体间差异。本实现把 s_model 按单体
   拆开，**槽位 c 只对齐单体 c 的内阻**。
2. **论文全文未给出 λ1/λ2/λ3 的取值**，因此它的结果无法复算。本实现把三个权重
   放进配置并在验证集上定，**报告中必须声明这一点**。

目标归一化：论文没给具体做法，但 Eq.(3) 原文强调两个物理量要 "first normalized to
a comparable range"。本实现用与输入同款的稳健标准化（仅用训练集拟合）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

MAD_SCALE = 1.4826


@dataclass
class TargetScaler:
    """把 Hiddall 的内阻与 SOC 变成可比的标准化目标。**只允许在训练集上 fit**。

    四个统计量随 checkpoint 一起存盘。
    """

    res_median: torch.Tensor   # (n_cells,)
    res_scale: torch.Tensor    # (n_cells,)
    soc_median: torch.Tensor   # ()
    soc_scale: torch.Tensor    # ()

    @staticmethod
    def fit(res: torch.Tensor, soc: torch.Tensor, floor: float = 1e-3) -> "TargetScaler":
        """``res`` (N, n_cells)，``soc`` (N,)。NaN 会被忽略。"""
        def _med_mad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x.double()
            med = torch.nanmedian(x, dim=0).values
            mad = torch.nanmedian((x - med).abs(), dim=0).values * MAD_SCALE
            return med.float(), torch.maximum(mad, floor * med.abs().clamp_min(1e-12)).float()

        rm, rs = _med_mad(res)
        sm, ss = _med_mad(soc.unsqueeze(-1))
        return TargetScaler(rm, rs, sm.squeeze(0), ss.squeeze(0))

    def to(self, device: torch.device | str) -> "TargetScaler":
        return TargetScaler(
            self.res_median.to(device),
            self.res_scale.to(device),
            self.soc_median.to(device),
            self.soc_scale.to(device),
        )

    def norm_res(self, res: torch.Tensor) -> torch.Tensor:
        # 统计量随输入搬同一设备：漏搬只报 "found at least two devices"，
        # 且只在 GPU 上才暴露。
        return (res - self.res_median.to(res.device)) / self.res_scale.to(res.device)

    def norm_soc(self, soc: torch.Tensor) -> torch.Tensor:
        return (soc - self.soc_median.to(soc.device)) / self.soc_scale.to(soc.device)

    def state_dict(self) -> dict:
        return {
            "res_median": self.res_median,
            "res_scale": self.res_scale,
            "soc_median": self.soc_median,
            "soc_scale": self.soc_scale,
        }

    @staticmethod
    def from_state(d: dict) -> "TargetScaler":
        return TargetScaler(d["res_median"], d["res_scale"], d["soc_median"], d["soc_scale"])


def recon_loss(v_hat: torch.Tensor, v_norm: torch.Tensor) -> torch.Tensor:
    """Eq.(2)：只在被重建的电压维度上算 MSE。

    论文写的是 ``(1/N)Σ‖x_volt − x̂_volt‖²``；这里取 batch 内所有单体、所有
    时间步的均值，与论文式子在常数因子上等价（常数并入 λ1）。
    """
    return F.mse_loss(v_hat, v_norm)


def alignment_loss(
    a_pred: torch.Tensor,
    res_target: torch.Tensor,
    soc_pred: torch.Tensor,
    soc_target: torch.Tensor,
    *,
    w_res: float = 1.0,
    w_soc: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Eq.(3)：物理对齐损失，逐单体对齐内阻 + 包级对齐 SOC。

    ``a_pred`` (B, n_cells) 是潜空间的物理槽；``res_target`` (B, n_cells) 是
    已标准化的单体内阻。含 NaN 的项按掩码剔除（没有 Hiddall 的数据集就没有目标）。
    """
    m_res = torch.isfinite(res_target)
    if m_res.any():
        d_res = (a_pred[m_res] - res_target[m_res]) ** 2
        l_res = d_res.mean()
    else:
        l_res = a_pred.sum() * 0.0

    m_soc = torch.isfinite(soc_target)
    if m_soc.any():
        d_soc = (soc_pred[m_soc] - soc_target[m_soc]) ** 2
        l_soc = d_soc.mean()
    else:
        l_soc = soc_pred.sum() * 0.0

    total = w_res * l_res + w_soc * l_soc
    stats = {
        "late_res": float(l_res.detach()),
        "late_soc": float(l_soc.detach()),
        "late_res_frac_avail": float(m_res.float().mean()),
    }
    return total, stats


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Eq.(4)：对角高斯 q(z|x) 对 N(0,I) 的 KL 散度。

    ``KL = −0.5 · Σ_dims (1 + logvar − mu² − exp(logvar))``，
    在 batch 与槽位维度上取均值，在潜维上求和。
    """
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)).mean()


def total_loss(
    *,
    v_hat: torch.Tensor,
    v_norm: torch.Tensor,
    a_pred: torch.Tensor,
    res_target: torch.Tensor,
    soc_pred: torch.Tensor,
    soc_target: torch.Tensor,
    mu_cell: torch.Tensor,
    logvar_cell: torch.Tensor,
    mu_pack: torch.Tensor,
    logvar_pack: torch.Tensor,
    lambda1: float,
    lambda2: float,
    lambda3: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Eq.(1)：三项加权求和，分项同时记入日志 —— 消融时要能看清增益来自哪一项。"""
    l_rec = recon_loss(v_hat, v_norm)
    l_late, late_stats = alignment_loss(a_pred, res_target, soc_pred, soc_target)
    l_reg = kl_loss(mu_cell, logvar_cell) + kl_loss(mu_pack, logvar_pack)

    total = lambda1 * l_rec + lambda2 * l_late + lambda3 * l_reg
    stats = {
        "loss": float(total.detach()),
        "recon": float(l_rec.detach()),
        "late": float(l_late.detach()),
        "reg": float(l_reg.detach()),
        **late_stats,
    }
    return total, stats
