"""潜变量诊断（轴线①的核心证据）。

论文声称「把一部分潜变量与物理量对齐」能提升故障敏感性，但它在 §4.2 / 图 7 自己承认
失败：潜变量 *"fails to capture the inter-cell discrepancies"*，且**没有做消融**，
那 123–215 天的领先是不是潜变量对齐带来的无从判断。本模块提供两个**可证伪**的定量
诊断，把「潜空间是否真的单体分辨」变成能算的数：

1. **互相关矩阵** ``corr(a_c, r_c')``（n_cells × n_cells）：潜槽位真按单体分辨时应
   **接近对角** —— 单体 c 的物理槽与单体 c 的内阻相关，与其他单体的内阻不相关。
2. **单体间差异捕获度** ``corr(a_c − mean_{c'} a_{c'}, r_c − mean_{c'} r_{c'})``：
   正是论文自称失败的那一项 —— 减去同包均值后的「相对偏差」潜变量能否跟上，
   数值越高说明潜空间越能反映单体间的相对分化。

对照方式：同样的诊断跑在单体分辨结构（``cell_resolved=True``）与共享潜向量的消融
（``cell_resolved=False``，论文那种「未指明哪一部分对哪个单体」）上，比较对角优势
与差异捕获度 —— 这就是论文缺的那个隔离验证。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import channels, dataset as ds_mod


@dataclass
class LatentProbe:
    """一次潜空间采样的结果，均为 numpy。"""

    a_cell: np.ndarray      # (N, n_cells) 物理槽（对齐内阻的那个）
    z_cell: np.ndarray      # (N, n_cells, k) 完整单体潜变量
    soc_hat: np.ndarray     # (N,) SOC 物理槽
    res: np.ndarray         # (N, n_cells) 实测（Hiddall）单体内阻
    soc: np.ndarray         # (N,) 实测 SOC
    ids: np.ndarray         # (N,)


@torch.no_grad()
def probe_latent(
    model, cfg: dict, cache: dict, *, device="cpu", batch_size: int = 512
) -> LatentProbe:
    """把整个数据集过一遍编码器，取回潜槽位与对应的物理量。

    编码器输出取 ``mu``（确定性部分）而非采样值，保证诊断可复现。
    """
    if cache.get("hidden") is None:
        raise ValueError("该数据集没有 Hiddall，无法做潜变量-物理量诊断")

    hcfg = cfg["hidden"]
    res_cols = [c - 1 for c in hcfg["resistance_channels"]]
    soc_col = hcfg["soc_channel"] - 1

    device = torch.device(device)
    model = model.to(device).eval()

    n = int(cache["signal"].shape[0])
    ds = ds_mod.WindowDataset(cfg, cache, np.arange(n))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    n_cells = len(channels.cell_voltage_cols(cfg))
    k = int(cfg["model"]["cell_latent"])
    a = np.empty((n, n_cells), dtype=np.float32)
    z = np.empty((n, n_cells, k), dtype=np.float32)
    sh = np.empty(n, dtype=np.float32)
    ids = np.empty(n, dtype=np.int16)

    pos = 0
    for batch in loader:
        x = batch["x"].to(device)
        v = batch["v"].to(device)
        i = batch["i"].to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            # 电流走模型自己的归一化出口，不要在此重复实现
            # （曾直接调 normalizer 而被按通道广播，形状从 (B,1,T) 变成 (B,20,T)）
            mu_c, _, mu_p, _ = model.encode(
                model.normalizer(x), model.cell_norm(v), model._normalize_current(i)
            )
        b = mu_c.shape[0]
        z[pos : pos + b] = mu_c.float().cpu().numpy()
        a[pos : pos + b] = mu_c[..., 0].float().cpu().numpy()
        sh[pos : pos + b] = mu_p[..., 0].float().cpu().numpy()
        ids[pos : pos + b] = batch["id"].numpy()
        pos += b

    h = np.asarray(cache["hidden"])[:, res_cols].astype(np.float64)
    soc = np.asarray(cache["hidden"])[:, soc_col].astype(np.float64)
    return LatentProbe(a_cell=a, z_cell=z, soc_hat=sh, res=h, soc=soc, ids=ids)


def _safe_corr(u: np.ndarray, v: np.ndarray) -> float:
    """皮尔逊相关；任一侧为常量时返回 nan 而不是 0（0 会被误读成「无关系」）。"""
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    m = np.isfinite(u) & np.isfinite(v)
    if m.sum() < 3:
        return float("nan")
    u, v = u[m], v[m]
    if u.std() < 1e-12 or v.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(u, v)[0, 1])


def cross_correlation(probe: LatentProbe) -> np.ndarray:
    """``corr(a_c, r_c')`` 矩阵，形状 (n_cells, n_cells)。

    行 = 潜槽位对应的单体，列 = 内阻对应的单体。
    单体分辨成功时应该**接近对角**。
    """
    n_cells = probe.res.shape[1]
    m = np.full((n_cells, n_cells), np.nan)
    for c in range(n_cells):
        for c2 in range(n_cells):
            m[c, c2] = _safe_corr(probe.a_cell[:, c], probe.res[:, c2])
    return m


def diagonal_advantage(probe: LatentProbe) -> dict:
    """对角的「优势」：对角相关均值 − 非对角相关均值（把「接近对角」变成一个数）。

    单体分辨的结构应当显著为正；共享潜向量的消融应当接近 0
    （所有单体拿到同一个潜值，对角无从谈起）。
    """
    m = cross_correlation(probe)
    off = m[~np.eye(m.shape[0], dtype=bool)]
    diag = np.diag(m)
    return {
        "diag_mean": float(np.nanmean(diag)),
        "offdiag_mean": float(np.nanmean(off)),
        "advantage": float(np.nanmean(diag) - np.nanmean(off)),
        "matrix": m,
    }


def discrepancy_capture(probe: LatentProbe) -> dict:
    """单体间差异捕获度 —— 论文自称失败的那一项。

    潜槽位与内阻都减去「同包（同窗口）跨单体均值」，只留相对偏差，再逐单体求相关。
    这正是 §4.2 图 7 想画但没抓住的东西（那里 Cell 8 的异常演化没被潜变量反映出来）。

    Returns：字典含逐单体相关系数与均值；均值越接近 1，潜空间越能跟上单体分化。
    """
    a_dev = probe.a_cell - probe.a_cell.mean(axis=1, keepdims=True)
    r_dev = probe.res - probe.res.mean(axis=1, keepdims=True)
    per_cell = [_safe_corr(a_dev[:, c], r_dev[:, c]) for c in range(probe.res.shape[1])]
    finite = [x for x in per_cell if np.isfinite(x)]
    return {
        "per_cell": per_cell,
        "mean": float(np.mean(finite)) if finite else float("nan"),
        "n_finite": len(finite),
    }


def soc_alignment(probe: LatentProbe) -> dict:
    """SOC 物理槽与实测 SOC 的相关（论文 Eq.3 的另一半）。"""
    return {"corr": _safe_corr(probe.soc_hat, probe.soc)}


def report(probe: LatentProbe, *, label: str = "") -> dict:
    """一次性打出全部诊断。"""
    da = diagonal_advantage(probe)
    dc = discrepancy_capture(probe)
    sa = soc_alignment(probe)

    n = probe.res.shape[1]
    print(f"\n--- 潜变量诊断 {label} ---")
    print(f"  样本 {probe.a_cell.shape[0]} 窗口，{n} 单体")
    print(f"  互相关矩阵 corr(a_c, r_c')（行=潜槽位单体，列=内阻单体）:")
    hdr = "        " + "".join(f"  r{c + 1:>5}" for c in range(n))
    print(hdr)
    for c in range(n):
        print(f"     a{c + 1:>2} " + "".join(f" {da['matrix'][c, c2]:+.3f}" for c2 in range(n)))
    print(f"  对角均值 {da['diag_mean']:+.4f}  非对角均值 {da['offdiag_mean']:+.4f}  "
          f"对角优势 {da['advantage']:+.4f}")
    print(f"  单体间差异捕获度：均值 {dc['mean']:+.4f}"
          f"  （逐单体 {[None if not np.isfinite(x) else round(x, 3) for x in dc['per_cell']]}）")
    print(f"  SOC 对齐：corr {sa['corr']:+.4f}")

    return {"label": label, "diagonal_advantage": da, "discrepancy_capture": dc, "soc_alignment": sa}
