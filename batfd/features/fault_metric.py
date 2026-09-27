"""论文 §2.3 的多维故障指标。

论文定义了三个逐单体指标（Eq.5–8 与 §2.3.3），再 concatenated across all cells
to form a unified pack-level feature vector，交给 LOF。

======================  ==========================================  ==========
指标                     定义（论文式号）                              方向
======================  ==========================================  ==========
sigma_v                 单体电压偏差的标准差（Eq.5–7）                  越大越差
cos_sim                 实测与重建电压的余弦相似度（Eq.8）              越小越差
q3_err                  电压重建误差绝对值的第三四分位数（§2.3.3）      越大越差
======================  ==========================================  ==========

两处刻意的实现声明（论文未说明，必须如实记录而不是假装照做）：

1. **Eq.7 用的是有符号偏差的标准差** ``σ = sqrt(Σ(V_E − V̄_E)²/(L−1))``，而旧 MATLAB
   实现 ``+bat/residualFeatures.m`` 算的是 ``std(|dev|)`` —— 绝对值的标准差不是同一个
   统计量。本实现按论文原式，用**有符号**偏差。
2. **论文没有说 Z 是否标准化就直接喂给 LOF**。24 维指标里三个量纲完全不同（伏特、
   无量纲余弦、伏特），而 LOF 用欧氏距离，不标准化会让量纲大的维度主导距离。本实现
   先做稳健标准化（仅用训练集拟合）作为我们方法的一部分；对论文基准则两种都跑，
   以便看出这个未声明的选择影响多大。

方向统一：``orient=True`` 的版本把三个指标都转成「越大越异常」（余弦取 ``1 − CS``），
使后续打分、逐单体聚合、阈值比较方向一致；同时返回原始值，出图与报告用原始量纲。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METRIC_NAMES = ("sigma_v", "cos_sim", "q3_err")


@dataclass
class FaultMetrics:
    """逐窗口逐单体指标，均为 ``(N, n_cells)``。"""

    sigma_v: np.ndarray   # 越大越差
    cos_sim: np.ndarray   # 越小越差（原始方向，未翻转）
    q3_err: np.ndarray    # 越大越差
    v_mean_pack: np.ndarray  # (N, T) 包内平均电压，出图与复核用

    def oriented(self) -> np.ndarray:
        """转成统一「越大越差」方向并拼成 ``(N, 3*n_cells)`` 包级向量。

        拼接顺序为**单体优先**：``[σ_1, 1−CS_1, Q3_1, σ_2, 1−CS_2, Q3_2, ...]``。
        论文只说「concatenated across all cells」，未给顺序。顺序对 LOF 无影响
        （欧氏距离对坐标置换不变），但会影响逐单体聚合的切片，所以定死并记录。
        """
        n, c = self.sigma_v.shape
        out = np.empty((n, 3 * c), dtype=np.float64)
        out[:, 0::3] = self.sigma_v
        out[:, 1::3] = 1.0 - self.cos_sim
        out[:, 2::3] = self.q3_err
        return out

    def per_cell_blocks(self, n_cells: int) -> list[slice]:
        """每个单体在包级向量里占的三个切片。"""
        return [slice(3 * c, 3 * c + 3) for c in range(n_cells)]


def paper_metrics(
    v_meas: np.ndarray, v_rec: np.ndarray, *, ddof: int = 1
) -> FaultMetrics:
    """算论文三个指标。

    ``v_meas``/``v_rec``：``(N, n_cells, T)`` 实测与重建的单体电压，**必须同一量纲**
    （本项目缓存是原始缩放值，两者一致，故指标可直接用；换算成伏特只影响出图坐标标签）。
    返回的 ``sigma_v``/``q3_err`` 用有符号偏差 / 绝对误差，见模块 docstring。
    """
    if v_meas.shape != v_rec.shape:
        raise ValueError(f"形状不一致：{v_meas.shape} vs {v_rec.shape}")
    if v_meas.ndim != 3:
        raise ValueError(f"应为 (N, n_cells, T)，收到 {v_meas.shape}")

    # Eq.5–6：包内跨单体平均电压，再取每个单体相对它的偏差
    v_mean = v_meas.mean(axis=1, keepdims=True)          # (N, 1, T)
    dev = v_meas - v_mean                                # (N, C, T)
    # Eq.7：有符号偏差的样本标准差（1/(L−1)）
    sigma_v = dev.std(axis=2, ddof=ddof)                 # (N, C)

    # Eq.8：实测与重建电压序列的余弦相似度
    num = np.sum(v_meas * v_rec, axis=2)
    den = np.linalg.norm(v_meas, axis=2) * np.linalg.norm(v_rec, axis=2)
    cos_sim = np.divide(num, den, out=np.ones_like(num), where=den > 0)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # §2.3.3：|重建误差| 的第三四分位数
    q3_err = np.percentile(np.abs(v_meas - v_rec), 75, axis=2)

    return FaultMetrics(
        sigma_v=sigma_v,
        cos_sim=cos_sim,
        q3_err=q3_err,
        v_mean_pack=v_mean[:, 0, :],
    )


def cell_anomaly_scores(
    metrics: FaultMetrics, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """逐单体异常分，用于单体定位评估（论文从未报告过这个能力）。

    ``center``/``scale`` 形状 ``(3, n_cells)``：三个指标各自的稳健中心与尺度，
    **只用训练集拟合**。做法是把三个指标各自标准化到「越大越异常」方向后相加
    （等价于旧 MATLAB 实现 ``applyScore`` 的 ``sum|z|`` 思路，但这里方向已经统一，
    所以不需要取绝对值）。

    Returns
    -------
    (N, n_cells) 每个窗口每个单体的异常分。
    """
    z = np.stack(
        [
            (metrics.sigma_v - center[0]) / scale[0],
            ((1.0 - metrics.cos_sim) - center[1]) / scale[1],
            (metrics.q3_err - center[2]) / scale[2],
        ],
        axis=0,
    )  # (3, N, C)
    return z.sum(axis=0)
