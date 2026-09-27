"""稳健统计量：中心与尺度的估计。

只用 ``1.4826·MAD`` 估尺度在**重尾分布**上会严重低估：LOF 分数在纯正常期的训练数据上
用基线段 MAD 标准化后 z 最大值达到 60.5（pack 9），根源是分布主体极度集中、尾部很长
—— MAD 只描述主体，尾部一比就爆，于是阈值形同虚设，虚警率由尾部形状而非阈值决定。

因此同时算几个对不同分位段敏感的稳健估计量并**取最大**：分位段越高，对尾部越容忍、
尺度越大、z 越保守（少误报）。取最大是刻意选的方向 —— 「漏报一点」比「天天误报」
可接受得多，漏报可由后续的持久性规则与灵敏度权衡曲线补回。

分母的常数是正态下的换算系数：MAD→σ 用 1.4826；IQR(=q75−q25)→σ 用 1.349；
(q90−q50)→σ 用 1.2816。
"""

from __future__ import annotations

import numpy as np

MAD_TO_SIGMA = 1.4826
IQR_TO_SIGMA = 1.349
Q90Q50_TO_SIGMA = 1.2816


def robust_center(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """稳健中心：中位数。"""
    return np.median(x, axis=axis)


def robust_scale(
    x: np.ndarray,
    *,
    axis: int = 0,
    floor_rel: float = 1e-3,
    floor_abs: float = 0.0,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """多估计量取最大的稳健尺度。

    Parameters
    ----------
    x : 数据
    floor_rel : 相对下限，乘 ``|center|``
    floor_abs : 绝对下限；数据是「以 0 为中心的偏差」时**必须有**，
        否则 center 近零会让相对下限失效
    center : 若已算过中心可传入，避免重复计算

    返回尺度数组，形状为去掉 ``axis`` 后的形状。
    """
    x = np.asarray(x, dtype=np.float64)
    c = robust_center(x, axis=axis) if center is None else np.asarray(center, dtype=np.float64)

    expand = [slice(None)] * x.ndim
    expand[axis] = None
    cx = c[tuple(expand)]

    mad = np.median(np.abs(x - cx), axis=axis) * MAD_TO_SIGMA
    q25, q75 = np.percentile(x, [25, 75], axis=axis)
    q50, q90 = np.percentile(x, [50, 90], axis=axis)
    iqr_sc = (q75 - q25) / IQR_TO_SIGMA
    q90_sc = (q90 - q50) / Q90Q50_TO_SIGMA

    s = np.maximum.reduce([mad, iqr_sc, q90_sc])
    floor = np.maximum(floor_rel * np.abs(c), floor_abs)
    return np.maximum(s, floor)


def zscore(
    x: np.ndarray, center: np.ndarray, scale: np.ndarray, *, axis: int = 0
) -> np.ndarray:
    """按给定中心与尺度做标准化。尺度为 0 的位置返回 0 而不是 inf。"""
    expand = [slice(None)] * np.asarray(x).ndim
    expand[axis] = None
    c = np.asarray(center)[tuple(expand)]
    s = np.asarray(scale)[tuple(expand)]
    safe = np.where(s > 0, s, 1.0)
    out = (x - c) / safe
    return np.where(s > 0, out, 0.0)
