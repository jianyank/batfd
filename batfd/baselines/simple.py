"""可复核的原始电压统计基线。

这些函数只处理逐窗口的单体电压，不依赖深度模型。脚本层负责按电池包排序、
校准阈值和调用评价协议；本模块保持纯数值逻辑，便于单元测试。
"""

from __future__ import annotations

import numpy as np


def _finite_1d(values: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} 应为一维数组，收到 {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} 含 NaN 或无穷值")
    return out


def window_cell_mean(signal: np.ndarray, voltage_cols: list[int] | tuple[int, ...]) -> np.ndarray:
    """计算每个窗口、每个单体的平均电压，返回 (N, n_cells)。"""
    x = np.asarray(signal)
    if x.ndim != 3:
        raise ValueError(f"signal 应为 (N,T,C)，收到 {x.shape}")
    cols = np.asarray(voltage_cols, dtype=np.int64)
    if cols.ndim != 1 or len(cols) < 2 or np.any(cols < 0) or np.any(cols >= x.shape[2]):
        raise ValueError(f"voltage_cols 无效：{voltage_cols}")
    return np.asarray(x[:, :, cols], dtype=np.float32).mean(axis=1, dtype=np.float64)


def peer_spread_from_means(cell_means: np.ndarray) -> np.ndarray:
    """以窗口内单体中位数为中心，取最大绝对偏差。"""
    means = np.asarray(cell_means, dtype=np.float64)
    if means.ndim != 2 or means.shape[1] < 2:
        raise ValueError(f"cell_means 应为 (N,n_cells)，收到 {means.shape}")
    center = np.median(means, axis=1, keepdims=True)
    return np.max(np.abs(means - center), axis=1)


def peer_spread(signal: np.ndarray, voltage_cols: list[int] | tuple[int, ...]) -> np.ndarray:
    """直接从窗口信号计算单体间电压离散分数。"""
    return peer_spread_from_means(window_cell_mean(signal, voltage_cols))


def ewma(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """因果 EWMA；不跨电池包传递状态。"""
    x = _finite_1d(values, "values")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha 应在 (0,1]，收到 {alpha}")
    out = np.empty_like(x)
    state = float(x[0])
    out[0] = state
    for i in range(1, len(x)):
        state = alpha * float(x[i]) + (1.0 - alpha) * state
        out[i] = state
    return out


def cusum(
    values: np.ndarray,
    center: float,
    scale: float,
    slack: float = 0.5,
) -> np.ndarray:
    """单侧上行 CUSUM，检测持续高于正常中心的分数。"""
    x = _finite_1d(values, "values")
    if not np.isfinite(center):
        raise ValueError("center 必须是有限数")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale 必须为正数，收到 {scale}")
    if slack < 0.0 or not np.isfinite(slack):
        raise ValueError(f"slack 必须为非负有限数，收到 {slack}")
    z = (x - float(center)) / float(scale)
    out = np.empty_like(z)
    state = 0.0
    for i, value in enumerate(z):
        state = max(0.0, state + float(value) - float(slack))
        out[i] = state
    return out


def robust_center_scale(values: np.ndarray, floor: float = 1e-8) -> tuple[float, float]:
    """返回中位数与 1.4826*MAD，并施加尺度下限。"""
    x = _finite_1d(values, "values")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError(f"floor 必须为正数，收到 {floor}")
    center = float(np.median(x))
    mad = float(np.median(np.abs(x - center)))
    return center, max(1.4826 * mad, float(floor))
