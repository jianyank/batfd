"""异常判定阈值：论文的固定分位阈值 vs 工况自适应阈值（轴线③）。

论文用**固定**阈值：训练集 LOF 分数的 99 分位（§2.4）。它自己举反例批评 GDI 基线在
pack 5 第 94 天高电流脉冲下误报（Fig.14/15），却没意识到固定阈值有同一结构性缺陷 ——
故障指标（电压偏差、重建误差）对工况高度敏感，高电流下即便电池完全正常，指标也会
整体抬高而越限。

本模块把「分数高」分解为「工况本来就容易高分」与「真的异常」两部分，用条件分位数
q(s|u) 作该工况的正常上界，判据取**超出量**：

    exceedance = score − q̂_q(condition)    异常 ⟺ exceedance > 0

三种实现统一返回 ``ThresholdModel``（``stat`` 越大越异常），便于同一评价协议下比较：

``fixed``               论文原做法：训练集分数的 q 分位，不分工况（对照基线）。
``quantile_regression`` 梯度提升分位数回归在全部工况协变量上拟合 q(s|u)，本项目默认方法。
``binned_quantile``     按「与分数最相关的单一工况」分位分箱，箱内取 q 分位，再对箱边界
                        做单调平滑；更简单易解释，用来验证梯度提升没有过拟合工况。

q 默认 0.99（与论文一致），三种方法同名参数口径相同。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

METHODS = ("fixed", "quantile_regression", "binned_quantile")


@dataclass
class ThresholdModel:
    """拟合好的阈值模型。``stat`` 统一是「越大越异常」，``decision`` 为阈值。"""

    method: str
    decision: float
    payload: dict = field(default_factory=dict)

    def stat(self, score: np.ndarray, cond: np.ndarray | None = None) -> np.ndarray:
        """把原始分数换算成该方法的异常统计量。"""
        score = np.asarray(score, dtype=np.float64)

        if self.method == "fixed":
            return score - self.decision

        if self.method == "quantile_regression":
            if cond is None:
                raise ValueError("quantile_regression 需要工况协变量")
            qhat = self.payload["model"].predict(np.asarray(cond, dtype=np.float64))
            return score - qhat

        if self.method == "binned_quantile":
            if cond is None:
                raise ValueError("binned_quantile 需要工况协变量")
            qhat = self._binned_predict(cond)
            return score - qhat

        raise ValueError(f"未知方法 {self.method}")

    def _binned_predict(self, cond: np.ndarray) -> np.ndarray:
        j = self.payload["dim"]
        edges = self.payload["edges"]
        qs = self.payload["bin_q"]
        x = np.asarray(cond, dtype=np.float64)[:, j]
        # 落在箱外的点用最近箱（np.digitize 的边界约定需要处理）
        b = np.clip(np.digitize(x, edges, right=False) - 1, 0, len(qs) - 1)
        return qs[b]


def fit_fixed(score_train: np.ndarray, q: float = 0.99) -> ThresholdModel:
    """论文的做法：训练集分数的 q 分位，不分工况。"""
    s = np.asarray(score_train, dtype=np.float64)
    return ThresholdModel("fixed", float(np.quantile(s, q)), {"q": q})


def fit_quantile_regression(
    score_train: np.ndarray,
    cond_train: np.ndarray,
    q: float = 0.99,
    *,
    random_state: int = 42,
    max_iter: int = 300,
) -> ThresholdModel:
    """梯度提升分位数回归拟合 ``q(score | condition)``。

    用梯度提升而不是线性分位数回归：故障指标与工况的关系明显非线性
    （高电流、低温下误差的非线性放大），线性模型会系统性低估极端工况下的上界。
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    s = np.asarray(score_train, dtype=np.float64)
    c = np.asarray(cond_train, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError(f"工况应为 (N, K)，收到 {c.shape}")
    finite = np.isfinite(s) & np.isfinite(c).all(axis=1)
    if finite.sum() < 100:
        raise ValueError(f"可用于拟合的样本只有 {int(finite.sum())} 个，太少")

    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=max_iter, random_state=random_state
    )
    model.fit(c[finite], s[finite])
    return ThresholdModel(
        "quantile_regression",
        0.0,  # 判据是「超出 0」，阈值即 0
        {"model": model, "q": q, "n_fit": int(finite.sum())},
    )


def fit_binned_quantile(
    score_train: np.ndarray,
    cond_train: np.ndarray,
    q: float = 0.99,
    *,
    n_bins: int = 10,
    dim: int | None = None,
) -> ThresholdModel:
    """按单一工况分位分箱，箱内取 q 分位，再对箱分位做单调平滑。

    ``dim`` 为 None 时自动选「与分数秩相关最强」的那一维 —— 用 Spearman
    而不是 Pearson，因为故障指标与工况的关系常常单调但非线性。
    """
    from scipy.stats import spearmanr

    s = np.asarray(score_train, dtype=np.float64)
    c = np.asarray(cond_train, dtype=np.float64)
    finite = np.isfinite(s) & np.isfinite(c).all(axis=1)
    s, c = s[finite], c[finite]
    if len(s) < n_bins * 10:
        raise ValueError(f"可用于分箱的样本只有 {len(s)} 个，按 {n_bins} 箱分太稀")

    if dim is None:
        rho = [abs(spearmanr(c[:, j], s).statistic) for j in range(c.shape[1])]
        dim = int(np.nanargmax(rho))

    x = c[:, dim]
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.clip(np.digitize(x, edges, right=False), 0, n_bins - 1)
    bin_q = np.array(
        [np.quantile(s[b == i], q) if (b == i).sum() >= 5 else np.nan for i in range(n_bins)]
    )
    # 用累计最大值做单调平滑：工况越苛刻，正常上界不应反而下降
    good = np.isfinite(bin_q)
    if good.sum() >= 2:
        filled = np.interp(np.arange(n_bins), np.flatnonzero(good), bin_q[good])
        bin_q = np.maximum.accumulate(filled)
    elif good.sum() == 1:
        bin_q = np.full(n_bins, bin_q[good][0])
    else:
        bin_q = np.full(n_bins, float(np.quantile(s, q)))

    return ThresholdModel(
        "binned_quantile",
        0.0,
        {"dim": int(dim), "edges": edges, "bin_q": bin_q, "q": q, "n_bins": n_bins},
    )


def fit(method: str, score_train: np.ndarray, cond_train: np.ndarray | None, q: float = 0.99):
    """按方法名分派。"""
    if method == "fixed":
        return fit_fixed(score_train, q)
    if method == "quantile_regression":
        if cond_train is None:
            raise ValueError("quantile_regression 需要工况协变量")
        return fit_quantile_regression(score_train, cond_train, q)
    if method == "binned_quantile":
        if cond_train is None:
            raise ValueError("binned_quantile 需要工况协变量")
        return fit_binned_quantile(score_train, cond_train, q)
    raise ValueError(f"未知方法 {method}，可选 {METHODS}")
