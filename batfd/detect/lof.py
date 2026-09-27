"""LOF（局部离群因子）异常检测，实现论文 §2.4 的两种用法。

论文的做法（Eq.9–12）：k 距离 → k 距离邻域 → 可达距离 → 局部可达密度 LRD →
LOF 分数 = 邻居 LRD 与本点 LRD 之比的均值；阈值取**正常训练数据 LOF 分布的
99 分位**，超过即判故障。

两种模式（都必须跑，不预设结论，最终选哪个由检出率/虚警率的权衡曲线定）：

- ``train_novelty``：用训练集窗口拟合，再给测试窗口打分，与论文「阈值取自训练集
  分布」一致。⚠ 若整段测试数据处于训练时没见过的工况（温度带、老化阶段整体偏移），
  **所有**窗口都会显得离群，整包从头报警 —— 表现为「密集误报」。
- ``per_pack``：在**每个包自己的窗口序列内**拟合与打分（自参照），工况漂移被包内
  自比吸收。⚠ 缓慢的单体退化会让**整包一起漂移**，相对密度变化很小，真异常被钝化漏检。
  ⚠ 此模式必须**逐包**调用 :meth:`LOFDetector.score_self_referential`，不能把多个包
  拼成一块一次性调用；判定阈值必须用返回结果里的 ``.threshold``（块内 q 分位），
  不能另算 —— 这两种错法都会静默给出错误的报警集合。

两者失效方向相反。

标准化（论文未说明，本项目自行决定，报告中必须声明）：论文没有说是否标准化就直接
喂给 LOF。24 维指标量纲不同（伏特、无量纲、伏特），而 LOF 基于欧氏距离，不标准化会
让量纲大的维度主导距离。因此提供 ``standardize`` 开关，**默认 True**（用训练集拟合
的 median/MAD），并保留 False 以便量化这个未声明的选择带来的影响。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# MAD → σ 的正态一致性常数（MAD_SCALE = 1.4826）
MAD_SCALE = 1.4826


@dataclass
class LOFResult:
    """一次打分的产物。"""

    score: np.ndarray                       # (N,) 越大越异常
    threshold: float | None                 # 判定阈值
    flag: np.ndarray | None = None          # (N,) bool
    meta: dict = field(default_factory=dict)


class LOFDetector:
    """带稳健标准化的 LOF。

    用法：``det = LOFDetector(n_neighbors=20, mode="train_novelty")`` →
    ``det.fit(X_train)`` → ``thr = det.threshold_from_train(q=0.99)`` →
    ``det.score(X_test, threshold=thr)``；``mode="per_pack"`` 用
    :meth:`score_self_referential` 逐包调用。

    ``n_neighbors`` 默认 20；``norm_floor`` 默认 1e-3，稳健尺度下限比例
    （scale = max(MAD·1.4826, norm_floor·|median|, 1e-12)，避免近零尺度放大噪声）。
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        *,
        mode: str = "train_novelty",
        standardize: bool = True,
        norm_floor: float = 1e-3,
    ) -> None:
        if mode not in ("train_novelty", "per_pack"):
            raise ValueError(f"mode 应为 train_novelty 或 per_pack，收到 {mode}")
        self.n_neighbors = int(n_neighbors)
        self.mode = mode
        self.standardize = bool(standardize)
        self.norm_floor = float(norm_floor)

        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.model_: LocalOutlierFactor | None = None
        self._train_score: np.ndarray | None = None

    def _fit_scaler(self, x: np.ndarray) -> None:
        med = np.median(x, axis=0)
        mad = np.median(np.abs(x - med), axis=0) * MAD_SCALE
        floor = np.maximum(self.norm_floor * np.abs(med), 1e-12)
        self.center_ = med
        self.scale_ = np.maximum(mad, floor)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.standardize:
            return x
        if self.center_ is None:
            raise RuntimeError("尚未 fit，标准化统计量不可用")
        return (x - self.center_) / self.scale_

    def fit(self, x_train: np.ndarray) -> "LOFDetector":
        """用训练集特征拟合标准化统计量；（train_novelty 模式下）再拟合 LOF。"""
        x = np.asarray(x_train, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            bad = int((~np.isfinite(x)).sum())
            raise ValueError(f"训练特征含 {bad} 个非有限值，先清理再 fit")

        if self.standardize:
            self._fit_scaler(x)

        if self.mode == "train_novelty":
            n = x.shape[0]
            k = min(self.n_neighbors, n - 1)
            if k < 1:
                raise ValueError(f"训练样本数 {n} 太少，无法拟合 LOF")
            self.model_ = LocalOutlierFactor(n_neighbors=k, novelty=True)
            self.model_.fit(self.transform(x))
            self._train_score = -self.model_.score_samples(self.transform(x))
        return self

    def threshold_from_train(self, q: float = 0.99) -> float:
        """论文的阈值规则：训练集 LOF 分数的 q 分位。"""
        if self._train_score is None:
            raise RuntimeError("只有 train_novelty 模式能给出训练集分数；先 fit")
        return float(np.quantile(self._train_score, q))

    def score(self, x: np.ndarray, *, threshold: float | None = None) -> LOFResult:
        """给新样本打分（train_novelty 模式）。``score`` 越大越异常。"""
        if self.model_ is None:
            raise RuntimeError("mode=train_novelty 需要先 fit(X_train)")
        x = np.asarray(x, dtype=np.float64)
        s = -self.model_.score_samples(self.transform(x))
        flag = None if threshold is None else (s > threshold)
        return LOFResult(score=s, threshold=threshold, flag=flag, meta={"mode": self.mode})

    def score_self_referential(self, x: np.ndarray, q: float = 0.99) -> LOFResult:
        """在给定块内自参照拟合与打分（per_pack 模式）。

        阈值取**块内**分数的 q 分位 —— 注意这与论文「阈值取自训练集」的做法
        语义不同，是模式 (b) 的定义使然，报告中必须说明。
        """
        x = np.asarray(x, dtype=np.float64)
        n = x.shape[0]
        k = min(self.n_neighbors, max(1, n - 1))
        if n < 3:
            return LOFResult(
                score=np.full(n, np.nan),
                threshold=None,
                meta={"mode": "per_pack", "note": f"样本数 {n} 太少"},
            )
        xt = (x - np.median(x, axis=0)) / np.maximum(
            np.median(np.abs(x - np.median(x, axis=0)), axis=0) * MAD_SCALE, 1e-12
        ) if self.standardize else x
        model = LocalOutlierFactor(n_neighbors=k, novelty=False)
        model.fit_predict(xt)
        s = -model.negative_outlier_factor_
        thr = float(np.quantile(s, q))
        return LOFResult(
            score=s,
            threshold=thr,
            flag=(s > thr),
            meta={"mode": "per_pack", "n_neighbors": k, "q": q},
        )


def apply_persistence(flag: np.ndarray, m: int) -> np.ndarray:
    """持久性规则：连续 m 个窗口越限才判定报警。

    论文的「检测日」隐含了某种此类规则但未说明具体形式。显式化之后各方法用
    同一套规则，比较才公平；m 本身作为消融项。
    返回每个窗口位置是否已构成「确认报警」。
    """
    flag = np.asarray(flag, dtype=bool)
    if m <= 1:
        return flag.copy()
    out = np.zeros_like(flag)
    if flag.size < m:
        return out
    cs = np.concatenate([[0], np.cumsum(flag.astype(np.int64))])
    counts = cs[m:] - cs[:-m]          # 长度 N-m+1
    hits = np.flatnonzero(counts == m)  # 每个 m 连击的**末尾**下标 = hits + m - 1
    for h in hits:
        out[h + m - 1] = True
    return out


def first_alarm_index(confirmed: np.ndarray) -> int | None:
    """首个确认报警的窗口下标；无则 None。"""
    idx = np.flatnonzero(confirmed)
    return int(idx[0]) if idx.size else None
