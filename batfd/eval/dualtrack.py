"""双轨虚警率（回应 §5.2）。

原来只有一个数：`起点之前的确认报警数 / 起点之前的窗口数`，叫「虚警率」。但实测该分母
**不是正常期** —— pack 6 的 test1 记录中位数比它自己前 10% 高出 2.6 个尺度单位，即
test1 整段都在退化（论文说的 "subsequent phase of progressive anomaly evolution"）；
起点之前的报警很可能是**真阳性**（电压信号早于内阻穿越阈值），算成虚警会系统性高估。

两轨的定义
----------
**轨 A —— 真虚警率**：分母 = 训练集（``StandTrainData``）里的**已知正常期**窗口。
train 段是论文明确标注的正常运行期，起点搜索（见 eval/onset.py）也只在 train 段之后
找起点，说明 train 段在项目内部同样被当作正常。
**轨 B —— 起点前触发率**：分母 = test1 里、故障起点之前的窗口。沿用现有定义，但改名
「触发率」而不是「虚警率」，因为那些触发有相当一部分是真阳性。

两轨分开算、分开命名、分开报告，**绝不合并成一个数**；且两轨分母场景不同（train 段 vs
test1 段）、窗口数也不同，因此两轨数值**不可直接相减**，只能各自横向比较方法。

⚠ 轨 A 的两个实现陷阱（都实测踩过）
----------------------------------
**陷阱 1：不能拿 LOF 给自己训练过的点打分。** ``LocalOutlierFactor(novelty=True)`` 对
**训练点本身**打分是乐观的 —— 实测在训练集上「拟合+打分」单窗越限 266 个，但按 ``m=5``
的持久性规则确认报警为 **0**，即该口径下图 A 恒等于 0，是**退化结果**而非结论。改用留一
（每折用另外三个包拟合、给留出的包打分）后越限率立刻变成 18%–49%，差了一个数量级。
所以轨 A **必须留出**，不能 in-sample。

**陷阱 2：留一也会踩到「参照系错了」的老问题，但两件事必须分清。** 轨 A 的留一测的是
**「这个检测器在没见过的包上，对自己的正常期报不报警」** —— 正是轨 A 想问的问题，
必须保留偶见的跨包偏移；而 ``pack_baseline`` 修的是**在 test1 上**那一侧的参照系问题，
与本模块无关。因此轨 A 的分母是**留出的那个包自己**的训练窗口，参照系（LOF 流形 +
标准化统计量）来自其余三个包 —— 恰好是 ``train_novelty`` 模式在真实使用时的情形。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..detect.lof import apply_persistence, first_alarm_index


@dataclass
class TrackAResult:
    """轨 A：已知正常期上的真虚警率（留一口径）。"""

    pack_id: int
    n_windows: int              # 留出包的正常期窗口数（= 分母）
    n_confirmed: int            # 其中构成「确认报警」（连续 m 窗越限）的窗数
    far_per_window: float
    n_windows_flagged: int      # 单窗越限数（未经持久性规则）
    threshold: float            # 拟合集（其余包）的 q 分位阈值
    median_score: float         # 留出包自身分数的中位数 —— 用来看跨包偏移


@dataclass
class TrackBResult:
    """轨 B：起点前窗口上的触发率（不是虚警率）。"""

    pack_id: int
    n_windows: int
    n_confirmed: int
    trigger_per_window: float


def _confirmed_count(stat: np.ndarray, persistence: int) -> tuple[int, int]:
    """返回（确认报警窗数，单窗越限数）。"""
    flag = np.asarray(stat) > 0.0
    conf = apply_persistence(flag, persistence)
    return int(conf.sum()), int(flag.sum())


def track_a_leave_one_id(
    scores_by_pack: dict[int, np.ndarray],
    thresholds: dict[int, float],
    *,
    persistence: int,
    min_windows: int = 30,
) -> list[TrackAResult]:
    """在已知正常期上算真虚警率，逐包留一。

    Parameters
    ----------
    scores_by_pack : {pack_id: 该包训练窗口的 LOF 分数}
        **必须来自「用其余包拟合的检测器」**，不能是 in-sample 的（见模块 docstring 陷阱 1）。
    thresholds : {pack_id: 该折的判定阈值}
        同样来自其余包（拟合集的 q 分位）；不能用本包自己的分位数 ——
        那会把「本包整体偏移」这个待检出的事实抹掉。
    """
    out: list[TrackAResult] = []
    for pid in sorted(scores_by_pack):
        s = np.asarray(scores_by_pack[pid], dtype=np.float64)
        if s.size < min_windows:
            continue
        thr = float(thresholds[pid])
        stat = s - thr
        n_conf, n_flag = _confirmed_count(stat, persistence)
        out.append(
            TrackAResult(
                pack_id=int(pid),
                n_windows=int(s.size),
                n_confirmed=n_conf,
                far_per_window=n_conf / s.size,
                n_windows_flagged=n_flag,
                threshold=thr,
                median_score=float(np.median(s)),
            )
        )
    return out


def track_b(
    stat_test: np.ndarray,
    *,
    onset_index: int | None,
    persistence: int,
    min_windows: int = 30,
) -> TrackBResult | None:
    """在起点之前的窗口上算触发率。``stat_test`` 须为该包按时间排序后的分数。"""
    if onset_index is None or onset_index < min_windows:
        return None
    s = np.asarray(stat_test)[:onset_index]
    n_conf, _ = _confirmed_count(s, persistence)
    return TrackBResult(
        pack_id=-1,
        n_windows=int(len(s)),
        n_confirmed=n_conf,
        trigger_per_window=n_conf / len(s),
    )


def summarize_track_a(results: list[TrackAResult]) -> dict:
    """轨 A 汇总：分母是全部留出正常期窗口之和（按窗口等权，不按包等权）。"""
    n_win = sum(r.n_windows for r in results)
    n_hit = sum(r.n_confirmed for r in results)
    return {
        "track": "A_known_normal_far_leave_one_id",
        "n_packs": len(results),
        "windows": n_win,
        "confirmed": n_hit,
        "rate": (n_hit / n_win) if n_win else None,
        "per_pack": [r.__dict__ for r in results],
    }


def summarize_track_b(results: list[TrackBResult]) -> dict:
    """轨 B 汇总：分母是全部起点前窗口之和。"""
    n_win = sum(r.n_windows for r in results)
    n_hit = sum(r.n_confirmed for r in results)
    return {
        "track": "B_pre_onset_trigger",
        "n_packs": len(results),
        "windows": n_win,
        "confirmed": n_hit,
        "rate": (n_hit / n_win) if n_win else None,
        "per_pack": [r.__dict__ for r in results],
    }


def first_alarm_day(stat: np.ndarray, days: np.ndarray, *, persistence: int) -> float | None:
    """便捷函数：给定统计量与天数轴，返回首个确认报警的天数。"""
    ai = first_alarm_index(apply_persistence(np.asarray(stat) > 0.0, persistence))
    return None if ai is None else float(np.asarray(days)[ai])
