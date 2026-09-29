"""评价指标：提前量、检出率、虚警率、单体定位。

先明确三条口径（论文在这三点上都没交代清楚，而它们直接决定数字怎么读）：

1. **「第几天」用我们数据里的相对天数，不是论文的绝对运行天数。** 论文的
   day of operation 从投运日算起，而投运日期未知（数据包里没有），且我们的数据是
   论文数据的子集（截止 2019-11，论文称到 2022-03），绝对天数**无法复现**。这里一律
   用「距本包首条记录的天数」，且所有方法在**同一批窗口**上比较 —— 方法间的相对提前量
   因此公平，而绝对数字不与论文表格直接可比（报告中必须写明）。
2. **起点之前只统计触发率，不称为真虚警率。** 起点标签前可能已有退化；
   真虚警率只在独立已知正常期上估计，见 eval/dualtrack.py 的轨 A。
3. **报警 = 连续 m 个窗口越限**（持久性规则）。论文的「检测日」隐含了某种此类规则但未
   说明具体形式；把它显式化并让所有方法共用，比较才公平。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np

from ..detect.lof import apply_persistence, first_alarm_index


@dataclass
class PackEval:
    """单个电池包的评价结果。"""

    pack_id: int
    n_windows: int
    alarm_index: int | None
    alarm_day: float | None
    onset_index: int | None          # 本包窗口序列内的下标
    onset_day: float | None
    lead_days: float | None          # onset_day − alarm_day，正数表示报警早于起点
    n_confirmed_before_onset: int    # 起点之前的确认报警窗口数
    n_windows_before_onset: int
    trigger_rate_before_onset: float
    alarm_ever: bool                 # 任何时刻有确认报警，包括晚于起点
    early_detected: bool | None      # 严格早于起点；无有效起点标签时未知
    notes: list[str] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        """仅兼容旧 P5 API：曾报警，不等于提前检出；新 CSV 不再导出此字段。"""
        warnings.warn("detected 仅表示曾报警，请改用 alarm_ever 或 early_detected",
                      DeprecationWarning, stacklevel=2)
        return self.alarm_ever

    @property
    def far_per_window(self) -> float:
        """仅兼容旧 API；此值是起点前触发率，不是真虚警率。"""
        warnings.warn("请使用 trigger_rate_before_onset；真虚警率见轨 A",
                      DeprecationWarning, stacklevel=2)
        return self.trigger_rate_before_onset


def pack_order(cache: dict, pid: int, *, sort_by_time: bool = True) -> np.ndarray:
    """取某包在本数据集内的窗口下标，按时间排序。

    实测：**ID 内的时间轴是严格单调的**（14 个 ID 全部无负间隔），
    但同一文件内 ID 块的先后顺序不按时间排，所以跨 ID 汇总时必须排。
    """
    idx = np.flatnonzero(np.asarray(cache["ids"]) == pid)
    if sort_by_time and cache.get("time") is not None:
        t = np.asarray(cache["time"])[idx]
        idx = idx[np.argsort(t, kind="stable")]
    return idx


def days_from_start(time_sorted: np.ndarray) -> np.ndarray:
    """把时间轴换成「距首条记录的天数」。无时间时退回等间隔的窗口序号。"""
    if time_sorted is None:
        raise ValueError("需要时间轴")
    t = np.asarray(time_sorted).astype("datetime64[s]").astype(np.int64)
    return (t - t[0]) / 86400.0


def evaluate_pack(
    pack_id: int,
    score: np.ndarray,
    threshold: float,
    *,
    persistence: int,
    time_sorted: np.ndarray | None = None,
    onset_index: int | None = None,
    onset_source: str = "",
) -> PackEval:
    """在单个包的窗口序列上算报警、提前量、起点前触发率。

    Parameters
    ----------
    score : (N,) 该包按时间排序后的异常分
    onset_index : 该包序列内故障起点的下标（来自 Tier A 内阻标签）；None 表示无标签
    """
    n = len(score)
    confirmed = apply_persistence(score > threshold, persistence)
    alarm_idx = first_alarm_index(confirmed)

    days = days_from_start(time_sorted) if time_sorted is not None else np.arange(n, dtype=float)
    alarm_day = float(days[alarm_idx]) if alarm_idx is not None else None

    notes: list[str] = []
    onset_day = None
    lead = None
    n_before = 0
    n_win_before = 0

    if onset_index is not None:
        if not 0 <= onset_index < n:
            notes.append(f"起点下标 {onset_index} 超出范围 [0,{n})")
            onset_index = None
        else:
            onset_day = float(days[onset_index])
            lead = onset_day - alarm_day if alarm_day is not None else None
            n_before = int(confirmed[:onset_index].sum())
            n_win_before = int(onset_index)
    if onset_index is None and onset_source:
        notes.append(f"无起点标签（{onset_source}）")

    trigger_rate = (n_before / n_win_before) if n_win_before > 0 else float("nan")
    return PackEval(
        pack_id=pack_id,
        n_windows=n,
        alarm_index=alarm_idx,
        alarm_day=alarm_day,
        onset_index=onset_index,
        onset_day=onset_day,
        lead_days=lead,
        n_confirmed_before_onset=n_before,
        n_windows_before_onset=n_win_before,
        trigger_rate_before_onset=trigger_rate,
        alarm_ever=alarm_idx is not None,
        early_detected=(alarm_idx is not None and alarm_idx < onset_index)
        if onset_index is not None else None,
        notes=notes,
    )


def summarize(evals: list[PackEval]) -> dict:
    """汇总成一组可报告的数字。所有量都只在真有对应标签的包上算，不凑分母。"""
    labelled = [e for e in evals if e.onset_index is not None]
    leads = [e.lead_days for e in labelled if e.lead_days is not None]
    far_win = sum(e.n_windows_before_onset for e in labelled)
    far_hit = sum(e.n_confirmed_before_onset for e in labelled)

    return {
        "n_packs": len(evals),
        "n_packs_with_onset_label": len(labelled),
        "n_alarm_ever": sum(e.alarm_ever for e in evals),
        "n_alarm_ever_labelled": sum(e.alarm_ever for e in labelled),
        "n_early_detected": sum(bool(e.early_detected) for e in labelled),
        "early_detection_rate": (sum(bool(e.early_detected) for e in labelled) / len(labelled))
        if labelled else None,
        # 提前量：只在既有标签又有报警的包上统计
        "n_lead_samples": len(leads),
        "lead_days_median": float(np.median(leads)) if leads else None,
        "lead_days_min": float(np.min(leads)) if leads else None,
        "lead_days_max": float(np.max(leads)) if leads else None,
        "n_lead_positive": sum(l > 0 for l in leads),
        # 起点前触发率：分母是「有标签的包的起点前窗口总数」
        "trigger_rate_before_onset": (far_hit / far_win) if far_win > 0 else None,
        "windows_before_onset": far_win,
        "confirmed_before_onset": far_hit,
    }


def cell_localization(
    cell_scores: np.ndarray,
    expected_cells: list[int],
    *,
    top_k: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """单体定位：已知故障单体是否排进逐单体异常分的前 k 名。

    论文**从未报告过**这个能力 —— 它只给包级报警时间，不回答「是哪个单体坏了」。
    这正是单体分辨潜变量结构的目的所在，所以这里做成可检验的指标。

    Parameters
    ----------
    cell_scores : (n_windows, n_cells) 逐窗口逐单体异常分（已按故障期聚合前的原始值）
    expected_cells : Table 5 给出的故障单体编号（1-based）；空表示该包无法定位
    """
    if not expected_cells:
        return {"usable": False, "reason": "Table 5 未指明单体"}
    mean_score = np.nanmean(cell_scores, axis=0)          # (n_cells,)
    order = np.argsort(-mean_score)
    rank = {int(c) + 1: int(np.flatnonzero(order == c)[0]) + 1 for c in range(len(mean_score))}
    worst = [c for c in expected_cells if c in rank]
    best_rank = min(rank[c] for c in worst) if worst else None
    return {
        "usable": True,
        "expected_cells": expected_cells,
        "ranks": {int(c): rank[int(c)] for c in expected_cells},
        "best_rank_of_expected": best_rank,
        "hit_at_k": {int(k): (best_rank is not None and best_rank <= k) for k in top_k},
        "top3_cells": [int(c) + 1 for c in order[:3]],
        "mean_score_per_cell": [float(x) for x in mean_score],
    }
