"""故障起点标签（Tier A）：从 Hiddall 的单体内阻轨迹推出「什么时候开始坏」。

论文唯一的评价指标是「第几天报警」，没有起点真值时不可证伪（**无限敏感的检测器第 0 天
就报警、直接拿第一**）；本模块用 Hiddall 逐单体欧姆内阻作物理参照定义起点，使 lead time、
检出率、虚警率变得可用。

⚠ 循环性防护（硬约束）：检测器**永远不接触 Hiddall**，检测只用电压/电流/温度的重建误差
与电压偏差；标签来自 Hiddall、检测不来自 Hiddall，两者物理量不同、数据通路不同。
调用方必须保证这一点，本模块只负责生成标签。

⚠ 覆盖范围：Hiddall 只存在于 train 与 test1，因此**只有 pack 6 / 8 / 9 / 10 有起点标签**；
test2 / test3 的 10 个包无内阻轨迹，只能用于「单体身份」（Tier B，见
configs/fault_table.yaml）与虚警率统计。

⚠ 时序拼接：训练集没有时间变量，只能假设「train 段在前、test1 段在后」并拼接；
:func:`continuity_check` 用拼接处内阻跳变幅度检验该假设，跳变过大则该包标签不可信，
调用方应据 :func:`build_pack_series` 返回的 ``boundary_jump`` 决定是否剔除。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAD_SCALE = 1.4826  # 正态下 MAD → σ 的一致性系数


@dataclass
class PackSeries:
    """一个电池包按时间顺序拼好的内阻序列。"""

    pack_id: int
    res: np.ndarray          # (N, 8) 单体内阻（mΩ，推断量纲）
    time: np.ndarray | None  # (N,) datetime64[ns] 或 None
    origin: np.ndarray       # (N,) 字符串数组，'train' / 'test1'
    n_train: int             # 拼接点位置
    boundary_jump: np.ndarray  # (8,) 每个单体在拼接处的相对跳变
    time_sorted: bool        # test1 段是否成功按时间排序

    @property
    def n(self) -> int:
        return int(self.res.shape[0])


def robust_scale(
    x: np.ndarray, floor_rel: float = 1e-3, floor_abs: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """稳健中心与尺度：median 与 1.4826·MAD，并对尺度加下限截断。

    用中位数/MAD 而非均值/标准差：待检出的异常本身会抬高均值和标准差，导致阈值虚高、
    真异常漏检。

    ⚠ 下限必须同时给绝对量：只给 ``floor_rel·|median|`` 在 median 近零时几乎失效，而本
    模块的「跨单体偏差」恰以 0 为中心，实测会把单体尺度估成 ~0，于是任何噪声都判为越限
    （曾出现 onset=0 的假结果）；调用方应同时传 ``floor_abs``（如典型内阻的 5%）。
    """
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * MAD_SCALE
    floor = np.maximum(floor_rel * np.abs(med), floor_abs)
    return med, np.maximum(mad, floor)


def build_pack_series(cfg: dict, caches: dict[str, dict]) -> dict[int, PackSeries]:
    """把 train 段与 test1 段按时间拼成每个包的完整内阻序列。

    只处理同时出现在两个文件里的 ID（即 pack 6/8/9/10）。
    """
    hid_cfg = cfg["hidden"]
    res_cols = [c - 1 for c in hid_cfg["resistance_channels"]]  # -> 0-based

    train = caches.get("StandTrainData")
    test1 = caches.get("StandTestData1")
    if train is None or test1 is None:
        raise KeyError("需要 StandTrainData 与 StandTestData1 的缓存才能构造起点标签")
    if train["hidden"] is None or test1["hidden"] is None:
        raise KeyError("两个文件的 Hiddall 都必须存在")

    out: dict[int, PackSeries] = {}
    common_ids = sorted(set(np.unique(train["ids"]).tolist()) & set(np.unique(test1["ids"]).tolist()))

    for pid in common_ids:
        tr_idx = np.flatnonzero(train["ids"] == pid)
        te_idx = np.flatnonzero(test1["ids"] == pid)

        # test1 段：有时间就按时间排序（实测时间轴非单调，必须排）
        time_sorted = False
        if test1["time"] is not None:
            t = np.asarray(test1["time"])[te_idx]
            order = np.argsort(t, kind="stable")
            te_idx = te_idx[order]
            time_sorted = True

        res = np.vstack(
            [
                np.asarray(train["hidden"])[tr_idx][:, res_cols],
                np.asarray(test1["hidden"])[te_idx][:, res_cols],
            ]
        ).astype(np.float64)

        t_all = None
        if test1["time"] is not None:
            t_te = np.asarray(test1["time"])[te_idx]
            t_all = np.concatenate([np.full(len(tr_idx), np.datetime64("NaT"), "datetime64[ns]"), t_te])
        origin = np.array(["train"] * len(tr_idx) + ["test1"] * len(te_idx))

        n_tr = len(tr_idx)
        # 拼接处跳变：用拼接点前后各 5 个窗口的中位数之差，除以训练段整体标准差
        w = min(5, n_tr, len(te_idx))
        if w > 0:
            before = np.median(res[n_tr - w : n_tr], axis=0)
            after = np.median(res[n_tr : n_tr + w], axis=0)
            denom = np.maximum(np.std(res[:n_tr], axis=0), 1e-12)
            jump = np.abs(after - before) / denom
        else:
            jump = np.full(res.shape[1], np.nan)

        out[int(pid)] = PackSeries(
            pack_id=int(pid),
            res=res,
            time=t_all,
            origin=origin,
            n_train=n_tr,
            boundary_jump=jump,
            time_sorted=time_sorted,
        )
    return out


def onset_grid(
    series: PackSeries,
    cfg: dict,
    *,
    baseline_frac: float | None = None,
    k_grid: list[float] | None = None,
    p_grid: list[int] | None = None,
) -> list[dict]:
    """对 (k, P) 网格算故障起点，返回逐行结果。

    单体「相对偏差」``d_c(w) = r_c(w) − median_{c'} r_{c'}(w)``（自己的内阻减同包
    其他单体的中位数）能自动吸收整包随温度/老化的共同漂移，只留下「这个单体相对
    别人跑偏了」的部分。

    起点 = 该单体偏差首次连续 P 个窗口超过 ``median + k·MAD``（基线只用正常期前段）；
    包起点 = 各单体起点的最小值。
    """
    ocfg = cfg["onset"]
    baseline_frac = baseline_frac if baseline_frac is not None else ocfg["baseline_frac"]
    k_grid = k_grid or list(ocfg["k_grid"])
    p_grid = p_grid or list(ocfg["p_grid"])
    floor_frac = float(ocfg.get("sigma_floor_frac", 0.05))

    res = series.res
    n, n_cells = res.shape
    w0 = max(2, int(round(baseline_frac * n)))

    dev = res - np.median(res, axis=1, keepdims=True)      # (N, 8) 相对偏差

    # 下限取典型内阻的一个比例；dev 与 res 同量纲（mΩ），可直接用
    floor_abs = floor_frac * float(np.median(res))
    mu, sigma = robust_scale(dev[:w0], floor_abs=floor_abs)

    rows: list[dict] = []
    for k in k_grid:
        thresh = mu + k * sigma
        exceed = dev > thresh[None, :]                     # (N, 8)
        # 起点只可能在基线期之后：基线期还没建立「正常」参照，在那里报起点会让
        # 「某单体一直偏高」被误判成在第 0 窗发生故障
        exceed[:w0] = False
        for p in p_grid:
            onsets = [_first_run(exceed[:, c], p) for c in range(n_cells)]
            valid = [o for o in onsets if o is not None]
            rows.append(
                {
                    "pack_id": series.pack_id,
                    "k": float(k),
                    "persistence": int(p),
                    "baseline_windows": int(w0),
                    "sigma_floor_abs": floor_abs,
                    "onset_cells": onsets,
                    "onset_cells_pack_min": int(min(valid)) if valid else None,
                }
            )
    return rows


def onset_summary(
    rows: list[dict], pack_id: int, *, min_valid_frac: float = 0.5
) -> dict:
    """把网格结果归约成「逐单体 + 整包」的点估计与不确定区间。

    两级稳健化，都是被实测的失败模式逼出来的：

    1. **不能直接对网格取 min 当点估计**：越松的 (k, P) 起点越早，取 min 等于永远选最松
       设定，一个噪声单体就能把包起点拉早（实测出现过跨 11000 窗口的区间）。改为每个单体
       先在网格上取**中位数**，再在单体间取最小 —— 「包起点 = 第一个开始坏的单体」。
    2. **只采信在足够多超参组合下稳定出现的起点**：实测有单体只在 2/16 或 3/16 组合下才
       触发，更可能是阈值边界的偶然而非真实退化。这类归入 ``unstable_cells``，不参与包
       起点，但**完整保留在输出里**以便复核（不静默丢弃）。

    ``onset_point`` 用稳定单体；若无一稳定，退回全部单体并把 ``stable_only`` 标为 False。
    """
    n_cells = len(rows[0]["onset_cells"])
    n_total = len(rows)
    per_cell: dict[int, dict] = {}
    for c in range(n_cells):
        vals = [r["onset_cells"][c] for r in rows if r["onset_cells"][c] is not None]
        if not vals:
            per_cell[c + 1] = {
                "n_valid": 0, "n_total": n_total, "min": None, "max": None, "median": None
            }
        else:
            per_cell[c + 1] = {
                "n_valid": len(vals),
                "n_total": n_total,
                "min": int(np.min(vals)),
                "max": int(np.max(vals)),
                "median": int(np.median(vals)),
            }

    def _agg(keys: list[int]) -> tuple[int | None, int | None, int | None]:
        meds = [per_cell[k]["median"] for k in keys if per_cell[k]["median"] is not None]
        early = [per_cell[k]["min"] for k in keys if per_cell[k]["min"] is not None]
        late = [per_cell[k]["max"] for k in keys if per_cell[k]["max"] is not None]
        return (
            int(min(meds)) if meds else None,
            int(min(early)) if early else None,
            int(min(late)) if late else None,
        )

    threshold = min_valid_frac * n_total
    stable = [k for k, v in per_cell.items() if v["n_valid"] >= threshold]
    unstable = [k for k, v in per_cell.items() if 0 < v["n_valid"] < threshold]
    no_onset = [k for k, v in per_cell.items() if v["n_valid"] == 0]

    stable_only = True
    keys = stable
    if not keys:
        keys = [k for k, v in per_cell.items() if v["n_valid"] > 0]
        stable_only = False
    point, early, late = _agg(keys)

    return {
        "pack_id": pack_id,
        "n_cells_with_onset": len([k for k, v in per_cell.items() if v["n_valid"] > 0]),
        "stable_cells": stable,
        "unstable_cells": unstable,
        "no_onset_cells": no_onset,
        "min_valid_frac": min_valid_frac,
        "stable_only": stable_only,
        "onset_point": point,
        "onset_early": early,
        "onset_late": late,
        "per_cell": per_cell,
    }


def _first_run(mask: np.ndarray, p: int) -> int | None:
    """首个「连续 p 个 True」的起始下标；不存在则 None。"""
    if p <= 0:
        raise ValueError("persistence 必须为正")
    if mask.size < p:
        return None
    # 滑动计数（cumsum 差分，O(N)）：counts 长度 N-p+1，第 i 项 = mask[i:i+p] 的和
    cs = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    counts = cs[p:] - cs[:-p]
    hit = np.flatnonzero(counts == p)
    return int(hit[0]) if hit.size else None


def onset_range(rows: list[dict]) -> dict:
    """把网格结果归约成区间，用于报告「起点不是一个点而是一个范围」。"""
    vals = [r["onset_pack"] for r in rows if r["onset_pack"] is not None]
    if not vals:
        return {"n_valid": 0, "min": None, "max": None, "median": None}
    return {
        "n_valid": len(vals),
        "n_total": len(rows),
        "min": int(np.min(vals)),
        "max": int(np.max(vals)),
        "median": float(np.median(vals)),
    }


def continuity_check(series: PackSeries, threshold: float = 3.0) -> dict:
    """检验 train→test1 拼接假设：拼接处内阻跳变是否过大。

    ``boundary_jump`` = 拼接点前后各 5 窗口的中位数之差 ÷ 训练段标准差。
    某单体超过 ``threshold`` 倍即说明两段不连续，该包时序拼接不可信 ——
    不应使用它的起点标签。
    """
    j = series.boundary_jump
    bad = np.flatnonzero(np.isfinite(j) & (j > threshold))
    return {
        "pack_id": series.pack_id,
        "n_train": series.n_train,
        "jump_max": float(np.nanmax(j)) if np.isfinite(j).any() else float("nan"),
        "jump_cells": j.round(3).tolist(),
        "suspect_cells": [int(c) + 1 for c in bad],
        "ok": bool(bad.size == 0),
    }
