"""P9：pack 5 高电流误报案例（回应 §9 待办 7）。论文用 pack 5 第 94 天的高电流脉冲误报批评
GDI 基线（Fig.14/15）；本脚本问同一个问题，但对象换成**论文自己的固定阈值** —— 它的「训练集
LOF 的 99 分位」在同一类工况下会不会也被击穿？

⚠ 一处必须说清的口径差异：论文的「第 94 天」从**投运日**算起，我们的数据没有投运日期（见
eval/metrics.py 的口径声明），只能用「距本包首条记录的天数」。两者不是同一个零点，故不能
声称「复现了第 94 天那一刻」，能复现的只是**机制**（高电流脉冲窗口是否把分数推过固定阈值、
``pack_baseline`` 能否把它压回去）；此区分必须写进报告。实测结论（跑一次即得，此处不复述数字、
以免与运行结果脱节）见脚本输出与 ``outputs/tables/pack5_case.csv``。

用法：``python scripts/09_pack5_case.py [--tag TAG]``
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batfd import config, console, robust  # noqa: E402
from batfd.data import cache, conditions as cond_mod  # noqa: E402
from batfd.detect import lof as lof_mod  # noqa: E402
from batfd.eval import metrics as metrics_mod  # noqa: E402

console.setup()

PACK = 5
DATASET = "StandTestData2"


def pack_baseline_stat(score, ids, *, baseline_frac, min_windows=30):
    """逐包自身基线标准化，与 05_detect / 06_far_dualtrack 同口径。"""
    stat = np.full(np.asarray(score).shape, np.nan, dtype=np.float64)
    for pid in np.unique(ids):
        m = ids == pid
        s = np.asarray(score)[m]
        w0 = max(min_windows, int(round(baseline_frac * len(s))))
        w0 = min(w0, len(s) - 1) if len(s) > 1 else len(s)
        base = s[:w0].reshape(-1, 1)
        med = robust.robust_center(base, axis=0)
        scale = robust.robust_scale(base, axis=0, center=med, floor_rel=1e-3, floor_abs=1e-6)
        stat[m] = robust.zscore(s.reshape(-1, 1), med, scale, axis=0).ravel()
    return stat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lfaae", "ours"], default="lfaae")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--top", type=int, default=15, help="列出分数最高的多少个窗口")
    args = ap.parse_args()

    cfg = config.load()
    tag = args.tag or args.model
    q = float(cfg["detect"]["threshold_q"])
    persistence = int(cfg["detect"]["persistence_windows"])
    bf = float(cfg["detect"]["pack_baseline_frac"])

    print("=" * 78)
    print(f"P9 pack 5 高电流误报案例：{tag}")
    print("=" * 78)
    print("口径：天数是「距本包首条记录」，与论文的投运日**零点不同**，")
    print("      因此只复现机制、不声称复现论文的绝对第 94 天。\n")

    feats = {
        n: np.asarray(np.load(f"outputs/runs/{tag}/recon/{n}/feat.npy", mmap_mode="r"))
        for n in ("StandTrainData", DATASET)
    }
    c = cache.load_cache(cfg, DATASET)
    ids = np.asarray(c["ids"])
    time = np.asarray(c["time"])
    cond = np.asarray(c["cond"])
    names = list(cond_mod.CONDITION_NAMES)

    det = lof_mod.LOFDetector(
        n_neighbors=int(cfg["detect"]["lof_n_neighbors"]),
        mode="train_novelty", standardize=True,
        norm_floor=float(cfg["model"]["norm_floor"]),
    )
    det.fit(feats["StandTrainData"])
    s_tr = det.score(feats["StandTrainData"]).score
    thr = float(np.quantile(s_tr, q))
    s_all = det.score(feats[DATASET]).score

    order = metrics_mod.pack_order(c, PACK)
    days = metrics_mod.days_from_start(time[order])
    s = s_all[order]
    stat = s - thr

    # 固定阈值
    conf = lof_mod.apply_persistence(stat > 0.0, persistence)
    ai = lof_mod.first_alarm_index(conf)
    print(f"[固定阈值] 训练集 {q:.2f} 分位 = {thr:.4f}")
    print(f"  pack 5 报警下标 {ai}，第 {days[ai]:.2f} 天"
          f"（共 {len(days)} 窗口，跨度 {days[-1]:.1f} 天）")
    print(f"  启动前越限窗口 {int((stat[:ai] > 0).sum())} 个")

    # pack_baseline
    tr_ids = np.asarray(cache.load_cache(cfg, "StandTrainData")["ids"])
    z = pack_baseline_stat(s_all, ids, baseline_frac=bf)
    stat_tr = pack_baseline_stat(s_tr, tr_ids, baseline_frac=bf)
    fin = np.isfinite(stat_tr)
    dec = float(np.quantile(stat_tr[fin], q))
    st2 = z[order] - dec
    conf2 = lof_mod.apply_persistence(st2 > 0.0, persistence)
    ai2 = lof_mod.first_alarm_index(conf2)
    print(f"\n[逐包基线] 训练集 z 的 {q:.2f} 分位 = {dec:.4f}")
    if ai2 is None:
        print("  pack 5 无报警")
    else:
        print(f"  pack 5 报警下标 {ai2}，第 {days[ai2]:.2f} 天")
    print(f"  报警日推迟：{'—' if (ai is None or ai2 is None) else f'{days[ai2] - days[ai]:.1f} 天'}")

    # 窗口级证据：分数最高的那些窗口，工况是什么
    print(f"\n分数最高的 {args.top} 个窗口（看它们是否都是高电流脉冲）：")
    print(f"  {'下标':>6} {'天数':>9} {'LOF分':>8} {'固定判据':>9} {'基线判据':>9} "
          + " ".join(f"{n:>10}" for n in ("mean_abs_i", "p95_abs_i", "disch_frac", "mean_v")))
    top = np.argsort(-s)[: args.top]
    rows = []
    for i in top:
        ci = np.where(order == order[i])[0]
        print(
            f"  {i:>6} {days[i]:>9.2f} {s[i]:>8.3f} {stat[i]:>+9.3f} {st2[i]:>+9.3f} "
            + " ".join(
                f"{cond[order[i], j]:>10.2f}" for j in (0, 2, 3, 4)
            )
        )
        rows.append({
            "model": args.model, "tag": tag, "pack_id": PACK,
            "window_in_pack": int(i),
            "day_from_first_record": float(days[i]),
            "lof_score": float(s[i]),
            "exceedance_fixed": float(stat[i]),
            "exceedance_pack_baseline": float(st2[i]),
            "mean_abs_i": float(cond[order[i], 0]),
            "p95_abs_i": float(cond[order[i], 2]),
            "disch_frac": float(cond[order[i], 3]),
            "mean_v": float(cond[order[i], 4]),
            "flagged_fixed": bool(stat[i] > 0),
            "flagged_pack_baseline": bool(st2[i] > 0),
        })

    # 相关性：分数跟不跟电流走
    r_pack = float(np.corrcoef(s, cond[order, 0])[0, 1])
    r_all = float(np.corrcoef(s_all, cond[:, 0])[0, 1])
    print(f"\ncorr(LOF 分数, mean_abs_i)：pack 5 内 {r_pack:+.3f}；整个 test2 {r_all:+.3f}")
    print("  接近 0 说明：阈值击穿**不是**「电流大 ⇒ 分数高」的简单单调关系，")
    print("  而是这些脉冲窗口落在训练分布的稀疏区、LOF 距离整体抬高所致。")

    # 把报警前那些越限窗口单独列出来
    pre = np.flatnonzero(stat > 0)[: max(args.top, 10)]
    print(f"\n固定阈值下最早的 {len(pre)} 个越限窗口：")
    for i in pre:
        tag_i = "连续" if (i > 0 and stat[i - 1] > 0) else "孤立"
        print(f"  下标 {i:>5} 第 {days[i]:>8.2f} 天  {tag_i}  "
              f"mean_abs_i {cond[order[i], 0]:>7.2f}  p95 {cond[order[i], 2]:>7.2f}  "
              f"disch_frac {cond[order[i], 3]:>5.2f}")

    tables = Path(cfg["paths"]["outputs_dir"]) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    out = tables / "pack5_case.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n案例表：{out}（{len(rows)} 行）")

    # 完整逐窗口序列（供画时间线用）：上面那张表只有分数**前 N 名**、不含最早越限的
    # 那个窗口（分数低），画不出报警时刻；这里存全序列，避免绘图脚本重写一遍 LOF 计算。
    # ⚠ `flagged_*` 是**单窗越限**，`confirmed_*` 才是加了持久性规则后的**确认报警**。
    # 报告里写的"第 X 天报警"是后者，画图必须用后者，否则图上的线比表里的数字早。
    series = tables / "pack5_series.csv"
    cols = ("window_in_pack", "day_from_first_record", "lof_score",
            "exceedance_fixed", "exceedance_pack_baseline",
            "mean_abs_i", "p95_abs_i", "disch_frac", "mean_v",
            "flagged_fixed", "flagged_pack_baseline",
            "confirmed_fixed", "confirmed_pack_baseline")
    with series.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols))
        w.writeheader()
        for i in range(len(days)):
            w.writerow({
                "window_in_pack": int(i),
                "day_from_first_record": float(days[i]),
                "lof_score": float(s[i]),
                "exceedance_fixed": float(stat[i]),
                "exceedance_pack_baseline": float(st2[i]),
                "mean_abs_i": float(cond[order[i], 0]),
                "p95_abs_i": float(cond[order[i], 2]),
                "disch_frac": float(cond[order[i], 3]),
                "mean_v": float(cond[order[i], 4]),
                "flagged_fixed": bool(stat[i] > 0),
                "flagged_pack_baseline": bool(st2[i] > 0),
                "confirmed_fixed": bool(conf[i]),
                "confirmed_pack_baseline": bool(conf2[i]),
            })
    print(f"完整序列：{series}（{len(days)} 行）")
    if ai is not None:
        print(f"  （确认报警首发下标：固定阈值 {ai}；逐包基线 "
              f"{'无' if ai2 is None else ai2}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
