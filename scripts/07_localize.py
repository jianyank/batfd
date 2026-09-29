"""P7：单体定位评估（回应 §9 待办 5）。论文**从未报告过**这个能力（只给包级报警时间，不
回答「是哪个单体坏了」），而轴线①（单体分辨潜变量）的意义正在于此，故必须做成可检验的指标。

做法：用已缓存重建 ``(v_meas, v_rec)`` 算论文 §2.3 的三个逐单体指标 → 在 **pack 6/8/9/10
的训练（正常）期**上拟合三指标的稳健中心与尺度（只能用训练期，用待检故障期会使尺度被异常
抬高、定位钝化）→ 逐窗口逐单体合成 ``cell_anomaly_scores`` → 在**故障期**取均值聚合，看
真值单体是否排进前 k 名。

⚠ 两个口径必须说清：**(a) 聚合区间** 默认在整段非训练期上聚合，但故障发生很晚的包会被前
面大段正常期稀释，故同时报「全期」与「末段 20%」。**(b) 参照系** ``cell_anomaly_scores``
用三指标各自的稳健 z，其中 ``sigma_v`` 本身是跨单体相对偏差（见 features/fault_metric.py
的 Eq.5-7）、已去掉包级共同漂移，而 ``cos_sim``/``q3_err`` 仍含包级成份；故**两个口径都
报**：原值，以及「减去同窗口跨单体中位数」的版本（后者更接近「哪个单体相对别的单体异常」，
正是定位要问的问题）。

用法：``python scripts/07_localize.py --model {lfaae,ours} [--tag TAG]``
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batfd import experiments, config, console, robust  # noqa: E402
from batfd.data import cache  # noqa: E402
from batfd.eval import metrics as metrics_mod  # noqa: E402
from batfd.features import fault_metric  # noqa: E402

from batfd.models import inference  # noqa: E402

console.setup()

DATASETS = ("StandTestData1", "StandTestData2", "StandTestData3")


def load_fault_table() -> dict:
    p = Path(__file__).resolve().parent.parent / "configs" / "fault_table.yaml"
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def read_labels(cfg) -> dict[int, dict]:
    p = Path(cfg["paths"]["outputs_dir"]) / "tables" / "labels.csv"
    out: dict[int, dict] = {}
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            pid = int(row["pack_id"])

            def _int(k):
                v = row.get(k, "")
                return int(v) if v not in ("", "None", "nan") else None

            out[pid] = {"onset_point": _int("onset_point"), "n_train": _int("n_train")}
    return out


def fit_metric_stats(m: fault_metric.FaultMetrics) -> tuple[np.ndarray, np.ndarray]:
    """在给定（正常期）指标上拟合三指标的稳健中心与尺度，形状均 (3, n_cells)；用多功能
    稳健尺度（见 batfd/robust.py）：单用 MAD 在重尾下低估尺度，使 z 爆炸、定位被噪声主导。
    """
    stack = np.stack(
        [m.sigma_v, 1.0 - m.cos_sim, m.q3_err], axis=0
    )  # (3, N, C)
    n_metric, _n, n_cell = stack.shape
    center = np.empty((n_metric, n_cell))
    scale = np.empty((n_metric, n_cell))
    for i in range(n_metric):
        for c in range(n_cell):
            col = stack[i, :, c].reshape(-1, 1)
            ce = robust.robust_center(col, axis=0)
            sc = robust.robust_scale(col, axis=0, center=ce, floor_rel=1e-3, floor_abs=1e-9)
            center[i, c] = ce[0]
            scale[i, c] = sc[0]
    return center, scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lfaae", "ours"], default="lfaae")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--late-frac", type=float, default=0.20,
                    help="末段聚合的比例（默认末 20%%）")
    args = ap.parse_args()

    cfg = config.load()
    tag = args.tag or args.model
    experiment = experiments.Experiment.create(cfg, args.model, tag)
    ft = load_fault_table()
    labels = read_labels(cfg)

    print("=" * 78)
    print(f"P7 单体定位：{tag}  末段比例 {args.late_frac:.0%}")
    print("=" * 78)
    print("参照系：三指标在 pack 6/8/9/10 的**训练正常期**上拟合中心与尺度")
    print("判据：故障期聚合后，真值单体是否排进逐单体异常分的前 k 名\n")

    # 1. 只在训练集（正常期）上拟合指标统计量
    train_cache = cache.load_cache(cfg, "StandTrainData")
    tr_meas, tr_rec, _ = inference.load_reconstruction(cfg, train_cache, tag, "StandTrainData")
    # 训练集窗口量大，(N,8,256) 逐块算指标以控内存
    step = 4000
    parts = []
    for s in range(0, tr_meas.shape[0], step):
        parts.append(
            fault_metric.paper_metrics(
                np.asarray(tr_meas[s : s + step]), np.asarray(tr_rec[s : s + step])
            )
        )
    tr_metrics = fault_metric.FaultMetrics(
        sigma_v=np.concatenate([p.sigma_v for p in parts], axis=0),
        cos_sim=np.concatenate([p.cos_sim for p in parts], axis=0),
        q3_err=np.concatenate([p.q3_err for p in parts], axis=0),
        v_mean_pack=np.concatenate([p.v_mean_pack for p in parts], axis=0),
    )
    del parts
    import gc

    gc.collect()

    center, scale = fit_metric_stats(tr_metrics)
    print(f"指标统计量已拟合（仅训练正常期）：{tr_metrics.sigma_v.shape[0]} 窗口 × 8 单体")
    print(f"  sigma_v 中心 {np.round(center[0], 4).tolist()}")
    print(f"  sigma_v 尺度 {np.round(scale[0], 5).tolist()}")
    print()

    rows: list[dict] = []
    per_pack: dict[int, list] = {}

    for ds in DATASETS:
        c = cache.load_cache(cfg, ds)
        vm, vr, _ = inference.load_reconstruction(cfg, c, tag, ds)
        ids = np.asarray(c["ids"])

        score_parts = []
        for s in range(0, vm.shape[0], step):
            mm = fault_metric.paper_metrics(
                np.asarray(vm[s : s + step]), np.asarray(vr[s : s + step])
            )
            score_parts.append(fault_metric.cell_anomaly_scores(mm, center, scale))
            del mm
        scores = np.concatenate(score_parts, axis=0)  # (N, 8)
        del score_parts
        gc.collect()

        for pid in sorted(np.unique(ids).tolist()):
            info = ft["packs"].get(int(pid), {})
            if not info.get("localizable"):
                continue
            expected = [int(x) for x in info["cells"]]
            if not expected:
                continue

            order = metrics_mod.pack_order(c, int(pid))
            n = len(order)
            lab = labels.get(int(pid), {})
            # 故障期起点（本包序列内下标）；无标签时退回「后一半」
            start = 0
            if lab.get("onset_point") is not None and lab.get("n_train") is not None:
                v = lab["onset_point"] - lab["n_train"]
                if v > 0:
                    start = min(v, n - 1)
            late0 = max(start, int(round((1.0 - args.late_frac) * n)))

            for agg_name, lo, hi, note in (
                ("fault_period", start, n, "起点后全部"),
                (f"last_{int(args.late_frac * 100)}pct", late0, n, "末段"),
            ):
                if hi - lo < 5:
                    continue
                block = scores[order[lo:hi]]                 # (n_win, 8)
                # 口径 1：原值
                mean_raw = np.nanmean(block, axis=0)
                # 口径 2：减去同窗口跨单体中位数（去掉包级共同成份）
                adj = block - np.nanmedian(block, axis=1, keepdims=True)
                mean_adj = np.nanmean(adj, axis=0)

                for variant, mv in (("raw", mean_raw), ("cell_centered", mean_adj)):
                    res = metrics_mod.cell_localization(mv.reshape(1, -1), expected)
                    star = "  <-- 命中" if res["hit_at_k"][1] else ""
                    print(
                        f"  pack {pid:>2} {ds[-1]}  {agg_name:>12} {variant:>14}  "
                        f"n={hi - lo:>5}  真值 {expected}  名次 {res['ranks']}  "
                        f"top3 {res['top3_cells']}{star}"
                    )
                    row = {
                        "model": args.model,
                        "tag": tag,
                        "dataset": ds,
                        "pack_id": int(pid),
                        "description": info["description"],
                        "expected_cells": expected,
                        "aggregation": agg_name,
                        "variant": variant,
                        "n_windows_aggregated": hi - lo,
                        "best_rank_of_expected": res["best_rank_of_expected"],
                        "hit_at_1": res["hit_at_k"][1],
                        "hit_at_2": res["hit_at_k"][2],
                        "hit_at_3": res["hit_at_k"][3],
                        "top3_cells": res["top3_cells"],
                        "ranks": res["ranks"],
                        "mean_score_per_cell": [round(float(x), 4) for x in res["mean_score_per_cell"]],
                    }
                    rows.append(row)
                    per_pack.setdefault(int(pid), []).append(row)

    if not rows:
        print("\n没有可评估的包 —— 检查重建缓存与 fault_table。")
        return 1

    out = experiment.write_table("localization", rows, vars(args))

    print()
    print("=" * 78)
    print(f"定位表：{out}（{len(rows)} 行）")
    print("=" * 78)
    print("\n汇总（按聚合口径 × 变体，在可标注包上）：")
    print(f"{'聚合':>16} {'变体':>14} {'包数':>5} {'top-1 命中':>10} {'top-3 命中':>10}")
    for agg_name in sorted({r["aggregation"] for r in rows}):
        for variant in ("raw", "cell_centered"):
            sub = [r for r in rows if r["aggregation"] == agg_name and r["variant"] == variant]
            if not sub:
                continue
            h1 = sum(bool(r["hit_at_1"]) for r in sub) / len(sub)
            h3 = sum(bool(r["hit_at_3"]) for r in sub) / len(sub)
            print(f"{agg_name:>16} {variant:>14} {len(sub):>5} {h1:>10.1%} {h3:>10.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
