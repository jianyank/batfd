"""P6：双轨虚警率（回应 §5.2）。产出 ``outputs/tables/far_dualtrack.csv``：每个阈值方法 ×
每个有起点标签的包（pack 6/8/9/10）各一行，给出两轨数字 —— 轨 A ``A_far`` = 训练集已知
正常期上的**真虚警率（留一口径）**；轨 B ``B_trigger_rate`` = test1 起点之前窗口上的
**触发率（不是虚警率）**。

**轨 A 必须留一**（理由见 batfd/eval/dualtrack.py 的 docstring 陷阱 1）：in-sample 拿 LOF
给训练过的点打分实测报警恒为 0，是退化结果；故对每个 pack 用**其余三个包**拟合检测器与
阈值，再给该包自己的训练窗口打分。

两轨分母场景不同、**不可相减**；报警日一并报出（虚警率须与检出能力同读）。

用法：``python scripts/06_far_dualtrack.py [--methods fixed pack_baseline]``
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batfd import config, console, progress, robust  # noqa: E402
from batfd.data import cache, conditions as cond_mod  # noqa: E402
from batfd.detect import lof as lof_mod  # noqa: E402
from batfd.detect import threshold as thr_mod  # noqa: E402
from batfd.eval import dualtrack, metrics as metrics_mod  # noqa: E402

console.setup()

DATASETS = ("StandTrainData", "StandTestData1", "StandTestData2", "StandTestData3")
ALL_METHODS = list(thr_mod.METHODS) + ["pack_baseline"]


def condition_matrix(cache_dict) -> tuple[np.ndarray, list[str]]:
    """取工况协变量并全局排除 SOC（理由见 scripts/05_detect.py 的 docstring）。"""
    names = list(cond_mod.CONDITION_NAMES)
    cond = np.asarray(cache_dict["cond"], dtype=np.float64)
    keep = [i for i, n in enumerate(names) if n != "mean_soc"]
    return cond[:, keep], [names[i] for i in keep]


def pack_baseline_stat(score, ids, *, baseline_frac, min_windows=30):
    """逐包自身基线标准化。与 scripts/05_detect.py 的实现保持同一口径。"""
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


def read_labels(cfg) -> dict[int, dict]:
    p = Path(cfg["paths"]["outputs_dir"]) / "tables" / "labels.csv"
    out: dict[int, dict] = {}
    with p.open("r", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            pid = int(row["pack_id"])

            def _int(k):
                v = row.get(k, "")
                return int(v) if v not in ("", "None", "nan") else None

            out[pid] = {"onset_point": _int("onset_point"), "n_train": _int("n_train")}
    return out


def fit_and_score_leave_one_id(
    x_train: np.ndarray,
    train_ids: np.ndarray,
    cond_train: np.ndarray,
    *,
    method: str,
    q: float,
    n_neighbors: int,
    baseline_frac: float,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """逐包留一：用其余包拟合，给留出的包打分。

    Returns: ``{pack_id: 该包训练窗口的分数}`` 与 ``{pack_id: 该折判定阈值（拟合集 q 分位）}``。
    """
    scores: dict[int, np.ndarray] = {}
    thresholds: dict[int, float] = {}
    ids_u = sorted(np.unique(train_ids).tolist())

    if method == "pack_baseline":
        # pack_baseline 的参照系就是「该包自己的前 10%」，每个包自成一个参照系，
        # **没有「留一」可言**；硬套留一只会把别的包拿来作与它无关的基线，数字无意义。
        # 因此轨 A 不报该方法，而是显式抛错（见 05_detect 的同一口径说明）。
        raise ValueError(
            "pack_baseline 不适用于轨 A：它的参照系是逐包自身的，无法留一。"
        )

    for pid in ids_u:
        m = train_ids == pid
        x_fit, x_out = x_train[~m], x_train[m]
        c_fit = cond_train[~m]
        fin = np.isfinite(c_fit).all(axis=1)

        det = lof_mod.LOFDetector(
            n_neighbors=n_neighbors, mode="train_novelty", standardize=True
        )
        det.fit(x_fit)
        if method == "fixed":
            tm = thr_mod.fit_fixed(det.score(x_fit).score, q)
        else:
            tm = thr_mod.fit(method, det.score(x_fit).score, c_fit[fin], q)
        # 阈值判据一律折进 stat（>0 即越限），所以 thresholds 恒为 0
        scores[pid] = tm.stat(det.score(x_out).score, cond_train[m])
        thresholds[pid] = 0.0

    return scores, thresholds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lfaae", "ours"], default="lfaae")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--methods", nargs="*", default=ALL_METHODS)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    if args.no_progress:
        progress.set_enabled(False)

    cfg = config.load()
    tag = args.tag or args.model
    q = float(cfg["detect"]["threshold_q"])
    persistence = int(cfg["detect"]["persistence_windows"])
    bf = float(cfg["detect"]["pack_baseline_frac"])
    k = int(cfg["detect"]["lof_n_neighbors"])

    print("=" * 78)
    print(f"P6 双轨虚警率：{tag}  LOF=train_novelty  持久性 m={persistence}  q={q}")
    print("=" * 78)
    print("轨 A = 已知正常期上的真虚警率（**逐包留一**：用其余包拟合）")
    print("轨 B = test1 起点前窗口上的触发率（不是虚警率）")
    print()

    caches = {n: cache.load_cache(cfg, n) for n in DATASETS}
    feats = {
        n: np.asarray(np.load(f"outputs/runs/{tag}/recon/{n}/feat.npy", mmap_mode="r"))
        for n in DATASETS
    }
    labels = read_labels(cfg)

    train_ids = np.asarray(caches["StandTrainData"]["ids"])
    x_train = feats["StandTrainData"]
    cond_tr, cond_names = condition_matrix(caches["StandTrainData"])
    print(f"训练集工况协变量（已排除 SOC）：{cond_names}")
    print(f"训练集窗口 {x_train.shape}，包 {sorted(np.unique(train_ids).tolist())}")
    print()

    rows: list[dict] = []
    for method in args.methods:
        print("-" * 78)
        print(f"阈值方法：{method}")
        print("-" * 78)

        # 轨 A：逐包留一（pack_baseline 参照系逐包自身、无法留一，见函数内说明）
        a: list = []
        if method != "pack_baseline":
            a_scores, a_thr = fit_and_score_leave_one_id(
                x_train, train_ids, cond_tr,
                method=method, q=q, n_neighbors=k, baseline_frac=bf,
            )
            a = dualtrack.track_a_leave_one_id(
                a_scores, a_thr, persistence=persistence, min_windows=30
            )
            sa = dualtrack.summarize_track_a(a)
            print(f"  轨 A 真虚警率（留一）：{sa['confirmed']}/{sa['windows']} = "
                  f"{(sa['rate'] or 0):.4%}")
            for r in a:
                print(f"      pack {r.pack_id:>2}: {r.n_confirmed:>5}/{r.n_windows:<6} = "
                      f"{r.far_per_window:7.4%}  单窗越限 {r.n_windows_flagged:>5}  "
                      f"阈值 {r.threshold:8.4f}  中位分 {r.median_score:8.4f}")
        else:
            print("  轨 A：不适用（pack_baseline 的参照系是逐包自身的，无法留一）")

        # 全训练集拟合的口径（用于 test1 侧，与 05_detect 一致）
        det_full = lof_mod.LOFDetector(n_neighbors=k, mode="train_novelty", standardize=True)
        det_full.fit(x_train)
        if method == "pack_baseline":
            stat_tr_full = pack_baseline_stat(det_full.score(x_train).score, train_ids, baseline_frac=bf)
            fin = np.isfinite(stat_tr_full)
            dec_full = float(np.quantile(stat_tr_full[fin], q))
            print(f"  全训练集口径：逐包基线 z 的 {q:.2f} 分位 = {dec_full:.4f}")
        elif method == "fixed":
            tm_full = thr_mod.fit_fixed(det_full.score(x_train).score, q)
            dec_full = None
        else:
            fin = np.isfinite(cond_tr).all(axis=1)
            tm_full = thr_mod.fit(method, det_full.score(x_train).score, cond_tr[fin], q)
            dec_full = None

        # 轨 B + 检出，逐包
        te_ids = np.asarray(caches["StandTestData1"]["ids"])
        cond_te, _ = condition_matrix(caches["StandTestData1"])
        sc_te_all = det_full.score(feats["StandTestData1"]).score

        for pid in sorted(labels):
            order = metrics_mod.pack_order(caches["StandTestData1"], pid)
            if len(order) == 0:
                continue
            sc = sc_te_all[order]
            if method == "pack_baseline":
                z = pack_baseline_stat(sc_te_all, te_ids, baseline_frac=bf)
                stat_te = (z - dec_full)[order]
            elif method == "fixed":
                stat_te = tm_full.stat(sc_te_all)[order]
            else:
                stat_te = tm_full.stat(sc_te_all, cond_te)[order]

            lab = labels[pid]
            onset_local = None
            if lab["onset_point"] is not None and lab["n_train"] is not None:
                v = lab["onset_point"] - lab["n_train"]
                onset_local = v if v >= 0 else None

            tb = dualtrack.track_b(stat_te, onset_index=onset_local, persistence=persistence)
            time_all = caches["StandTestData1"].get("time")
            days = (
                metrics_mod.days_from_start(np.asarray(time_all)[order])
                if time_all is not None
                else None
            )
            alarm_day = (
                dualtrack.first_alarm_day(stat_te, days, persistence=persistence)
                if days is not None
                else None
            )

            print(f"      pack {pid:>2}  轨 B 触发率 "
                  f"{'—' if tb is None else f'{tb.n_confirmed}/{tb.n_windows} = {tb.trigger_per_window:.4%}'}"
                  f"   报警日 {'—' if alarm_day is None else f'{alarm_day:.1f}'}"
                  f"   起点日 {'—' if onset_local is None else f'{days[onset_local]:.1f}'}")

            ra = next((x for x in a if x.pack_id == pid), None)
            rows.append({
                "model": args.model,
                "tag": tag,
                "lof_mode": "train_novelty",
                "threshold_method": method,
                "pack_id": pid,
                # 轨 A（留一）
                "A_windows": None if ra is None else ra.n_windows,
                "A_confirmed": None if ra is None else ra.n_confirmed,
                "A_far": None if ra is None else ra.far_per_window,
                "A_windows_flagged": None if ra is None else ra.n_windows_flagged,
                "A_threshold": None if ra is None else ra.threshold,
                "A_median_score": None if ra is None else ra.median_score,
                # 轨 B
                "B_windows": None if tb is None else tb.n_windows,
                "B_confirmed": None if tb is None else tb.n_confirmed,
                "B_trigger_rate": None if tb is None else tb.trigger_per_window,
                # 检出
                "alarm_day": alarm_day,
                "onset_day": (
                    None if (days is None or onset_local is None) else float(days[onset_local])
                ),
            })
        print()

    out = Path(cfg["paths"]["outputs_dir"]) / "tables" / "far_dualtrack.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("=" * 78)
    print(f"双轨表：{out}（{len(rows)} 行）")
    print("=" * 78)
    print("\n汇总（按方法池化；轨 A 与轨 B 分母不同，不可相减）：")
    print(f"{'方法':>18} {'轨A 虚警率':>12} {'轨B 触发率':>12} {'A分母':>8} {'B分母':>8}")
    for method in args.methods:
        sub = [r for r in rows if r["threshold_method"] == method]
        aw = sum(r["A_windows"] or 0 for r in sub)
        ac = sum(r["A_confirmed"] or 0 for r in sub)
        bw = sum(r["B_windows"] or 0 for r in sub)
        bc = sum(r["B_confirmed"] or 0 for r in sub)
        print(
            f"{method:>18} {(ac / aw if aw else float('nan')):>12.4%} "
            f"{(bc / bw if bw else float('nan')):>12.4%} {aw:>8} {bw:>8}"
        )
    print("\n注：轨 A 是「留出包自己的正常期」为分母、「其余三个包」为参照系的真虚警率；")
    print("    轨 B 沿用 05_detect 的口径，改名触发率。两者不可相减。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
