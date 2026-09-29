"""P8：检出率 vs 起点前触发率的权衡曲线（回应 §9 待办 6）。只报「某阈值下检出/未检出」是
把二维权衡压成一个点（论文 Table 3 就是这样：唯一指标「第几天报警」，无限敏感的检测器第
0 天报警就拿第一，见 eval/onset.py）；本脚本把阈值从松到紧扫一遍，给出**整条曲线**。

``train_novelty`` 与 ``per_pack`` 失效方向相反（见 detect/lof.py 的 docstring）：前者在
未见过的包上整包偏高，后者对整包一起漂移的缓变退化会钝化；选哪个由这条曲线定，而非先验偏好。

⚠ 两个模式**不是同一套阈值语义**：``train_novelty`` 用训练集分数的 q 分位作阈值，
``per_pack`` 用**该包自身序列**的 q 分位。故 per_pack 的「检出」天生带构造性 —— 按构造
总有约 (1-q) 的窗口越限、最早的几个可能落在第 0 窗附近；因此额外报 ``median_alarm_index``
与 ``median_lead_windows``，报警窗中位数若接近 0 则那个「100% 检出」是构造使然，不是检出能力。

口径：**检出率** = 4 个有起点标签的包中报警日早于起点日的比例（分母恒为 4）；**起点前
触发率**沿用 05_detect 的口径（轨 B，见 eval/dualtrack.py），分母 = 起点前窗口总数，
**不叫虚警率**（那些触发有相当部分是真阳性）；两者都用同一条持久性规则（连续 m 窗越限）。

用法：``python scripts/08_tradeoff.py [--tag TAG] [--modes train_novelty per_pack]``
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batfd import experiments, config, console  # noqa: E402
from batfd.data import cache  # noqa: E402
from batfd.detect import lof as lof_mod  # noqa: E402
from batfd.eval import metrics as metrics_mod  # noqa: E402

from batfd.features import cache as feature_cache  # noqa: E402

console.setup()

COND_LABELS = ["mean_abs_i", "std_i", "p95_abs_i", "disch_frac",
               "mean_v", "std_v", "range_v", "mean_t", "std_t"]  # 已排除 mean_soc


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


def evaluate_point(
    det: lof_mod.LOFDetector,
    feat_tr: np.ndarray,
    feat_te: np.ndarray,
    ids_te: np.ndarray,
    cache_te: dict,
    labels: dict[int, dict],
    *,
    mode: str,
    q: float,
    persistence: int,
) -> dict:
    """在一个 (模式, 分位, 持久性) 点上算检出率与起点前触发率。"""
    if mode == "train_novelty":
        s_tr = det.score(feat_tr).score
        s_te = det.score(feat_te).score
        # 论文的做法：阈值取自**训练集**分数的 q 分位（detect/lof.py, Eq.9–12）
        thr = float(np.quantile(s_tr, q))
        stat_te = s_te - thr
    else:
        # per_pack：逐包自参照 —— 对**每个包自己的窗口序列**拟合 LOF，判据是该包
        # 自己分数的 q 分位（LOFDetector.score_self_referential）。
        # ⚠ 这里**不能**像 train_novelty 那样拿训练集的分位数去切：它返回的是「块内
        # 自己 median/MAD 标准化后」的**原始 LOF 分**（恒正），训练块与测试块各自标准化、
        # 尺度不可比，跨块相减得到的越限没有含义 —— 本脚本前一版正是这么写的，结果
        # per_pack 触发率被压到近 0，是伪结果（留注备查）。
        stat_te = np.full(len(feat_te), np.nan, dtype=np.float64)
        for pid in np.unique(ids_te):
            m = ids_te == pid
            res = det.score_self_referential(feat_te[m], q)
            if res.threshold is None:      # 窗口太少，该包整体置 NaN
                continue
            stat_te[m] = res.score - res.threshold
        thr = None      # 该模式没有单一阈值：每个包各有一条自己的判据

    n_lab = n_det = 0
    win_before = hit_before = 0
    per_pack = []
    for pid in sorted(labels):
        order = metrics_mod.pack_order(cache_te, pid)
        if len(order) == 0:
            continue
        lab = labels[pid]
        if lab["onset_point"] is None or lab["n_train"] is None:
            continue
        onset_local = lab["onset_point"] - lab["n_train"]
        if onset_local < 0:
            continue
        n_lab += 1

        s = stat_te[order]
        flag = s > 0.0
        conf = lof_mod.apply_persistence(flag, persistence)
        ai = lof_mod.first_alarm_index(conf)
        detected = ai is not None and ai < onset_local
        n_det += int(detected)

        before = conf[:onset_local]
        win_before += int(onset_local)
        hit_before += int(before.sum())
        per_pack.append(
            {
                "pack_id": pid,
                "alarm_index": ai,
                "onset_index": int(onset_local),
                "detected": bool(detected),
                "n_confirmed_before": int(before.sum()),
                "n_windows_before": int(onset_local),
            }
        )

    # 「检出」必须和「第几窗报的」一起看：per_pack 的阈值取自该包自身序列的 q 分位，
    # 按构造总有约 (1-q) 的窗口越限、最早的几个可能就在第 0 窗附近；若
    # median_alarm_index 接近 0，则「100% 检出」是构造使然而非检出能力。
    det_alarms = [p["alarm_index"] for p in per_pack if p["detected"]]
    det_leads = [p["onset_index"] - p["alarm_index"] for p in per_pack if p["detected"]]

    return {
        "detection_rate": (n_det / n_lab) if n_lab else float("nan"),
        "n_labelled": n_lab,
        "n_detected": n_det,
        "median_alarm_index": float(np.median(det_alarms)) if det_alarms else None,
        "median_lead_windows": float(np.median(det_leads)) if det_leads else None,
        "trigger_rate_before_onset": (hit_before / win_before) if win_before else float("nan"),
        "windows_before_onset": win_before,
        "confirmed_before_onset": hit_before,
        "threshold": thr,
        "per_pack": per_pack,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lfaae", "ours"], default="lfaae")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--modes", nargs="*", default=["train_novelty", "per_pack"])
    ap.add_argument("--qs", nargs="*", type=float,
                    default=[0.90, 0.95, 0.975, 0.99, 0.995, 0.999])
    ap.add_argument("--persistence", nargs="*", type=int, default=[1, 5, 10])
    args = ap.parse_args()

    cfg = config.load()
    tag = args.tag or args.model
    experiment = experiments.Experiment.create(cfg, args.model, tag)
    k = int(cfg["detect"]["lof_n_neighbors"])

    print("=" * 78)
    print(f"P8 权衡曲线：{tag}  模式 {args.modes}")
    print(f"  分位网格 {args.qs}  持久性网格 {args.persistence}")
    print("=" * 78)

    labels = read_labels(cfg)
    c_tr = cache.load_cache(cfg, "StandTrainData")
    c_te = cache.load_cache(cfg, "StandTestData1")
    feat_tr = np.asarray(feature_cache.load_metrics(cfg, c_tr, tag, "StandTrainData")[0])
    feat_te = np.asarray(feature_cache.load_metrics(cfg, c_te, tag, "StandTestData1")[0])
    ids_te = np.asarray(c_te["ids"])
    print(f"训练特征 {feat_tr.shape}  测试特征 {feat_te.shape}")
    print(f"有起点标签的包：{sorted(labels)}")
    print()

    rows: list[dict] = []
    for mode in args.modes:
        det = lof_mod.LOFDetector(n_neighbors=k, mode=mode, standardize=True)
        det.fit(feat_tr)
        print("-" * 78)
        print(f"LOF 模式：{mode}")
        print("-" * 78)
        print(f"{'q':>7} {'m':>4} {'阈值':>10} {'检出率':>8} {'起点前触发率':>12} "
              f"{'中位报警窗':>10} {'中位提前窗':>10} {'分母':>7}")
        for q in args.qs:
            for m in args.persistence:
                r = evaluate_point(
                    det, feat_tr, feat_te, ids_te, c_te, labels,
                    mode=mode, q=q, persistence=m,
                )
                rows.append({
                    "model": args.model, "tag": tag, "lof_mode": mode,
                    "q": q, "persistence": m,
                    "threshold": r["threshold"],
                    "n_labelled": r["n_labelled"],
                    "n_detected": r["n_detected"],
                    "detection_rate": r["detection_rate"],
                    "median_alarm_index": r["median_alarm_index"],
                    "median_lead_windows": r["median_lead_windows"],
                    "windows_before_onset": r["windows_before_onset"],
                    "confirmed_before_onset": r["confirmed_before_onset"],
                    "trigger_rate_before_onset": r["trigger_rate_before_onset"],
                })
                f2 = lambda v: "—" if v is None else f"{v:.0f}"  # noqa: E731
                thr_txt = "逐包自身" if mode == "per_pack" else f"{r['threshold']:.4f}"
                print(
                    f"{q:>7.3f} {m:>4} {thr_txt:>10} "
                    f"{r['detection_rate']:>7.1%} {r['trigger_rate_before_onset']:>12.2%} "
                    f"{f2(r['median_alarm_index']):>10} {f2(r['median_lead_windows']):>10} "
                    f"{r['windows_before_onset']:>7}"
                )
        print()

    out = experiment.write_table("tradeoff", rows, vars(args))
    print("=" * 78)
    print(f"权衡表：{out}（{len(rows)} 行）")
    print("=" * 78)
    print("\n读法：同一模式下横向比较各点 —— 检出率上升伴随着起点前触发率上升，")
    print("      两者就是那条权衡曲线；不同模式之间的比较才是「选哪个模式」的依据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
