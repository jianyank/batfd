"""P5：故障检测全流程 —— 故障指标 → LOF → 阈值 → 持久性 → 报警。

1. 用模型 checkpoint 对四个数据集重建（带磁盘缓存）。
2. 算论文 §2.3 的三个逐单体指标，拼成 24 维包级向量，按「越大越异常」定向。
3. 用**只用训练集**拟合的稳健标准化 + LOF 给测试窗口打分。
4. 阈值口径（见 --methods）：``fixed`` = 论文原文的训练集分数 99 分位、不分工况；
   ``quantile_regression`` = 工况条件分位数（梯度提升）；``binned_quantile`` = 工况分位
   分箱 + 单调平滑；``pack_baseline`` = 逐包自身基线。
5. 持久性规则（连续 m 窗越限）→ 逐包报警时间 → 与 Tier A 起点标签比提前量。

两个必须提前定死的口径：**(a) 工况协变量一律排除 SOC** —— ``mean_soc`` 只有 pack
6/8/9/10 有（只有它们带 Hiddall），训练用了而另外 10 个测试包没有则维度对不上、比较也
不公平；**(b) 天数是相对天数** —— 投运日期未知 + 数据是论文子集，论文的绝对 day of
operation 无法复现，故用「距本包首条记录的天数」，所有方法在同一批窗口上比才公平。

用法：``python scripts/05_detect.py --model {lfaae,ours} [--tag TAG]``
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batfd import config, console, progress, robust  # noqa: E402
from batfd.data import cache, channels, conditions as cond_mod  # noqa: E402
from batfd.detect import lof as lof_mod  # noqa: E402
from batfd.detect import threshold as thr_mod  # noqa: E402
from batfd.eval import metrics as metrics_mod  # noqa: E402
from batfd.features import fault_metric, cache as feature_cache  # noqa: E402
from batfd.models import inference  # noqa: E402

console.setup()

DATASETS = ("StandTrainData", "StandTestData1", "StandTestData2", "StandTestData3")
TEST_DATASETS = ("StandTestData1", "StandTestData2", "StandTestData3")


def load_model(cfg: dict, kind: str, tag: str):
    ckpt = inference.checkpoint_path(cfg, tag)
    if not ckpt.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt}")
    if kind == "lfaae":
        from batfd.baselines import lfaae

        return lfaae.load(ckpt, cfg)
    if kind == "ours":
        from batfd.models import train as train_mod

        model, _, _ = train_mod.load_checkpoint(ckpt, cfg)
        return model
    raise ValueError(f"--model 应为 lfaae 或 ours，收到 {kind}")


def read_labels(cfg: dict) -> dict[int, dict]:
    p = Path(cfg["paths"]["outputs_dir"]) / "tables" / "labels.csv"
    if not p.exists():
        print(f"[warn] 找不到起点标签 {p}，将只报告检出情况、不报告提前量")
        return {}
    out: dict[int, dict] = {}
    with p.open("r", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            pid = int(row["pack_id"])

            def _int(k):
                v = row.get(k, "")
                return int(v) if v not in ("", "None", "nan") else None

            out[pid] = {
                "onset_point": _int("onset_point"),
                "n_train": _int("n_train"),
                "boundary_ok": row.get("boundary_ok") == "True",
                "cells_expected": [
                    int(x) for x in row.get("cells_expected", "[]").strip("[]").split(",") if x.strip()
                ],
            }
    return out


def get_metrics(cfg, tag, dataset_name, v_meas, v_rec, *, force=False):
    return feature_cache.get_metrics(cfg, tag, dataset_name, v_meas, v_rec, force=force)


PER_PACK_METHOD = "per_pack_self"


def per_pack_stat(det, feat: np.ndarray, ids: np.ndarray, q: float) -> np.ndarray:
    """``per_pack`` 模式的判据值：**逐包自参照**。

    每包拿**自身窗口序列**做块内稳健标准化，阈值取该块自身分数的 q 分位；判据值 = 本包
    分数 − 本包阈值（>0 即越限），与其余方法的「>0 判异常」口径一致。

    ⚠ 两处不能想当然（实测踩过的坑）：1) **阈值必须用 ``LOFResult.threshold``**，不能拿
    训练集（或整块）的分位数去切 —— 那是块外统计量配块内分数，参照系不同、结果不可比；
    2) **必须逐包调用**，整块调用等于把所有包混在一起自比，包间差异会互相吸收。
    """
    stat = np.full(len(feat), np.nan, dtype=np.float64)
    for pid in np.unique(ids):
        m = ids == pid
        res = det.score_self_referential(np.asarray(feat)[m], q)
        if res.threshold is None:
            continue
        stat[m] = res.score - res.threshold
    return stat


def condition_matrix(cfg, cache_dict) -> tuple[np.ndarray, list[str]]:
    """取工况协变量并**全局排除 SOC**（理由见模块 docstring）。"""
    names = list(cond_mod.CONDITION_NAMES)
    cond = np.asarray(cache_dict["cond"], dtype=np.float64)
    keep = [i for i, n in enumerate(names) if n != "mean_soc"]
    return cond[:, keep], [names[i] for i in keep]


def pack_baseline_stat(
    score: np.ndarray, ids: np.ndarray, *, baseline_frac: float, min_windows: int = 30
) -> np.ndarray:
    """逐包用**自身**前 ``baseline_frac`` 窗口做稳健标准化，返回 z 分数。

    为什么需要它：LOF 分数是与训练流形比较得来的，换到没见过的包时整体重建误差抬高，
    **中位** LOF 就已超过训练集 99 分位（实测未见包中位 1.34~2.23，训练集 99 分位 1.62），
    于是 16%~98% 的正常窗口被判异常 —— 这是阈值**参照系错了**，不是检测器不够灵敏：该问
    的不是「这包比训练包更异常吗」，而是「这包比**它自己正常时**更异常吗」。每包都含一段
    明确正常运行期，故该参照可得且合理（论文亦言所选包 "include both a well-defined
    period of normal operation and a subsequent phase of progressive anomaly evolution"）。

    实现要点：只用该包**最前面**一小段建基线，不回头看后面的数据，否则会把待检出的异常
    混进基线，导致阈值虚高、真异常漏检。
    """
    stat = np.full(score.shape, np.nan, dtype=np.float64)
    for pid in np.unique(ids):
        m = ids == pid
        s = score[m]
        w0 = max(min_windows, int(round(baseline_frac * len(s))))
        w0 = min(w0, len(s) - 1) if len(s) > 1 else len(s)
        base = s[:w0].reshape(-1, 1)
        # 多估计量取最大的稳健尺度（见 batfd/robust.py）：单用 MAD 在重尾下会严重
        # 低估尺度，实测会让正常数据出现 z=60 的点，于是任何阈值都拦不住。
        med = robust.robust_center(base, axis=0)
        scale = robust.robust_scale(base, axis=0, center=med, floor_rel=1e-3, floor_abs=1e-6)
        stat[m] = robust.zscore(s.reshape(-1, 1), med, scale, axis=0).ravel()
    return stat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lfaae", "ours"], default="lfaae")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--lof-mode", choices=["train_novelty", "per_pack"], default="train_novelty")
    ap.add_argument("--no-standardize", action="store_true",
                    help="不标准化指标（用于量化论文未声明的这个选择的影响）")
    ap.add_argument("--force-recon", action="store_true")
    ap.add_argument("--methods", nargs="*",
                    default=list(thr_mod.METHODS) + ["pack_baseline"],
                    help=f"阈值口径，可选 {list(thr_mod.METHODS) + ['pack_baseline']}")
    ap.add_argument("--no-progress", action="store_true",
                    help="关闭进度条（输出重定向到文件时会自动关闭，无需手动指定）")
    args = ap.parse_args()

    per_pack = args.lof_mode == "per_pack"
    if per_pack and list(args.methods) != [PER_PACK_METHOD]:
        # per_pack 下每个包各有一条取自**自身**序列的判据，不存在跨包的统一阈值，
        # 「阈值口径」这一维度因此不适用；若不拦，四个口径会产出四份完全相同的行，
        # 看起来像"四种方法一致"，实际是同一个东西重复了四遍。
        print(f"  ⚠ per_pack 模式下阈值口径不适用（阈值取自每包自身分位）——")
        print(f"    只跑 {PER_PACK_METHOD}，忽略 --methods 的其余取值。")
        args.methods = [PER_PACK_METHOD]

    if args.no_progress:
        progress.set_enabled(False)

    cfg = config.load()
    tag = args.tag or args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print(f"P5 检测：模型 {args.model}  tag={tag}  LOF={args.lof_mode}  "
          f"标准化={not args.no_standardize}  设备={device}")
    print("=" * 78)

    model = load_model(cfg, args.model, tag).to(device).eval()
    caches = {n: cache.load_cache(cfg, n) for n in DATASETS}

    # 1. 重建 + 2. 指标
    feats: dict[str, np.ndarray] = {}
    sigmas: dict[str, np.ndarray] = {}
    for name in DATASETS:
        v_meas, v_rec, _ = inference.reconstruct_dataset(
            model, cfg, caches[name], tag, name, device=device, force=args.force_recon
        )
        feat, sig, cos, q3 = get_metrics(cfg, tag, name, v_meas, v_rec, force=args.force_recon)
        feats[name] = np.asarray(feat)
        sigmas[name] = np.asarray(sig)
        del v_meas, v_rec

    print()
    print("--- LOF（仅用训练集特征拟合）---")
    det = lof_mod.LOFDetector(
        n_neighbors=int(cfg["detect"]["lof_n_neighbors"]),
        mode=args.lof_mode,
        standardize=not args.no_standardize,
        norm_floor=float(cfg["model"]["norm_floor"]),
    )
    det.fit(feats["StandTrainData"])
    print(f"  训练特征 {feats['StandTrainData'].shape}，"
          f"标准化={'开' if det.standardize else '关'}")

    q = float(cfg["detect"]["threshold_q"])
    persistence = int(cfg["detect"]["persistence_windows"])
    ids_tr = np.asarray(caches["StandTrainData"]["ids"])
    if not per_pack:
        train_score = det.score(feats["StandTrainData"]).score
        print(f"  训练集分数：median={np.median(train_score):.4f}  "
              f"{q:.2f} 分位={np.quantile(train_score, q):.4f}  max={train_score.max():.4f}")
    else:
        # 这里得到的是判据值（分数 − 本包阈值），不再是原始分数
        train_score = per_pack_stat(det, feats["StandTrainData"], ids_tr, q)
        fin = np.isfinite(train_score)
        over = int((train_score[fin] > 0).sum())
        print(f"  per_pack 模式：逐包自参照，判据值 = 本包分数 − 本包 {q:.2f} 分位阈值")
        print(f"    训练集 {len(np.unique(ids_tr))} 个包共越限 {over}/{int(fin.sum())} 个窗口"
              f"（按构造约 {100 * (1 - q):.1f}% = {int(fin.sum()) * (1 - q):.0f} 个）")

    cond_tr, cond_names = condition_matrix(cfg, caches["StandTrainData"])
    print(f"  工况协变量（已排除 SOC）：{cond_names}")
    finite = np.isfinite(train_score) & np.isfinite(cond_tr).all(axis=1)
    print(f"  可用于阈值拟合的训练窗口：{int(finite.sum())}/{len(train_score)}")

    labels = read_labels(cfg)
    all_rows: list[dict] = []

    # 4/5. 逐阈值方法 × 逐数据集
    for method in args.methods:
        print()
        print("=" * 78)
        print(f"阈值方法：{method}")
        print("=" * 78)

        # 所有方法统一成「stat 越大越异常、判据为 stat > 0」的形式，下游的持久性
        # 规则与评价代码因此只需一套逻辑。
        tm = None
        pack_base_decision = None
        if per_pack:
            # 判据已在 per_pack_stat 里逐包算好，这里没有跨包阈值可拟合
            print("  阈值取自每个包**自身**序列的 q 分位，不拟合跨包阈值")
        elif method == "pack_baseline":
            bf = float(cfg["detect"]["pack_baseline_frac"])
            stat_tr = pack_baseline_stat(train_score, ids_tr, baseline_frac=bf)
            fin = np.isfinite(stat_tr)
            pack_base_decision = float(np.quantile(stat_tr[fin], q))
            print(f"  逐包自身基线标准化：参照段比例 {bf:.0%}，"
                  f"训练集 4 个包的 z 分数 {q:.2f} 分位 = {pack_base_decision:.4f}")
            print(f"  训练集 z 分数：median={np.median(stat_tr[fin]):.4f}  "
                  f"max={stat_tr[fin].max():.4f}")
        else:
            tm = thr_mod.fit(
                method,
                train_score[finite],
                None if method == "fixed" else cond_tr[finite],
                q,
            )
            if method != "fixed":
                extra = "" if method != "binned_quantile" else f"（选用工况维 {tm.payload['dim']}）"
                print(f"  已拟合{extra}")

        for ds in TEST_DATASETS:
            feat = feats[ds]
            cond, _ = condition_matrix(cfg, caches[ds])
            ids = np.asarray(caches[ds]["ids"])
            if per_pack:
                stat = per_pack_stat(det, feat, ids, q)
            else:
                sc = det.score(feat).score      # train_novelty：训练集拟合的检测器打分
                if method == "pack_baseline":
                    z = pack_baseline_stat(
                        sc, ids, baseline_frac=float(cfg["detect"]["pack_baseline_frac"])
                    )
                    stat = z - pack_base_decision
                else:
                    stat = tm.stat(sc, None if method == "fixed" else cond)
            time_all = caches[ds].get("time")

            print(f"\n  [{ds}]")
            for pid in sorted(np.unique(ids).tolist()):
                order = metrics_mod.pack_order(caches[ds], int(pid))
                s = stat[order]
                t = np.asarray(time_all)[order] if time_all is not None else None

                lab = labels.get(int(pid), {})
                onset_local = None
                if lab.get("onset_point") is not None and lab.get("n_train") is not None:
                    v = lab["onset_point"] - lab["n_train"]
                    if v >= 0:
                        onset_local = v
                    else:
                        print(f"    pack {pid:>2}: 起点落在正常期（下标 {lab['onset_point']} "
                              f"< n_train {lab['n_train']}），不用于提前量")

                ev = metrics_mod.evaluate_pack(
                    int(pid), s, 0.0, persistence=persistence,
                    time_sorted=t, onset_index=onset_local,
                    onset_source="无 Hiddall" if not lab else "",
                )
                print(
                    f"    pack {pid:>2}  n={ev.n_windows:>6}  "
                    f"报警={'第 %.1f 天' % ev.alarm_day if ev.alarm_day is not None else '无':>12}  "
                    f"起点={'第 %.1f 天' % ev.onset_day if ev.onset_day is not None else '—':>12}  "
                    f"提前={'%.1f 天' % ev.lead_days if ev.lead_days is not None else '—':>9}  "
                    f"起点前虚警={ev.n_confirmed_before_onset}"
                    + ("  " + "; ".join(ev.notes) if ev.notes else "")
                )
                all_rows.append(
                    {
                        "model": args.model,
                        "tag": tag,
                        "lof_mode": args.lof_mode,
                        "standardize": not args.no_standardize,
                        "threshold_method": method,
                        "dataset": ds,
                        "pack_id": int(pid),
                        "n_windows": ev.n_windows,
                        "alarm_day": ev.alarm_day,
                        "onset_day": ev.onset_day,
                        "lead_days": ev.lead_days,
                        "false_alarms_before_onset": ev.n_confirmed_before_onset,
                        "windows_before_onset": ev.n_windows_before_onset,
                        "far_per_window": ev.far_per_window,
                        "detected": ev.detected,
                        "notes": "; ".join(ev.notes),
                    }
                )

    # 文件名带上方法列表：用 --methods 只跑子集时不会覆盖此前的完整结果
    mtag = "-".join(args.methods)
    out = Path(cfg["paths"]["outputs_dir"]) / "tables" / f"detection_{tag}_{args.lof_mode}_{mtag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print()
    print("=" * 78)
    print(f"检测表：{out}（{len(all_rows)} 行）")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
