"""版本隔离的LOF训练口径修正；复用每折封存特征，不重训或留出调参。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import platform
import sys

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batfd import console, provenance  # noqa: E402
from batfd.detect.lof_v2 import LOFDetectorV2  # noqa: E402

_spec = importlib.util.spec_from_file_location("phase6_replay_source", ROOT / "scripts/16_phase6_lopo_shift_diagnostic.py")
PHASE6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PHASE6)
PHASE5 = PHASE6.PHASE5
SOURCE_PATHS = (*PHASE6.SOURCE_PATHS, "batfd/detect/lof_v2.py", "scripts/17_phase7_lof_replay.py")
VARIANTS = ("all", "sigma_v", "reconstruction")
COLUMNS = {"all": list(range(24)), "sigma_v": list(range(0, 24, 3)),
           "reconstruction": [j for j in range(24) if j % 3]}


def select_features(features, variant):
    x = np.asarray(features, dtype=np.float64)
    if (variant not in COLUMNS or x.ndim != 2 or x.shape[1] != 24
            or not len(x) or not np.isfinite(x).all()):
        raise ValueError("特征必须为有限非空N×24交错矩阵，指标组必须预先固定")
    return x[:, COLUMNS[variant]]


def fit_scores(calibration, held_out, cfg, variant):
    cal = select_features(calibration, variant)
    held = select_features(held_out, variant)
    detector = LOFDetectorV2(n_neighbors=cfg["detect"]["lof_n_neighbors"], mode="train_novelty",
                             standardize=cfg["detect"]["score_normalize"],
                             norm_floor=cfg["model"]["norm_floor"]).fit(cal)
    threshold = detector.threshold_from_train(cfg["detect"]["threshold_q"])
    return detector, detector.score(held).score, threshold


def protocol(variants):
    return {"detector_version": "LOFDetectorV2", "training_score": "-negative_outlier_factor_",
            "held_out_score": "-score_samples(new_samples)", "lof_fit_scope": "另外三包全部封存特征",
            "n_neighbors": 20, "q": .99, "persistence_windows": 5,
            "norm_floor": .001, "standardize": True, "comparison": "score > threshold",
            "conditional_threshold_applied": False, "deep_retrained": False, "hidden_read": False,
            "labels_used": False, "test_data_used": False, "tuned_on_held_out": False,
            "automatic_winner_selection": False, "feature_columns": {v:COLUMNS[v] for v in variants},
            "order": "原训练缓存行顺序，不是真实时间轴", "not_claimed": "现场虚警率或故障检出证据"}


def overlap(a, b):
    return a == b or a in b.parents or b in a.parents


def checked_source(run_dir):
    directory = Path(run_dir).resolve()
    original = PHASE5.verify_run(directory)
    cache_dir, cfg = PHASE6.checked_inputs(original)
    detect = cfg["detect"]
    if (detect["lof_mode"] != "train_novelty" or detect["lof_n_neighbors"] != 20
            or detect["threshold_q"] != .99 or detect["persistence_windows"] != 5
            or detect["score_normalize"] is not True or cfg["model"]["norm_floor"] != .001):
        raise ValueError("输入不属于预先固定的k20/q.99/m5/稳健标准化协议")
    ids = np.load(cache_dir / "ids.npy", allow_pickle=False)
    PHASE5.split_outer_fold(ids, PHASE5.PACKS[0])
    for pid in PHASE5.PACKS:
        base = directory / "folds" / str(pid)
        with np.load(base / "split.npz", allow_pickle=False) as split:
            expected_cal, expected_held = PHASE5.split_outer_fold(ids, pid)
            if (not np.array_equal(split["calibration_idx"], expected_cal)
                    or not np.array_equal(split["held_out_idx"], expected_held)):
                raise ValueError(f"pack {pid} 来源IDs与封存外层切分不符")
        PHASE6.check_fold_protocol(cfg, provenance.read_manifest(base / "summary.json"))
    return directory, cache_dir, cfg


def identity(directory, cache_dir, cfg, variants):
    return {"source_run": str(directory), "source_files": PHASE6.tree_hashes(directory),
            "input_directory": str(cache_dir),
            "input_sha256": {n:provenance.file_digest(cache_dir/n) for n in ("signal.npy","ids.npy","meta.json")},
            "configuration_sha256": provenance.digest(cfg),
            "implementation_sha256": provenance.source_digest(SOURCE_PATHS),
            "runtime": {"python":platform.python_version(), "numpy":np.__version__, "sklearn":sklearn.__version__},
            "protocol": protocol(variants)}


def evaluate_fold(directory, cfg, pid, variant):
    base = directory / "folds" / str(pid)
    cal = np.load(base / "calibration_features.npy", allow_pickle=False)
    held = np.load(base / "held_out_features.npy", allow_pickle=False)
    with np.load(base / "split.npz", allow_pickle=False) as split:
        indices = split["held_out_idx"]
    old = provenance.read_manifest(base / "summary.json")
    detector, scores, threshold = fit_scores(cal, held, cfg, variant)
    # 全24维只更改训练评分语义；尺度与新样本分数必须精确保持旧实现。
    if variant == "all":
        with np.load(base / "lof_scaler.npz", allow_pickle=False) as scaler:
            if (not np.array_equal(detector.center_, scaler["center"])
                    or not np.array_equal(detector.scale_, scaler["scale"])
                    or not np.array_equal(scores, np.load(base / "held_out_scores.npy", allow_pickle=False))):
                raise ValueError(f"pack {pid} 全24维尺度或留出评分发生了口径之外的变化")
    row, windows = PHASE5.summarize_scores(scores, threshold, pid, indices,
                                          persistence=cfg["detect"]["persistence_windows"])
    row.update(variant=variant, n_features=len(COLUMNS[variant]), n_calibration_windows=len(cal),
               calibration_packs=[p for p in PHASE5.PACKS if p != pid],
               q=cfg["detect"]["threshold_q"], effective_n_neighbors=int(detector.model_.n_neighbors_),
               training_score_semantics="-negative_outlier_factor_",
               old_full24_threshold=old["calibration_threshold"],
               old_full24_exceedance_rate=old["exceedance_rate"],
               old_full24_confirmed_rate=old["persistence_confirmed_rate"])
    return row, windows, detector, scores


def aggregate(rows, variants, source_unchanged):
    return {"variants": list(variants), "n_windows":sum(r["n_held_out_windows"] for r in rows if r["variant"] == "all"),
            "source_unchanged":source_unchanged, "protocol":protocol(variants), "packs":rows,
            "aggregate":{v:PHASE5.aggregate_summaries([r for r in rows if r["variant"] == v]) for v in variants},
            "limitations":"保存特征检测链路诊断；较低越限率不等于更优检出；非现场虚警率；无时间与故障起点标签"}


def report_text(summary):
    lines = ["# 四折 LOPO：标准训练LOF口径与固定指标组对照", "",
             "深度模型未重训。k=20，校准训练分数q=.99，严格大于阈值，连续5窗从第5窗确认。",
             "仅另外三包拟合标准化和LOF。全部指标组在评估前固定，无自动优选或留出调参。", "",
             "| 指标组 | 包 | 维数 | 校准阈值 | 越限数/窗口数 | 越限率 | 确认数 | 确认率 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary["packs"]:
        lines.append(f"| {r['variant']} | {r['pack_id']} | {r['n_features']} | {r['calibration_threshold']:.12g} | "
                     f"{r['n_exceedance_windows']}/{r['n_held_out_windows']} | {r['exceedance_rate']:.2%} | "
                     f"{r['n_persistence_confirmed_windows']} | {r['persistence_confirmed_rate']:.2%} |")
    lines += ["", "sigma_v使用0::3；reconstruction使用1::3和2::3（按单体交错保留）；all使用全部24维。",
              "all分组的留出分数与旧Phase5完全一致，变化仅来自标准训练分数和其q99阈值。",
              "旧配置的工况阈值开关未在本实验应用；这里固定使用全局校准分位阈值。",
              "较少报警不能单独证明优于完整指标；无独立故障标签，不能报告检出率或选择最终模型。",
              "原始行顺序不是时间轴；窗口越限率和确认率不是现场虚警率。", ""]
    return "\n".join(lines)


def run_replay(run_dir, output_dir, *, ablation=False):
    directory, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if overlap(directory, output):
        raise ValueError("输出禁止与旧四折产物树重叠")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖：{output}")
    directory, cache_dir, cfg = checked_source(directory)
    if overlap(cache_dir, output):
        raise ValueError("输出禁止与输入缓存树重叠")
    variants = VARIANTS if ablation else ("all",)
    before = identity(directory, cache_dir, cfg, variants)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version":1, "status":"running", "identity":before,
                "variants":list(variants), "completed_folds":[], "current_fold":None}
    # 仅写一次终态清单，避免Windows监控/索引读取与原子替换既有文件竞争。
    rows = []
    try:
        for variant in variants:
            for pid in PHASE5.PACKS:
                print(f"拟合 {variant} / pack {pid}：仅校准包封存特征", flush=True)
                row, windows, detector, scores = evaluate_fold(directory, cfg, pid, variant)
                base = output / variant / "folds" / str(pid)
                base.mkdir(parents=True, exist_ok=False)
                np.save(base / "calibration_scores.npy", detector._train_score)
                np.save(base / "held_out_scores.npy", scores)
                np.savez(base / "lof_scaler.npz", center=detector.center_, scale=detector.scale_)
                PHASE5.write_csv(base / "windows.csv", windows)
                provenance.write_manifest(base / "summary.json", row)
                rows.append(row)
                manifest["completed_folds"].append(f"{variant}/{pid}")
        if identity(directory, cache_dir, cfg, variants) != before:
            raise ValueError("计算期间来源、输入或实现发生变化")
        summary = aggregate(rows, variants, True)
        PHASE5.write_csv(output / "pack_summary.csv", rows)
        provenance.write_manifest(output / "summary.json", summary)
        (output / "report.md").write_text(report_text(summary), encoding="utf-8")
        manifest.update(status="complete", current_fold=None)
        PHASE5.seal_manifest(output, manifest)
        return summary
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        provenance.write_manifest(output / "manifest.json", manifest)
        raise


def check_csv(path, rows):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        saved = list(csv.DictReader(handle))
    if saved != [{k:"" if v is None else str(v) for k,v in row.items()} for row in rows]:
        raise ValueError(f"CSV与重放语义不一致：{path}")


def verify_replay(output_dir):
    output = Path(output_dir).resolve()
    manifest = provenance.read_manifest(output / "manifest.json")
    if manifest is None or manifest.get("status") != "complete":
        raise ValueError("检测复算尚未完整完成")
    variants = manifest.get("variants")
    if variants not in (["all"], list(VARIANTS)):
        raise ValueError("指标组不符合预先固定协议")
    required = {"summary.json", "pack_summary.csv", "report.md"}
    for v in variants:
        for pid in PHASE5.PACKS:
            required.update(f"{v}/folds/{pid}/{name}" for name in
                            ("calibration_scores.npy","held_out_scores.npy","lof_scaler.npz","windows.csv","summary.json"))
    actual = PHASE6.tree_hashes(output)
    actual.pop("manifest.json", None)
    if set(actual) != required or actual != manifest.get("files"):
        raise ValueError("产物完整文件集合或哈希不一致")
    expected_folds = [f"{v}/{p}" for v in variants for p in PHASE5.PACKS]
    if manifest.get("completed_folds") != expected_folds or manifest.get("current_fold") is not None:
        raise ValueError("完成折清单不完整")
    saved_identity = manifest.get("identity", {})
    directory, cache_dir, cfg = checked_source(saved_identity["source_run"])
    if overlap(output, directory) or overlap(output, cache_dir):
        raise ValueError("输出与来源树重叠")
    if saved_identity != identity(directory, cache_dir, cfg, variants):
        raise ValueError("来源树、输入、配置、实现、运行库或协议身份不一致")
    rows = []
    for variant in variants:
        for pid in PHASE5.PACKS:
            row, windows, detector, scores = evaluate_fold(directory, cfg, pid, variant)
            base = output / variant / "folds" / str(pid)
            for name, expected in (("calibration_scores",detector._train_score), ("held_out_scores",scores)):
                saved = np.load(base / f"{name}.npy", allow_pickle=False)
                if not np.array_equal(saved, expected):
                    raise ValueError(f"pack {pid} {variant} {name}与标准口径重放不一致")
            with np.load(base / "lof_scaler.npz", allow_pickle=False) as scaler:
                if (set(scaler.files) != {"center","scale"} or not np.array_equal(scaler["center"], detector.center_)
                        or not np.array_equal(scaler["scale"], detector.scale_)):
                    raise ValueError("尺度与校准特征不一致")
            if provenance.read_manifest(base / "summary.json") != row:
                raise ValueError("每折摘要与标准口径重放不一致")
            check_csv(base / "windows.csv", windows)
            rows.append(row)
    expected_summary = aggregate(rows, variants, True)
    if provenance.read_manifest(output / "summary.json") != expected_summary:
        raise ValueError("总摘要、协议或聚合结果与重放不一致")
    check_csv(output / "pack_summary.csv", rows)
    if (output / "report.md").read_text(encoding="utf-8") != report_text(expected_summary):
        raise ValueError("报告与重放摘要不一致")
    return manifest


def main(argv=None):
    console.setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="已封存Phase5四折运行目录")
    parser.add_argument("--output-dir", type=Path, help="全新检测结果目录")
    parser.add_argument("--ablation", action="store_true", help="固定all/sigma_v/reconstruction三组；不自动择优")
    parser.add_argument("--verify", type=Path, help="独立验收已有检测复算目录")
    args = parser.parse_args(argv)
    if args.verify:
        if args.run_dir or args.output_dir or args.ablation:
            parser.error("--verify不能与运行参数混用")
        verify_replay(args.verify)
        print("检测复算验收通过")
        return 0
    if not args.run_dir:
        parser.error("运行时必须指定--run-dir")
    output = args.output_dir or ROOT / "outputs/diagnostics" / (
        "phase7_lof_" + ("ablation_" if args.ablation else "corrected_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    result = run_replay(args.run_dir, output, ablation=args.ablation)
    print(f"检测复算完成：{result['n_windows']}个留出窗口；输出 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
