"""从随机初始化重训四折 LOPO；留出包不参与任何拟合或选权。"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batfd import config, console, progress, provenance  # noqa: E402
from batfd.detect.lof import LOFDetector, apply_persistence  # noqa: E402
from batfd.features.fault_metric import paper_metrics  # noqa: E402
from batfd.models import inference, train as train_mod  # noqa: E402

PACKS = (6, 8, 9, 10)
EXPECTED_COUNTS = {6: 9127, 8: 11147, 9: 5093, 10: 1141}
SOURCE_PATHS = (
    "configs/base.yaml", "batfd/config.py", "batfd/console.py", "batfd/progress.py",
    "batfd/data/channels.py", "batfd/data/dataset.py", "batfd/robust.py",
    "batfd/models/cell_ae.py", "batfd/models/train.py", "batfd/models/losses.py",
    "batfd/models/inference.py", "batfd/features/fault_metric.py",
    "batfd/detect/lof.py", "batfd/provenance.py", "scripts/15_phase5_e2e_lopo.py",
)


def split_outer_fold(ids, held_out_pack):
    ids = np.asarray(ids)
    if ids.ndim != 1 or set(np.unique(ids).tolist()) != set(PACKS):
        raise ValueError("IDs 必须是一维且恰好包含四个目标包 6/8/9/10")
    if held_out_pack not in PACKS:
        raise ValueError("留出包不属于四折协议")
    return np.flatnonzero(ids != held_out_pack), np.flatnonzero(ids == held_out_pack)


def build_fold_cache(cache, indices, *, include_hidden):
    indices = np.asarray(indices)
    return {
        "name": cache.get("name", "StandTrainData"),
        "signal": cache["signal"][indices],
        "ids": np.asarray(cache["ids"])[indices],
        "hidden": cache["hidden"][indices] if include_hidden and cache.get("hidden") is not None else None,
        "time": cache["time"][indices] if cache.get("time") is not None else None,
    }


def summarize_scores(scores, threshold, pack_id, original_indices, *, persistence=5):
    scores = np.asarray(scores)
    indices = np.asarray(original_indices)
    if (scores.ndim != 1 or scores.size == 0 or indices.shape != scores.shape
            or not np.isfinite(scores).all() or not np.isfinite(threshold)
            or persistence < 1 or int(persistence) != persistence):
        raise ValueError("分数、阈值、行索引或持久性参数无效")
    exceeded = scores > threshold
    confirmed = apply_persistence(exceeded, persistence)
    alarms = np.flatnonzero(confirmed)
    summary = {
        "pack_id": int(pack_id), "n_held_out_windows": int(len(scores)),
        "calibration_threshold": float(threshold), "persistence_windows": int(persistence),
        "n_exceedance_windows": int(exceeded.sum()), "exceedance_rate": float(exceeded.mean()),
        "n_persistence_confirmed_windows": int(confirmed.sum()),
        "persistence_confirmed_rate": float(confirmed.mean()), "alarm_ever": bool(alarms.size),
        "first_confirmed_alarm_window": int(alarms[0]) if alarms.size else None,
    }
    windows = [
        {"pack_id": int(pack_id), "window_index": j, "original_index": int(indices[j]),
         "score": float(score), "calibration_threshold": float(threshold),
         "exceeded_threshold": bool(exceeded[j]), "persistence_confirmed": bool(confirmed[j])}
        for j, score in enumerate(scores)
    ]
    return summary, windows


def aggregate_summaries(rows):
    n = sum(row["n_held_out_windows"] for row in rows)
    return {
        "n_packs": len(rows), "n_windows": n,
        "n_packs_with_alarm": sum(bool(row["alarm_ever"]) for row in rows),
        **{f"{weight}_{metric}_rate": float(
            np.mean([row[count] / row["n_held_out_windows"] for row in rows])
            if weight == "macro" else sum(row[count] for row in rows) / n)
           for metric, count in (("exceedance", "n_exceedance_windows"),
                                 ("persistence_confirmed", "n_persistence_confirmed_windows"))
           for weight in ("macro", "weighted")},
    }


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seal_manifest(directory, manifest):
    manifest["files"] = {
        path.relative_to(directory).as_posix(): provenance.file_digest(path)
        for path in sorted(directory.rglob("*")) if path.is_file() and path != directory / "manifest.json"
    }
    provenance.write_manifest(directory / "manifest.json", manifest)


def run_fold(cfg, cache, held_out_pack, output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "status": "running", "pack_id": int(held_out_pack)}
    provenance.write_manifest(directory / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        calibration_idx, held_idx = split_outer_fold(cache["ids"], held_out_pack)
        fold_cfg = copy.deepcopy(dict(cfg))
        training_cache = build_fold_cache(cache, calibration_idx, include_hidden=True)
        if training_cache["hidden"] is None or not np.isfinite(training_cache["hidden"]).all():
            raise ValueError("校准包物理训练目标缺失或非有限")
        # 先剔除留出包，再切分与拟合所有标准化器；没有全包 prepare 通路。
        device = train_mod.pick_device(fold_cfg)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        bundle = train_mod.prepare(fold_cfg, training_cache, out_dir=directory / "training", tag="model")
        train_idx = calibration_idx[bundle.split.train_idx]
        val_idx = calibration_idx[bundle.split.val_idx]
        np.savez(directory / "split.npz", calibration_idx=calibration_idx, held_out_idx=held_idx,
                 train_idx=train_idx, val_idx=val_idx)
        training = train_mod.train(bundle)
        selected = next(row for row in training["history"] if row["epoch"] == training["best_epoch"])
        checkpoint = directory / "training" / "model" / "best.pt"
        del bundle, training_cache
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model, _, _ = train_mod.load_checkpoint(checkpoint, fold_cfg)
        features = {}
        for name, indices in (("calibration", calibration_idx), ("held_out", held_idx)):
            # 推理不传 hidden，连取样通路也不读取留出包物理目标。
            prediction_cache = build_fold_cache(cache, indices, include_hidden=False)
            meas, rec, predicted_ids = inference.reconstruct_all(
                model, fold_cfg, prediction_cache, device=device,
                batch_size=int(fold_cfg["train"]["batch_size"]))
            if not np.array_equal(predicted_ids, np.asarray(cache["ids"])[indices]):
                raise ValueError("推理输出 IDs 与原始行索引不一致")
            values = paper_metrics(meas, rec).oriented().astype(np.float64)
            if values.shape != (len(indices), 24) or not np.isfinite(values).all():
                raise ValueError("重建特征必须为有限的 N×24 数组")
            np.save(directory / f"{name}_features.npy", values)
            features[name] = values
            del prediction_cache, meas, rec
            gc.collect()
        detector = LOFDetector(
            n_neighbors=int(fold_cfg["detect"]["lof_n_neighbors"]), mode="train_novelty",
            standardize=bool(fold_cfg["detect"]["score_normalize"]),
            norm_floor=float(fold_cfg["model"]["norm_floor"])).fit(features["calibration"])
        q = float(fold_cfg["detect"]["threshold_q"])
        threshold = detector.threshold_from_train(q)
        calibration_scores = detector.score(features["calibration"]).score
        held_scores = detector.score(features["held_out"]).score
        np.save(directory / "calibration_scores.npy", calibration_scores)
        np.save(directory / "held_out_scores.npy", held_scores)
        np.savez(directory / "lof_scaler.npz", center=detector.center_, scale=detector.scale_)
        summary, windows = summarize_scores(
            held_scores, threshold, held_out_pack, held_idx,
            persistence=int(fold_cfg["detect"]["persistence_windows"]))
        summary.update({
            "n_calibration_windows": int(len(calibration_idx)), "n_train_windows": int(len(train_idx)),
            "n_val_windows": int(len(val_idx)), "q": q, "seed": int(fold_cfg["train"]["seed"]),
            "calibration_packs": sorted(set(np.asarray(cache["ids"])[calibration_idx].tolist())),
            "best_epoch": int(training["best_epoch"]), "best_val_loss": float(selected["val_loss"]),
            "best_val_recon": float(selected["val_recon"]), "epochs_run": int(training["epochs_run"]),
            "training_elapsed_sec": float(training["elapsed_sec"]),
            "mean_epoch_sec": float(training["elapsed_sec"] / training["epochs_run"]),
            "fold_elapsed_sec": time.perf_counter() - started, "device": str(device),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        })
        write_csv(directory / "windows.csv", windows)
        provenance.write_manifest(directory / "summary.json", summary)
        manifest.update(status="complete", summary=summary)
        seal_manifest(directory, manifest)
        del model, detector, features
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return summary
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        provenance.write_manifest(directory / "manifest.json", manifest)
        raise


def run_experiment(cfg, cache, output_dir, *, identity=None):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "run_id": directory.name, "status": "running",
                "identity": identity or {}, "completed_packs": [], "current_pack": None}
    provenance.write_manifest(directory / "manifest.json", manifest)
    summaries = []
    try:
        split_outer_fold(cache["ids"], PACKS[0])
        for pack_id in PACKS:
            manifest["current_pack"] = pack_id
            provenance.write_manifest(directory / "manifest.json", manifest)
            print(f"\n[LOPO] 留出 pack {pack_id}；其余三包从随机初始化完整训练", flush=True)
            summary = run_fold(cfg, cache, pack_id, directory / "folds" / str(pack_id))
            summaries.append(summary)
            manifest["completed_packs"].append(pack_id)
            write_name = directory / f"completed_pack_{pack_id}.json"
            provenance.write_manifest(write_name, summary)
            provenance.write_manifest(directory / "manifest.json", manifest)
        aggregate = aggregate_summaries(summaries)
        aggregate["packs"] = summaries
        aggregate["limitations"] = "训练缓存未见包诊断；非现场虚警率；无起点标签；原始行顺序非真实时间轴"
        write_csv(directory / "pack_summary.csv", summaries)
        provenance.write_manifest(directory / "summary.json", aggregate)
        manifest.update(status="complete", current_pack=None, summary=aggregate)
        seal_manifest(directory, manifest)
        verify_run(directory)
        return aggregate
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        provenance.write_manifest(directory / "manifest.json", manifest)
        raise


def verify_run(output_dir):
    directory = Path(output_dir)
    manifest = provenance.read_manifest(directory / "manifest.json")
    if manifest is None or manifest.get("status") != "complete" or manifest.get("completed_packs") != list(PACKS):
        raise ValueError("四折运行尚未完整完成")
    for base in [directory, *(directory / "folds" / str(pid) for pid in PACKS)]:
        check = provenance.read_manifest(base / "manifest.json")
        if check is None or check.get("status") != "complete" or not check.get("files"):
            raise ValueError(f"产物清单缺失或未完成：{base}")
        for relative, expected in check["files"].items():
            path = base / relative
            if not path.is_file() or provenance.file_digest(path) != expected:
                raise ValueError(f"产物哈希不一致：{path}")
    # 哈希证明内容一致，不证明切分正确；独立核对外层/内层索引与产物行数。
    folds = {}
    for pid in PACKS:
        with np.load(directory / "folds" / str(pid) / "split.npz", allow_pickle=False) as split:
            folds[pid] = {key: split[key] for key in ("calibration_idx", "held_out_idx", "train_idx", "val_idx")}
    held_all = np.concatenate([folds[pid]["held_out_idx"] for pid in PACKS])
    if not np.array_equal(np.sort(held_all), np.arange(len(held_all))):
        raise ValueError("四折留出索引没有互斥覆盖原始行")
    for pid, split in folds.items():
        base = directory / "folds" / str(pid)
        cal, held, train, val = (split[key] for key in ("calibration_idx", "held_out_idx", "train_idx", "val_idx"))
        if any(idx.ndim != 1 or idx.size == 0 or idx.dtype.kind not in "iu"
               or np.unique(idx).size != idx.size for idx in (cal, held, train, val)):
            raise ValueError(f"pack {pid} 行索引必须为非空、不重复的整数数组")
        if any(not np.all(np.diff(idx) > 0) for idx in (cal, held)):
            raise ValueError(f"pack {pid} 校准/留出索引必须保留原始行顺序")
        if (np.intersect1d(cal, held).size or np.intersect1d(train, val).size
                or not np.array_equal(np.sort(np.concatenate([train, val])), cal)
                or not np.array_equal(np.sort(np.concatenate([cal, held])), np.arange(len(held_all)))):
            raise ValueError(f"pack {pid} 训练/验证/留出索引隔离或覆盖错误")
        summary = provenance.read_manifest(base / "summary.json")
        if (summary is None or summary["pack_id"] != pid
                or summary["n_calibration_windows"] != len(cal) or summary["n_held_out_windows"] != len(held)):
            raise ValueError(f"pack {pid} 摘要与折索引行数不一致")
        for label, indices in (("calibration", cal), ("held_out", held)):
            feature = np.load(base / f"{label}_features.npy", allow_pickle=False)
            score = np.load(base / f"{label}_scores.npy", allow_pickle=False)
            if (feature.shape != (len(indices), 24) or score.shape != (len(indices),)
                    or not np.isfinite(feature).all() or not np.isfinite(score).all()):
                raise ValueError(f"pack {pid} 特征或分数形状/有限性不符")
        cal_scores = np.load(base / "calibration_scores.npy", allow_pickle=False)
        if not np.isclose(np.quantile(cal_scores, summary["q"]), summary["calibration_threshold"], rtol=1e-12, atol=1e-12):
            raise ValueError(f"pack {pid} 阈值不等于校准分数分位数")
        scores = np.load(base / "held_out_scores.npy", allow_pickle=False)
        expected_summary, expected_windows = summarize_scores(
            scores, summary["calibration_threshold"], pid, held,
            persistence=summary["persistence_windows"])
        if any(summary[key] != value for key, value in expected_summary.items()):
            raise ValueError(f"pack {pid} 报警摘要与留出分数不一致")
        with (base / "windows.csv").open(encoding="utf-8-sig", newline="") as handle:
            windows = list(csv.DictReader(handle))
        if windows != [{key: str(value) for key, value in row.items()} for row in expected_windows]:
            raise ValueError(f"pack {pid} 逐窗口输出与索引/分数不一致")
    return manifest


def load_inputs(cfg):
    directory = Path(cfg["paths"]["outputs_dir"]) / "cache" / "StandTrainData"
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if meta.get("cache_version") != 1 or meta.get("name") != "StandTrainData":
        raise ValueError("训练缓存来源或版本不符")
    cache = {"name": "StandTrainData", "time": None,
             **{name: np.load(directory / f"{name}.npy", mmap_mode="r")
                for name in ("signal", "ids", "hidden")}}
    ids = np.asarray(cache["ids"])
    split_outer_fold(ids, PACKS[0])
    counts = {int(pid): int((ids == pid).sum()) for pid in PACKS}
    if counts != EXPECTED_COUNTS or {int(k): int(v) for k, v in meta["id_counts"].items()} != counts:
        raise ValueError("训练包窗口计数与固定来源不符")
    if (cache["signal"].shape != (len(ids), 256, 20) or cache["hidden"].shape != (len(ids), 9)
            or meta["n_windows"] != len(ids) or meta["signal_shape"] != list(cache["signal"].shape)
            or meta.get("has_time") or (directory / "time.npy").exists()):
        raise ValueError("训练信号、物理目标或时间元数据形状不符")
    for start in range(0, len(ids), 512):
        if not np.isfinite(cache["signal"][start:start + 512]).all():
            raise ValueError("训练缓存原始信号含非有限值")
    fingerprints = {name: provenance.file_digest(directory / name)
                    for name in ("signal.npy", "ids.npy", "hidden.npy", "meta.json")}
    return cache, {"directory": str(directory), "sha256": fingerprints, "pack_counts": counts}


def main():
    console.setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="新的运行目录；已有目录一律拒绝覆盖")
    parser.add_argument("--verify", type=Path, help="仅核验已经完成的四折目录")
    args = parser.parse_args()
    if args.verify:
        manifest = verify_run(args.verify)
        print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
        return 0
    progress.set_enabled(False)
    cfg = config.load()
    # 真实入口拒绝无意修改核心协议，测试可直接调用 run_experiment 使用小模型。
    if (cfg["train"]["seed"] != 42 or cfg["train"]["max_epochs"] != 300
            or cfg["train"]["patience"] != 20 or cfg["split"]["val_ratio"] != .15
            or not cfg["model"]["decoder_current_skip"] or cfg["detect"]["threshold_q"] != .99
            or cfg["detect"]["persistence_windows"] != 5 or cfg["detect"]["lof_n_neighbors"] != 20
            or not cfg["detect"]["score_normalize"]):
        raise ValueError("当前配置与已确认固定协议不一致")
    cache, source = load_inputs(cfg)
    identity = {
        "inputs": source, "config": {k: cfg[k] for k in ("data", "channels", "hidden", "split", "model", "train", "detect")},
        "implementation_sha256": provenance.source_digest(SOURCE_PATHS),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": str(torch.__version__), "sklearn": sklearn.__version__},
        "protocol": {"held_out_packs": list(PACKS), "labels_used": False, "test_data_used": False,
                     "warm_start": False, "lof_fit_scope": "另外三包全部窗口，含该折深度验证窗口",
                     "hidden_use": "仅另外三包物理对齐训练；留出包只哈希文件来源，不读取目标值用于计算",
                     "order": "训练缓存原始行顺序，无时间列", "not_claimed": "现场虚警率或提前检出"},
    }
    run_id = "phase5_e2e_lopo_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + provenance.digest(identity)[:10]
    directory = args.output_dir or Path(cfg["paths"]["outputs_dir"]) / "diagnostics" / run_id
    print(f"[LOPO] 独立运行目录：{directory}", flush=True)
    run_experiment(cfg, cache, directory, identity=identity)
    print("[LOPO] 四折完成，产物 SHA-256 核验通过", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
