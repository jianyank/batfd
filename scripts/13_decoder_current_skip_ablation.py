"""固定训练与检测协议，对照 decoder_current_skip 开/关。"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batfd import config, progress, provenance  # noqa: E402
from batfd.data import cache as cache_mod  # noqa: E402
from batfd.detect.lof import LOFDetector  # noqa: E402
from batfd.eval import metrics as metrics_mod  # noqa: E402
from batfd.features.fault_metric import paper_metrics  # noqa: E402
from batfd.models import inference, train as train_mod  # noqa: E402

DATASETS = ("StandTrainData", "StandTestData1")
VARIANTS = (("skip_on", True), ("skip_off", False))
SOURCE_PATHS = (
    "configs/base.yaml",
    "batfd/data/cache.py",
    "batfd/data/channels.py",
    "batfd/data/dataset.py",
    "batfd/detect/lof.py",
    "batfd/eval/metrics.py",
    "batfd/features/fault_metric.py",
    "batfd/models/cell_ae.py",
    "batfd/models/inference.py",
    "batfd/models/train.py",
    "batfd/provenance.py",
    "scripts/13_decoder_current_skip_ablation.py",
)


def build_variant_configs(cfg: dict) -> dict[str, dict]:
    """只切换 decoder_current_skip，其余配置逐字段保持一致。"""
    variants = {}
    for name, enabled in VARIANTS:
        variant = copy.deepcopy(dict(cfg))
        variant["model"]["decoder_current_skip"] = enabled
        variants[name] = variant
    return variants


def best_validation_metrics(result: dict) -> dict:
    """取训练器按 val_loss 保存的 best checkpoint 对应的 val_recon。"""
    epoch = int(result["best_epoch"])
    row = next((item for item in result["history"] if int(item["epoch"]) == epoch), None)
    if row is None:
        raise ValueError(f"训练历史中找不到 best_epoch={epoch}")
    if not np.isclose(float(row["val_loss"]), float(result["best_val_loss"])):
        raise ValueError("best checkpoint 的 val_loss 与 history 不一致")
    return {
        "best_epoch": epoch,
        "best_val_loss": float(row["val_loss"]),
        "best_val_recon": float(row["val_recon"]),
    }


def evaluate_statistics(
    statistics: dict[str, np.ndarray],
    caches: dict[str, dict],
    labels: dict[int, dict],
    *,
    persistence: int,
) -> tuple[list[dict], dict[str, dict]]:
    """仅在异常统计量已生成后使用起点标签做逐包评价。"""
    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for dataset_name, stat in statistics.items():
        cache = caches[dataset_name]
        ids = np.asarray(cache["ids"])
        times = cache.get("time")
        evals = []
        for pid in sorted(np.unique(ids).tolist()):
            order = metrics_mod.pack_order(cache, int(pid))
            time_sorted = np.asarray(times)[order] if times is not None else None
            label = labels.get(int(pid), {})
            onset_local = None
            if label.get("onset_point") is not None and label.get("n_train") is not None:
                candidate = int(label["onset_point"]) - int(label["n_train"])
                if 0 <= candidate < len(order):
                    onset_local = candidate
            evaluated = metrics_mod.evaluate_pack(
                int(pid),
                np.asarray(stat)[order],
                0.0,
                persistence=persistence,
                time_sorted=time_sorted,
                onset_index=onset_local,
                onset_source="无有效起点标签" if onset_local is None else "",
            )
            evals.append(evaluated)
            rows.append({
                "dataset": dataset_name,
                "pack_id": int(pid),
                "n_windows": evaluated.n_windows,
                "alarm_index": evaluated.alarm_index,
                "alarm_day": evaluated.alarm_day,
                "onset_index": evaluated.onset_index,
                "onset_day": evaluated.onset_day,
                "lead_days": evaluated.lead_days,
                "early_detected": evaluated.early_detected,
                "alarm_ever": evaluated.alarm_ever,
                "confirmed_before_onset": evaluated.n_confirmed_before_onset,
                "windows_before_onset": evaluated.n_windows_before_onset,
                "trigger_rate_before_onset": evaluated.trigger_rate_before_onset,
                "notes": "; ".join(evaluated.notes),
            })
        summaries[dataset_name] = metrics_mod.summarize(evals)
    return rows, summaries


def read_labels(path: Path) -> dict[int, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到故障起点标签：{path}")
    labels = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            def as_int(key):
                value = row.get(key, "")
                return int(value) if value not in ("", "None", "nan") else None
            labels[int(row["pack_id"])] = {
                "onset_point": as_int("onset_point"),
                "n_train": as_int("n_train"),
            }
    return labels


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"拒绝写入空结果表：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def input_fingerprints(cfg: dict) -> dict[str, str]:
    output = Path(cfg["paths"]["outputs_dir"])
    paths = [Path(cfg.get("_config_path", config.DEFAULT_CONFIG)), output / "tables" / "labels.csv"]
    for dataset_name in DATASETS:
        directory = output / "cache" / dataset_name
        paths.extend(directory / f"{name}.npy" for name in ("signal", "ids", "hidden", "time"))
        paths.append(directory / "meta.json")
    fingerprints = {}
    for path in paths:
        if path.is_file():
            fingerprints[str(path.resolve())] = provenance.file_digest(path)
        elif path.name == "time.npy" and path.parent.name == "StandTrainData":
            continue
        else:
            raise FileNotFoundError(f"消融输入缺失：{path}")
    return fingerprints


def write_run_manifest(directory: Path, identity: dict, *, status: str) -> None:
    files = {}
    if status == "complete":
        files = {
            path.relative_to(directory).as_posix(): provenance.file_digest(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
    provenance.write_manifest(directory / "manifest.json", {
        "schema_version": 1,
        "status": status,
        "run_id": directory.name,
        "identity": identity,
        "files": files,
    })


def run_variant(
    name: str,
    variant_cfg: dict,
    train_cache: dict,
    test_cache: dict,
    labels: dict[int, dict],
    output_dir: Path,
    *,
    device: torch.device,
) -> tuple[list[dict], dict]:
    variant_dir = output_dir / name
    variant_dir.mkdir()
    bundle = train_mod.prepare(
        variant_cfg, train_cache, out_dir=variant_dir / "training", tag="model"
    )
    training = train_mod.train(bundle)
    selection = best_validation_metrics(training)
    checkpoint = bundle.out_dir / "best.pt"
    del bundle

    model, _, _ = train_mod.load_checkpoint(checkpoint, variant_cfg)
    model = model.to(device).eval()
    features: dict[str, np.ndarray] = {}
    feature_dir = variant_dir / "features"
    feature_dir.mkdir()
    for dataset_name, dataset_cache in ((DATASETS[0], train_cache), (DATASETS[1], test_cache)):
        measured, reconstructed, _ = inference.reconstruct_all(
            model, variant_cfg, dataset_cache, device=device
        )
        feature = paper_metrics(measured, reconstructed).oriented()
        feature_path = feature_dir / f"{dataset_name}.npy"
        np.save(feature_path, feature)
        features[dataset_name] = feature
        print(f"[{name}] {dataset_name} 特征：{feature.shape}")
        del measured, reconstructed, feature
        features[dataset_name] = np.load(feature_path, mmap_mode="r")
    del model

    q = float(variant_cfg["detect"]["threshold_q"])
    persistence = int(variant_cfg["detect"]["persistence_windows"])
    detector = LOFDetector(
        n_neighbors=int(variant_cfg["detect"]["lof_n_neighbors"]),
        mode="train_novelty",
        standardize=bool(variant_cfg["detect"]["score_normalize"]),
        norm_floor=float(variant_cfg["model"]["norm_floor"]),
    ).fit(features[DATASETS[0]])
    threshold = detector.threshold_from_train(q)
    train_score = detector.score(features[DATASETS[0]]).score
    test_score = detector.score(features[DATASETS[1]]).score
    stat = test_score - threshold
    rows, summaries = evaluate_statistics(
        {DATASETS[1]: stat}, {DATASETS[1]: test_cache}, labels, persistence=persistence
    )
    for row in rows:
        row.update({
            "variant": name,
            **selection,
            "threshold_q": q,
            "persistence_windows": persistence,
            "train_threshold": threshold,
        })
    summary = {
        **selection,
        "threshold_q": q,
        "persistence_windows": persistence,
        "train_threshold": threshold,
        "n_train_features": int(len(train_score)),
        "n_test_features": int(len(test_score)),
        "evaluation": summaries[DATASETS[1]],
    }
    write_csv(variant_dir / "detection.csv", rows)
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.no_progress:
        progress.set_enabled(False)

    cfg = config.load()
    if float(cfg["detect"]["threshold_q"]) != 0.99:
        raise ValueError("本消融要求固定 threshold_q=0.99")
    if int(cfg["detect"]["persistence_windows"]) != 5:
        raise ValueError("本消融要求固定 persistence_windows=5")
    if cfg["train"].get("seed") != 42:
        raise ValueError("本消融要求固定训练随机种子 seed=42")

    identities = {
        "protocol": {
            "training_dataset": "StandTrainData",
            "evaluation_dataset": "StandTestData1",
            "variants": dict(VARIANTS),
            "model_selection": "训练器既有规则：最小 val_loss checkpoint；同时记录同 epoch val_recon",
            "detector": "train_novelty LOF，阈值只从 StandTrainData 分数 q=0.99 拟合",
            "persistence_windows": 5,
            "seed": 42,
            "onset_labels_use": "仅在异常分数冻结后做最终评价，不参与拟合或选权",
        },
        "config": {key: cfg[key] for key in ("data", "channels", "hidden", "split", "model", "train", "detect")},
        "inputs": input_fingerprints(cfg),
        "implementation_sha256": provenance.source_digest(SOURCE_PATHS),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "sklearn": sklearn.__version__,
        },
    }
    run_id = (
        "phase3_decoder_skip_"
        + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + provenance.digest(identities)[:10]
    )
    output_dir = Path(cfg["paths"]["outputs_dir"]) / "diagnostics" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(output_dir, identities, status="running")
    print(f"[phase3] 输出：{output_dir}")

    caches = {name: cache_mod.load_cache(cfg, name) for name in DATASETS}
    labels = read_labels(Path(cfg["paths"]["outputs_dir"]) / "tables" / "labels.csv")
    device = train_mod.pick_device(cfg)
    comparison = []
    summaries = {}
    try:
        for name, variant_cfg in build_variant_configs(cfg).items():
            print(f"\n=== {name}: decoder_current_skip={variant_cfg['model']['decoder_current_skip']} ===")
            rows, summaries[name] = run_variant(
                name,
                variant_cfg,
                caches[DATASETS[0]],
                caches[DATASETS[1]],
                labels,
                output_dir,
                device=device,
            )
            comparison.extend(rows)
        write_csv(output_dir / "comparison.csv", comparison)
        provenance.write_manifest(output_dir / "summary.json", {
            "run_id": run_id,
            "protocol": identities["protocol"],
            "variants": summaries,
            "comparison": comparison,
        })
        write_run_manifest(output_dir, identities, status="complete")
    except Exception:
        write_run_manifest(output_dir, identities, status="failed")
        raise

    for name, summary in summaries.items():
        result = summary["evaluation"]
        print(
            f"{name}: val_recon={summary['best_val_recon']:.6f} "
            f"val_loss={summary['best_val_loss']:.6f}@{summary['best_epoch']}  "
            f"early={result['n_early_detected']}/{result['n_packs_with_onset_label']}  "
            f"trigger_rate_before_onset={result['trigger_rate_before_onset']:.6%}"
        )
    print(f"[phase3] 完成，清单：{output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
