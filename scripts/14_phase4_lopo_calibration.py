"""在冻结的 skip_on 表示上执行训练包留一 LOF 阈值诊断。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batfd import provenance  # noqa: E402
from batfd.detect.lof import LOFDetector, apply_persistence  # noqa: E402

PHASE3_RUN_ID = "phase3_decoder_skip_20261001T144548Z_fd81c8b0c1"
PHASE3_DIR = ROOT / "outputs" / "diagnostics" / PHASE3_RUN_ID
FEATURE_RELATIVE = Path("skip_on") / "features" / "StandTrainData.npy"
FEATURE_PATH = PHASE3_DIR / FEATURE_RELATIVE
IDS_PATH = ROOT / "outputs" / "cache" / "StandTrainData" / "ids.npy"
CACHE_META_PATH = IDS_PATH.with_name("meta.json")
EXPECTED_PACK_COUNTS = {6: 9127, 8: 11147, 9: 5093, 10: 1141}
SOURCE_PATHS = (
    "batfd/detect/lof.py",
    "batfd/provenance.py",
    "scripts/14_phase4_lopo_calibration.py",
)


def load_frozen_inputs() -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """加载固定第三轮特征与训练包 ID，并校验来源指纹。"""
    phase3_manifest_path = PHASE3_DIR / "manifest.json"
    manifest = provenance.read_manifest(phase3_manifest_path)
    if manifest is None or manifest.get("status") != "complete":
        raise ValueError(f"第三轮运行清单缺失或未完成：{phase3_manifest_path}")
    expected_hash = manifest.get("files", {}).get(FEATURE_RELATIVE.as_posix())
    actual_hash = provenance.file_digest(FEATURE_PATH)
    if expected_hash != actual_hash:
        raise ValueError("冻结特征 SHA-256 与第三轮 manifest 不一致")

    run_config = manifest.get("identity", {}).get("config", {})
    model_config = run_config.get("model", {})
    detect_config = run_config.get("detect", {})
    if model_config.get("decoder_current_skip") is not True:
        raise ValueError("固定第三轮来源不是 decoder_current_skip=on")
    if detect_config.get("lof_mode") != "train_novelty":
        raise ValueError("第三轮清单的 LOF 模式不是 train_novelty")

    features = np.load(FEATURE_PATH, mmap_mode="r", allow_pickle=False)
    ids = np.load(IDS_PATH, mmap_mode="r", allow_pickle=False)
    meta = json.loads(CACHE_META_PATH.read_text(encoding="utf-8"))
    if features.ndim != 2 or features.shape[1] != 24:
        raise ValueError(f"冻结特征必须为 N×24，实际为 {features.shape}")
    if ids.ndim != 1 or len(ids) != len(features):
        raise ValueError("训练包 ID 与冻结特征行数不匹配")
    if not np.isfinite(features).all():
        raise ValueError("冻结特征含非有限值")

    counts = {int(pid): int(np.sum(ids == pid)) for pid in np.unique(ids)}
    if counts != EXPECTED_PACK_COUNTS:
        raise ValueError(f"训练包 ID 计数与已核实基线不符：{counts}")
    metadata_counts = {int(pid): int(count) for pid, count in meta["id_counts"].items()}
    if int(meta["n_windows"]) != len(ids) or metadata_counts != counts:
        raise ValueError("训练缓存 meta.json 与 ID 行数/计数不一致")

    parameters = {
        "q": float(detect_config["threshold_q"]),
        "persistence": int(detect_config["persistence_windows"]),
        "n_neighbors": int(detect_config["lof_n_neighbors"]),
        "standardize": bool(detect_config["score_normalize"]),
        "norm_floor": float(model_config["norm_floor"]),
    }
    expected_parameters = {
        "q": 0.99,
        "persistence": 5,
        "n_neighbors": 20,
        "standardize": True,
        "norm_floor": 1e-3,
    }
    if parameters != expected_parameters:
        raise ValueError(f"第三轮 LOF 协议与冻结诊断预期不符：{parameters}")
    return features, ids, parameters, {
        "phase3_manifest_path": str(phase3_manifest_path.relative_to(ROOT)),
        "phase3_manifest_sha256": provenance.file_digest(phase3_manifest_path),
        "feature_path": str(FEATURE_PATH.relative_to(ROOT)),
        "feature_sha256": actual_hash,
        "ids_path": str(IDS_PATH.relative_to(ROOT)),
        "ids_sha256": provenance.file_digest(IDS_PATH),
        "cache_meta_path": str(CACHE_META_PATH.relative_to(ROOT)),
        "cache_meta_sha256": provenance.file_digest(CACHE_META_PATH),
        "pack_counts": {str(pid): count for pid, count in counts.items()},
    }


def calibrate_pack(
    features: np.ndarray,
    ids: np.ndarray,
    pack_id: int,
    *,
    q: float,
    persistence: int,
    n_neighbors: int,
    standardize: bool,
    norm_floor: float,
) -> tuple[dict, list[dict]]:
    """用其余训练包拟合 LOF 与阈值，并诊断指定留出包。"""
    features = np.asarray(features)
    ids = np.asarray(ids)
    if features.ndim != 2 or ids.ndim != 1 or len(features) != len(ids):
        raise ValueError("特征与 IDs 必须行数一致，且分别为二维和一维数组")
    if not 0 < q <= 1:
        raise ValueError("q 必须在 (0, 1] 范围内")
    if persistence < 1 or n_neighbors < 1:
        raise ValueError("persistence 和 n_neighbors 必须为正整数")
    if not np.isfinite(features).all():
        raise ValueError("特征含非有限值")

    held_out_mask = ids == pack_id
    if not held_out_mask.any():
        raise ValueError(f"找不到留出包 {pack_id}")
    train_features = features[~held_out_mask]
    held_out_features = features[held_out_mask]
    detector = LOFDetector(
        n_neighbors=n_neighbors,
        mode="train_novelty",
        standardize=standardize,
        norm_floor=norm_floor,
    ).fit(train_features)
    threshold = detector.threshold_from_train(q)
    scores = detector.score(held_out_features).score
    exceedance = scores > threshold
    confirmed = apply_persistence(exceedance, persistence)
    alarm_indices = np.flatnonzero(confirmed)
    first_alarm = int(alarm_indices[0]) if alarm_indices.size else None

    summary = {
        "pack_id": int(pack_id),
        "n_calibration_windows": int(len(train_features)),
        "n_held_out_windows": int(len(held_out_features)),
        "q": float(q),
        "persistence_windows": int(persistence),
        "calibration_threshold": float(threshold),
        "n_exceedance_windows": int(exceedance.sum()),
        "exceedance_rate": float(exceedance.mean()),
        "n_persistence_confirmed_windows": int(confirmed.sum()),
        "persistence_confirmed_rate": float(confirmed.mean()),
        "alarm_ever": bool(confirmed.any()),
        "first_confirmed_alarm_window": first_alarm,
    }
    windows = [
        {
            "pack_id": int(pack_id),
            "window_index": index,
            "score": float(score),
            "calibration_threshold": float(threshold),
            "exceeded_threshold": bool(exceed),
            "persistence_confirmed": bool(is_confirmed),
        }
        for index, (score, exceed, is_confirmed) in enumerate(
            zip(scores, exceedance, confirmed)
        )
    ]
    return summary, windows


def write_csv(path: Path, rows: list[dict]) -> None:
    """独占创建 CSV，避免覆盖已有诊断产物。"""
    if not rows:
        raise ValueError("拒绝写入空诊断表")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    features, ids, parameters, input_identity = load_frozen_inputs()
    identity = {
        "source": input_identity,
        "protocol": {
            **parameters,
            "held_out_packs": sorted(EXPECTED_PACK_COUNTS),
            "representation_scope": "冻结表示由第三轮模型基于全部四个训练包训练；本轮仅 LOF/阈值按包留一",
            "not_claimed": "端到端未见包泛化或现场虚警率",
            "labels_used": False,
            "test_data_used": False,
        },
        "implementation_sha256": provenance.source_digest(SOURCE_PATHS),
    }
    run_id = (
        "phase4_lopo_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + provenance.digest(identity)[:10]
    )
    output_dir = ROOT / "outputs" / "diagnostics" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    run_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "identity": identity,
    }
    manifest_path = output_dir / "manifest.json"
    provenance.write_manifest(manifest_path, run_manifest)

    try:
        summaries = []
        window_rows = []
        for pack_id in sorted(EXPECTED_PACK_COUNTS):
            summary, windows = calibrate_pack(
                features,
                ids,
                pack_id,
                **parameters,
            )
            summaries.append(summary)
            window_rows.extend(windows)

        summary_path = output_dir / "pack_summary.csv"
        windows_path = output_dir / "window_scores.csv"
        write_csv(summary_path, summaries)
        write_csv(windows_path, window_rows)
        run_manifest.update({
            "status": "complete",
            "results": summaries,
            "files": {
                "pack_summary.csv": provenance.file_digest(summary_path),
                "window_scores.csv": provenance.file_digest(windows_path),
            },
        })
        provenance.write_manifest(manifest_path, run_manifest)
    except Exception as exc:
        run_manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        provenance.write_manifest(manifest_path, run_manifest)
        raise

    print(f"[phase4] 输出：{output_dir}")
    for row in summaries:
        print(
            f"[phase4] pack {row['pack_id']}: "
            f"越限率={row['exceedance_rate']:.4%}, "
            f"持久确认窗口数={row['n_persistence_confirmed_windows']}, "
            f"首次确认报警窗口={row['first_confirmed_alarm_window']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
