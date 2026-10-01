"""第一轮诊断：简单电压基线、逐包结果和跨包分布审计。

本脚本不训练、不修改深度模型，也不覆盖 outputs/tables 下的历史 CSV。
它只读取 train/test1 缓存和现有历史检测表，在 outputs/diagnostics/ 下生成带输入
指纹的不可覆盖诊断产物。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batfd import config, provenance  # noqa: E402
from batfd.baselines import simple  # noqa: E402
from batfd.data import cache, channels  # noqa: E402
from batfd.eval import metrics  # noqa: E402


SCORE_MODES = ("peer_spread", "peer_spread_ewma", "peer_spread_cusum")
LEGACY_TABLES = (
    "detection_lfaae_train_novelty_fixed-quantile_regression-binned_quantile-pack_baseline.csv",
    "detection_ours_full_train_novelty_fixed-quantile_regression-binned_quantile-pack_baseline.csv",
    "detection_ours_shared_latent_train_novelty_fixed-quantile_regression-binned_quantile-pack_baseline.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成第一轮简单基线与跨包偏移诊断产物")
    parser.add_argument("--q", type=float, default=None, help="训练/校准分位数，默认读取 configs/base.yaml")
    parser.add_argument("--persistence", type=int, default=None, help="连续越限窗口数，默认读取 configs/base.yaml")
    parser.add_argument("--baseline-frac", type=float, default=0.10, help="逐包自身基线占比")
    parser.add_argument("--ewma-alpha", type=float, default=0.10, help="EWMA 平滑系数")
    parser.add_argument("--cusum-slack", type=float, default=0.50, help="CUSUM 标准化分数的松弛项")
    return parser.parse_args()


def as_float(value: str | None) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    return float(value)


def as_int(value: str | None) -> int | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    return int(float(value))


def read_labels(path: Path) -> dict[int, dict[str, int | None]]:
    if not path.is_file():
        return {}
    out: dict[int, dict[str, int | None]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            out[int(row["pack_id"])] = {
                "onset_point": as_int(row.get("onset_point")),
                "n_train": as_int(row.get("n_train")),
            }
    return out


def pack_order(cache_dict: dict, pack_id: int) -> np.ndarray:
    return metrics.pack_order(cache_dict, pack_id)


def transform_by_pack(
    cache_dict: dict,
    raw: np.ndarray,
    mode: str,
    *,
    ewma_alpha: float,
    cusum_center: float,
    cusum_scale: float,
    cusum_slack: float,
) -> np.ndarray:
    if mode not in SCORE_MODES:
        raise ValueError(f"未知分数模式：{mode}")
    out = np.empty(len(raw), dtype=np.float64)
    ids = np.asarray(cache_dict["ids"])
    for pack_id in np.unique(ids):
        order = pack_order(cache_dict, int(pack_id))
        sequence = np.asarray(raw[order], dtype=np.float64)
        if mode == "peer_spread":
            transformed = sequence
        elif mode == "peer_spread_ewma":
            transformed = simple.ewma(sequence, alpha=ewma_alpha)
        else:
            transformed = simple.cusum(
                sequence,
                center=cusum_center,
                scale=cusum_scale,
                slack=cusum_slack,
            )
        out[order] = transformed
    return out


def load_scores(cfg: dict, train_cache: dict, test_cache: dict, args: argparse.Namespace) -> tuple[dict, dict, dict]:
    voltage_cols = channels.cell_voltage_cols(cfg)
    train_raw = simple.peer_spread(train_cache["signal"], voltage_cols)
    test_raw = simple.peer_spread(test_cache["signal"], voltage_cols)
    center, scale = simple.robust_center_scale(train_raw)
    train_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}
    for mode in SCORE_MODES:
        train_scores[mode] = transform_by_pack(
            train_cache,
            train_raw,
            mode,
            ewma_alpha=args.ewma_alpha,
            cusum_center=center,
            cusum_scale=scale,
            cusum_slack=args.cusum_slack,
        )
        test_scores[mode] = transform_by_pack(
            test_cache,
            test_raw,
            mode,
            ewma_alpha=args.ewma_alpha,
            cusum_center=center,
            cusum_scale=scale,
            cusum_slack=args.cusum_slack,
        )
    stats = {"raw_center": center, "raw_scale": scale}
    return train_scores, test_scores, stats


def onset_local(label: dict[str, int | None], n_windows: int) -> int | None:
    point = label.get("onset_point")
    n_train = label.get("n_train")
    if point is None or n_train is None:
        return None
    value = point - n_train
    return value if 0 <= value < n_windows else None


def simple_rows(
    test_cache: dict,
    labels: dict[int, dict[str, int | None]],
    train_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    *,
    q: float,
    persistence: int,
    baseline_frac: float,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    drift: list[dict] = []
    test_ids = np.asarray(test_cache["ids"])
    for mode in SCORE_MODES:
        train_threshold = float(np.quantile(train_scores[mode], q))
        for pack_id in sorted(np.unique(test_ids).astype(int).tolist()):
            order = pack_order(test_cache, pack_id)
            sequence = np.asarray(test_scores[mode][order], dtype=np.float64)
            time_sorted = None
            if test_cache.get("time") is not None:
                time_sorted = np.asarray(test_cache["time"])[order]
            calibration_n = max(1, int(round(len(sequence) * baseline_frac)))
            pack_threshold = float(np.quantile(sequence[:calibration_n], q))
            label = labels.get(pack_id, {})
            local_onset = onset_local(label, len(sequence))
            normal_end = local_onset if local_onset is not None and local_onset > 0 else calibration_n
            normal = sequence[:normal_end]

            for threshold_mode, threshold, calibration_windows in (
                ("train_global", train_threshold, 0),
                ("pack_baseline", pack_threshold, calibration_n),
            ):
                result = metrics.evaluate_pack(
                    pack_id,
                    sequence,
                    threshold,
                    persistence=persistence,
                    time_sorted=time_sorted,
                    onset_index=local_onset,
                    onset_source="outputs/tables/labels.csv",
                )
                rows.append({
                    "model": "simple",
                    "tag": "simple_voltage_baselines",
                    "family": "simple_baseline",
                    "score_method": mode,
                    "threshold_mode": threshold_mode,
                    "threshold_q": q,
                    "persistence_windows": persistence,
                    "pack_id": pack_id,
                    "n_windows": result.n_windows,
                    "alarm_day": result.alarm_day,
                    "onset_day": result.onset_day,
                    "lead_days": result.lead_days,
                    "alarm_ever": result.alarm_ever,
                    "early_detected": result.early_detected,
                    "n_confirmed_before_onset": result.n_confirmed_before_onset,
                    "windows_before_onset": result.n_windows_before_onset,
                    "trigger_rate_before_onset": result.trigger_rate_before_onset,
                    "threshold": threshold,
                    "calibration_windows": calibration_windows,
                    "source": "scripts/11_phase1_diagnostics.py",
                    "parameter_status": "phase1_exact",
                    "notes": ";".join(result.notes),
                })

            drift.append({
                "score_method": mode,
                "pack_id": pack_id,
                "n_normal_proxy": len(normal),
                "normal_proxy_end_index": normal_end,
                "train_threshold_q": train_threshold,
                "normal_q50": float(np.quantile(normal, 0.50)),
                "normal_q95": float(np.quantile(normal, 0.95)),
                "normal_q99": float(np.quantile(normal, 0.99)),
                "normal_fraction_above_train_threshold": float(np.mean(normal > train_threshold)),
            })
    return rows, drift


def legacy_rows(output_tables: Path, q: float, persistence: int) -> list[dict]:
    rows: list[dict] = []
    for filename in LEGACY_TABLES:
        path = output_tables / filename
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for item in csv.DictReader(handle):
                if item.get("dataset") != "StandTestData1" or item.get("threshold_method") != "fixed":
                    continue
                lead = as_float(item.get("lead_days"))
                onset = as_float(item.get("onset_day"))
                rows.append({
                    "model": item.get("tag", item.get("model", "legacy")),
                    "tag": item.get("tag", ""),
                    "family": "deep_legacy",
                    "score_method": "lof",
                    "threshold_mode": "train_global",
                    "threshold_q": "",
                    "persistence_windows": "",
                    "pack_id": item.get("pack_id", ""),
                    "n_windows": item.get("n_windows", ""),
                    "alarm_day": item.get("alarm_day", ""),
                    "onset_day": onset,
                    "lead_days": lead,
                    "alarm_ever": item.get("detected", ""),
                    "early_detected": None if lead is None else lead > 0.0,
                    "n_confirmed_before_onset": "",
                    "windows_before_onset": item.get("windows_before_onset", ""),
                    "trigger_rate_before_onset": item.get("far_per_window", ""),
                    "threshold": "",
                    "calibration_windows": "",
                    "source": str(path.resolve()),
                    "parameter_status": (
                        f"legacy_csv未内嵌q/m；本轮简单基线使用q={q},m={persistence}，"
                        "历史行不重新解释"
                    ),
                    "notes": item.get("notes", ""),
                })
    return rows


def add_condition_drift(
    drift_rows: list[dict],
    test_cache: dict,
    labels: dict[int, dict[str, int | None]],
) -> None:
    names = list(test_cache["meta"].get("condition_names", []))
    cond = np.asarray(test_cache["cond"])
    ids = np.asarray(test_cache["ids"])
    for row in drift_rows:
        pack_id = int(row["pack_id"])
        order = pack_order(test_cache, pack_id)
        label = labels.get(pack_id, {})
        local = onset_local(label, len(order))
        end = local if local is not None and local > 0 else max(1, int(row["n_normal_proxy"]))
        values = cond[order[:end]]
        for j, name in enumerate(names):
            finite = values[:, j][np.isfinite(values[:, j])]
            row[f"normal_{name}_median"] = "" if len(finite) == 0 else float(np.median(finite))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"拒绝写空表：{path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    text_lines: list[str] = []
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue().encode("utf-8-sig")
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"诊断产物内容冲突，拒绝覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(payload)


def write_json_once(
    path: Path,
    payload: dict,
    *,
    preserve_keys: tuple[str, ...] = (),
) -> None:
    """只创建或验证 JSON 产物，避免重复运行覆盖已有内容。"""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError(f"JSON 产物不是对象：{path}")
        for key in preserve_keys:
            if key in existing:
                payload[key] = existing[key]
        expected = json.dumps(payload, ensure_ascii=False, indent=2)
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"JSON 诊断产物内容冲突，拒绝覆盖：{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def input_manifest() -> dict:
    root = Path(__file__).resolve().parent.parent
    files = [root / "configs" / "base.yaml", root / "outputs" / "tables" / "labels.csv"]
    for dataset in ("StandTrainData", "StandTestData1"):
        files.extend((root / "outputs" / "cache" / dataset).glob("*.npy"))
        files.append(root / "outputs" / "cache" / dataset / "meta.json")
    files = [path for path in files if path.is_file()]
    return {str(path.relative_to(root)).replace("\\", "/"): provenance.file_digest(path) for path in sorted(files)}


def summarize(rows: list[dict]) -> list[dict]:
    summary: list[dict] = []
    keys = sorted({(r["family"], r["model"], r["score_method"], r["threshold_mode"]) for r in rows})
    for family, model, score_method, threshold_mode in keys:
        subset = [
            r for r in rows
            if (r["family"], r["model"], r["score_method"], r["threshold_mode"])
            == (family, model, score_method, threshold_mode)
        ]
        valid_early = [r for r in subset if r["early_detected"] is not None]
        early = sum(bool(r["early_detected"]) for r in valid_early)
        leads = [float(r["lead_days"]) for r in valid_early if r["lead_days"] is not None and bool(r["early_detected"])]
        triggers = [float(r["trigger_rate_before_onset"]) for r in subset if r["trigger_rate_before_onset"] not in (None, "")]
        summary.append({
            "family": family,
            "model": model,
            "score_method": score_method,
            "threshold_mode": threshold_mode,
            "n_packs": len(subset),
            "early_detected": early,
            "early_detection_rate": None if not valid_early else early / len(valid_early),
            "median_positive_lead_days": None if not leads else float(np.median(leads)),
            "mean_trigger_rate_before_onset": None if not triggers else float(np.mean(triggers)),
        })
    return summary


def main() -> int:
    args = parse_args()
    cfg = config.load()
    q = float(args.q if args.q is not None else cfg["detect"]["threshold_q"])
    persistence = int(args.persistence if args.persistence is not None else cfg["detect"]["persistence_windows"])
    if not 0.0 < q < 1.0:
        raise ValueError(f"q 应在 (0,1)，收到 {q}")
    if persistence < 1:
        raise ValueError(f"persistence 必须为正数，收到 {persistence}")
    if not 0.0 < args.baseline_frac < 1.0:
        raise ValueError(f"baseline_frac 应在 (0,1)，收到 {args.baseline_frac}")

    train_cache = cache.load_cache(cfg, "StandTrainData")
    test_cache = cache.load_cache(cfg, "StandTestData1")
    labels_path = Path(cfg["paths"]["outputs_dir"]) / "tables" / "labels.csv"
    labels = read_labels(labels_path)
    train_scores, test_scores, score_stats = load_scores(cfg, train_cache, test_cache, args)
    simple_result_rows, drift_rows = simple_rows(
        test_cache,
        labels,
        train_scores,
        test_scores,
        q=q,
        persistence=persistence,
        baseline_frac=args.baseline_frac,
    )
    add_condition_drift(drift_rows, test_cache, labels)
    comparison_rows = legacy_rows(Path(cfg["paths"]["outputs_dir"]) / "tables", q, persistence)
    comparison_rows.extend(simple_result_rows)

    parameters = {
        "q": q,
        "persistence_windows": persistence,
        "baseline_frac": args.baseline_frac,
        "ewma_alpha": args.ewma_alpha,
        "cusum_slack": args.cusum_slack,
        "score_modes": list(SCORE_MODES),
        "onset_source": str(labels_path.resolve()),
        "protocol_note": "深度历史行读取旧 CSV；简单基线为本轮精确参数；不重训深度模型",
    }
    identity = {
        "schema_version": 1,
        "parameters": parameters,
        "inputs": input_manifest(),
        "implementation": {
            "simple": provenance.file_digest(Path(__file__).resolve().parent.parent / "batfd" / "baselines" / "simple.py"),
            "script": provenance.file_digest(Path(__file__).resolve()),
        },
        "git_commit": git_head(),
    }
    run_id = provenance.digest(identity)[:20]
    out_dir = Path(cfg["paths"]["outputs_dir"]) / "diagnostics" / f"phase1_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / "comparison.csv"
    drift_path = out_dir / "cross_pack_drift.csv"
    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "manifest.json"
    write_csv(comparison_path, comparison_rows)
    write_csv(drift_path, drift_rows)
    summary = {
        "run_id": run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameters": parameters,
        "score_stats": score_stats,
        "comparison": summarize(comparison_rows),
        "outputs": {
            "comparison": str(comparison_path.resolve()),
            "cross_pack_drift": str(drift_path.resolve()),
        },
    }
    write_json_once(summary_path, summary, preserve_keys=("generated_utc",))
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "identity": identity,
        "files": {
            "comparison.csv": provenance.file_digest(comparison_path),
            "cross_pack_drift.csv": provenance.file_digest(drift_path),
            "summary.json": provenance.file_digest(summary_path),
        },
    }
    write_json_once(manifest_path, manifest)

    print(f"诊断产物目录：{out_dir.resolve()}")
    print(f"逐包比较表：{comparison_path.resolve()}")
    print(f"跨包偏移表：{drift_path.resolve()}")
    print("\n汇总：")
    for row in summary["comparison"]:
        print(
            f"  {row['model']:>22} {row['score_method']:>20} {row['threshold_mode']:>13} "
            f"early={row['early_detected']}/{row['n_packs']} "
            f"lead={row['median_positive_lead_days']} "
            f"trigger={row['mean_trigger_rate_before_onset']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
