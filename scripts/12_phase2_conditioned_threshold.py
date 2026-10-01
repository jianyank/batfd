"""第二轮诊断：跨包、工况条件化阈值。

只用 StandTrainData 拟合 peer-spread 阈值，在 StandTestData1 上按包评估 fixed、
quantile_regression 和 binned_quantile。条件化方法统一将 score - q_hat(condition)
作为异常统计量，阈值固定为 0。本脚本不训练、不修改 checkpoint、不覆盖历史 tables，
产物写入 outputs/diagnostics/phase2_<run_id>/ 并带输入及输出指纹。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batfd import config, provenance  # noqa: E402
from batfd.baselines import simple  # noqa: E402
from batfd.data import cache, channels  # noqa: E402
from batfd.detect import threshold  # noqa: E402
from batfd.eval import metrics  # noqa: E402

OBSERVABLE_CONDITION_NAMES = (
    "mean_abs_i", "std_i", "p95_abs_i", "disch_frac", "mean_v",
    "std_v", "range_v", "mean_t", "std_t",
)
METHODS = ("fixed", "quantile_regression", "binned_quantile")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成第二轮工况条件化阈值诊断产物")
    parser.add_argument("--q", type=float, default=None, help="训练分位数，默认读取 configs/base.yaml")
    parser.add_argument("--persistence", type=int, default=None, help="连续越限窗口数，默认读取 configs/base.yaml")
    parser.add_argument("--binned-bins", type=int, default=10, help="binned_quantile 分箱数")
    return parser.parse_args()


def as_int(value: str | None) -> int | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    return int(float(value))


def read_labels(path: Path) -> dict[int, dict[str, int | None]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            int(row["pack_id"]): {
                "onset_point": as_int(row.get("onset_point")),
                "n_train": as_int(row.get("n_train")),
            }
            for row in csv.DictReader(handle)
        }


def onset_local(label: dict[str, int | None], n_windows: int) -> int | None:
    point, n_train = label.get("onset_point"), label.get("n_train")
    if point is None or n_train is None:
        return None
    value = point - n_train
    return value if 0 <= value < n_windows else None


def pack_order(cache_dict: dict, pack_id: int) -> np.ndarray:
    return metrics.pack_order(cache_dict, pack_id)


def observable_conditions(cache_dict: dict) -> tuple[np.ndarray, list[str]]:
    """按字段名选部署可观测工况，并显式排除 mean_soc。"""
    names = list(cache_dict["meta"].get("condition_names", []))
    index = {name: i for i, name in enumerate(names)}
    missing = [name for name in OBSERVABLE_CONDITION_NAMES if name not in index]
    if missing:
        raise ValueError(f"缓存缺少部署工况字段：{missing}")
    cols = [index[name] for name in OBSERVABLE_CONDITION_NAMES]
    return np.asarray(cache_dict["cond"])[:, cols], list(OBSERVABLE_CONDITION_NAMES)


def finite_condition_rows(cond: np.ndarray) -> np.ndarray:
    return np.isfinite(cond).all(axis=1)


def load_scores(cfg: dict, train_cache: dict, test_cache: dict) -> tuple[np.ndarray, np.ndarray]:
    cols = channels.cell_voltage_cols(cfg)
    train = simple.peer_spread(train_cache["signal"], cols)
    test = simple.peer_spread(test_cache["signal"], cols)
    return np.asarray(train, dtype=np.float64), np.asarray(test, dtype=np.float64)


def fit_models(train_score: np.ndarray, train_cond: np.ndarray, q: float, bins: int):
    return {
        "fixed": threshold.fit_fixed(train_score, q),
        "quantile_regression": threshold.fit_quantile_regression(train_score, train_cond, q),
        "binned_quantile": threshold.fit_binned_quantile(train_score, train_cond, q, n_bins=bins),
    }


def finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": int(len(finite)),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def condition_coverage_rows(
    train_cond: np.ndarray,
    test_cache: dict,
    test_cond: np.ndarray,
    names: list[str],
) -> list[dict]:
    train_ranges = []
    for j, name in enumerate(names):
        values = np.asarray(train_cond[:, j], dtype=np.float64)
        finite = values[np.isfinite(values)]
        train_ranges.append((name, finite))

    rows = []
    ids = np.asarray(test_cache["ids"])
    for pack_id in sorted(np.unique(ids).astype(int).tolist()):
        order = pack_order(test_cache, pack_id)
        cond = np.asarray(test_cond[order], dtype=np.float64)
        for j, (name, train_values) in enumerate(train_ranges):
            values = cond[:, j]
            finite = np.isfinite(values)
            test_values = values[finite]
            train_min = None if len(train_values) == 0 else float(np.min(train_values))
            train_max = None if len(train_values) == 0 else float(np.max(train_values))
            outside = (
                finite & ((values < train_min) | (values > train_max))
                if train_min is not None
                else np.zeros(len(values), dtype=bool)
            )
            rows.append({
                "pack_id": pack_id,
                "condition": name,
                "n_test_windows": int(len(values)),
                "n_test_finite": int(finite.sum()),
                "test_finite_fraction": float(finite.mean()) if len(values) else 0.0,
                "train_min": train_min,
                "train_max": train_max,
                "test_min": None if len(test_values) == 0 else float(np.min(test_values)),
                "test_median": None if len(test_values) == 0 else float(np.median(test_values)),
                "test_max": None if len(test_values) == 0 else float(np.max(test_values)),
                "n_test_outside_train_range": int(outside.sum()),
                "test_outside_train_range_fraction": (
                    float(outside.sum() / finite.sum()) if finite.sum() else None
                ),
            })
    return rows


def result_rows(
    models: dict,
    test_score: np.ndarray,
    test_cache: dict,
    test_cond: np.ndarray,
    labels: dict[int, dict[str, int | None]],
    persistence: int,
    q: float,
) -> list[dict]:
    rows = []
    ids = np.asarray(test_cache["ids"])
    for method, model in models.items():
        for pack_id in sorted(np.unique(ids).astype(int).tolist()):
            order = pack_order(test_cache, pack_id)
            score = np.asarray(test_score[order], dtype=np.float64)
            cond = np.asarray(test_cond[order], dtype=np.float64)
            cond_finite = finite_condition_rows(cond)
            stat = np.full(len(score), np.nan, dtype=np.float64)
            if method == "fixed":
                stat = model.stat(score)
            elif cond_finite.any():
                stat[cond_finite] = model.stat(score[cond_finite], cond[cond_finite])
            qhat = score - stat
            label = labels.get(pack_id, {})
            local_onset = onset_local(label, len(order))
            time_sorted = (
                np.asarray(test_cache["time"])[order]
                if test_cache.get("time") is not None
                else None
            )
            result = metrics.evaluate_pack(
                pack_id,
                stat,
                0.0,
                persistence=persistence,
                time_sorted=time_sorted,
                onset_index=local_onset,
                onset_source="outputs/tables/labels.csv",
            )
            stat_summary = finite_summary(stat)
            qhat_summary = finite_summary(qhat)
            rows.append({
                "model": "simple",
                "tag": "simple_voltage_conditioned_threshold",
                "family": "simple_baseline",
                "score_method": "peer_spread",
                "threshold_mode": method,
                "threshold_q": q,
                "threshold_decision": 0.0,
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
                "condition_finite_fraction": float(cond_finite.mean()) if len(cond_finite) else 0.0,
                "stat_finite_fraction": float(np.isfinite(stat).mean()) if len(stat) else 0.0,
                "qhat_min": qhat_summary["min"],
                "qhat_median": qhat_summary["median"],
                "qhat_max": qhat_summary["max"],
                "exceedance_p95": (
                    float(np.nanpercentile(stat, 95)) if stat_summary["n"] else None
                ),
                "exceedance_max": stat_summary["max"],
                "source": "scripts/12_phase2_conditioned_threshold.py",
                "parameter_status": "phase2_exact",
                "notes": ";".join(result.notes),
            })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for method in METHODS:
        subset = [row for row in rows if row["threshold_mode"] == method]
        labelled = [row for row in subset if row["early_detected"] is not None]
        early = sum(bool(row["early_detected"]) for row in labelled)
        leads = [
            float(row["lead_days"])
            for row in labelled
            if row["early_detected"] and row["lead_days"] is not None
        ]
        triggers = [
            float(row["trigger_rate_before_onset"])
            for row in labelled
            if row["trigger_rate_before_onset"] is not None
            and np.isfinite(float(row["trigger_rate_before_onset"]))
        ]
        output.append({
            "family": "simple_baseline",
            "model": "simple",
            "score_method": "peer_spread",
            "threshold_mode": method,
            "n_packs": len(subset),
            "early_detected": early,
            "early_detection_rate": early / len(labelled) if labelled else None,
            "median_positive_lead_days": float(np.median(leads)) if leads else None,
            "mean_trigger_rate_before_onset": float(np.mean(triggers)) if triggers else None,
            "max_trigger_rate_before_onset": float(np.max(triggers)) if triggers else None,
        })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"拒绝写空表：{path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue().encode("utf-8-sig")
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"诊断产物内容冲突，拒绝覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(payload)


def write_json_once(path: Path, payload: dict, *, preserve_keys: tuple[str, ...] = ()):
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
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


def input_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    files = [root / "configs" / "base.yaml", root / "outputs" / "tables" / "labels.csv"]
    for dataset in ("StandTrainData", "StandTestData1"):
        files.extend((root / "outputs" / "cache" / dataset).glob("*.npy"))
        files.append(root / "outputs" / "cache" / dataset / "meta.json")
    return {
        str(path.relative_to(root)).replace("\\", "/"): provenance.file_digest(path)
        for path in sorted(path for path in files if path.is_file())
    }


def main() -> int:
    args = parse_args()
    cfg = config.load()
    q = float(args.q if args.q is not None else cfg["detect"]["threshold_q"])
    persistence = int(
        args.persistence
        if args.persistence is not None
        else cfg["detect"]["persistence_windows"]
    )
    if not 0.0 < q < 1.0:
        raise ValueError(f"q 应在 (0,1)，收到 {q}")
    if persistence < 1:
        raise ValueError(f"persistence 必须为正数，收到 {persistence}")
    if args.binned_bins < 2:
        raise ValueError(f"binned_bins 至少为 2，收到 {args.binned_bins}")

    train_cache = cache.load_cache(cfg, "StandTrainData")
    test_cache = cache.load_cache(cfg, "StandTestData1")
    labels_path = Path(cfg["paths"]["outputs_dir"]) / "tables" / "labels.csv"
    labels = read_labels(labels_path)
    train_score, test_score = load_scores(cfg, train_cache, test_cache)
    train_cond, names = observable_conditions(train_cache)
    test_cond, test_names = observable_conditions(test_cache)
    if names != test_names:
        raise ValueError("训练和测试工况字段不一致")
    if "mean_soc" in names:
        raise AssertionError("部署条件化阈值不应使用 mean_soc")
    train_finite = finite_condition_rows(train_cond)
    if not train_finite.any():
        raise ValueError("训练工况没有完整有限行，无法拟合条件阈值")

    models = fit_models(train_score, train_cond, q, args.binned_bins)
    rows = result_rows(models, test_score, test_cache, test_cond, labels, persistence, q)
    coverage = condition_coverage_rows(train_cond, test_cache, test_cond, names)
    parameters = {
        "q": q, "persistence_windows": persistence, "score_method": "peer_spread",
        "threshold_methods": list(METHODS), "binned_bins": args.binned_bins,
        "condition_names": names, "excluded_conditions": ["mean_soc"],
        "fit_dataset": "StandTrainData", "eval_dataset": "StandTestData1",
        "fit_rows": int(train_finite.sum()), "fit_rows_total": int(len(train_cond)),
        "protocol_note": "条件阈值只在 StandTrainData 拟合；测试起点标签仅用于最终评价；stat=score-q_hat(condition)，阈值为 0；mean_soc 排除。",
    }
    root = Path(__file__).resolve().parent.parent
    identity = {
        "schema_version": 1, "parameters": parameters, "inputs": input_manifest(),
        "implementation": {
            name: provenance.file_digest(root / path)
            for name, path in {
                "simple": "batfd/baselines/simple.py", "threshold": "batfd/detect/threshold.py",
                "conditions": "batfd/data/conditions.py", "metrics": "batfd/eval/metrics.py",
                "script": "scripts/12_phase2_conditioned_threshold.py",
            }.items()
        },
        "git_commit": git_head(),
    }
    run_id = provenance.digest(identity)[:20]
    out_dir = Path(cfg["paths"]["outputs_dir"]) / "diagnostics" / f"phase2_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path, coverage_path = out_dir / "comparison.csv", out_dir / "condition_coverage.csv"
    summary_path, manifest_path = out_dir / "summary.json", out_dir / "manifest.json"
    write_csv(comparison_path, rows)
    write_csv(coverage_path, coverage)
    outside = [
        float(row["test_outside_train_range_fraction"])
        for row in coverage
        if row["test_outside_train_range_fraction"] is not None
    ]
    summary = {
        "run_id": run_id, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameters": parameters, "comparison": summarize(rows),
        "condition_coverage": {"n_rows": len(coverage), "max_outside_fraction": max(outside) if outside else None},
        "outputs": {"comparison": str(comparison_path.resolve()), "condition_coverage": str(coverage_path.resolve())},
    }
    write_json_once(summary_path, summary, preserve_keys=("generated_utc",))
    manifest = {
        "schema_version": 1, "run_id": run_id, "identity": identity,
        "files": {
            "comparison.csv": provenance.file_digest(comparison_path),
            "condition_coverage.csv": provenance.file_digest(coverage_path),
            "summary.json": provenance.file_digest(summary_path),
        },
    }
    write_json_once(manifest_path, manifest)
    print(f"诊断产物目录：{out_dir.resolve()}")
    print(f"逐包比较表：{comparison_path.resolve()}")
    print(f"工况覆盖表：{coverage_path.resolve()}")
    print("\n汇总：")
    for row in summary["comparison"]:
        print(
            f"  {row['threshold_mode']:>20} "
            f"early={row['early_detected']}/{row['n_packs']} "
            f"lead={row['median_positive_lead_days']} "
            f"trigger={row['mean_trigger_rate_before_onset']}"
        )
    print(
        "  condition_ood_max="
        f"{summary['condition_coverage']['max_outside_fraction']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
