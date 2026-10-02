"""只读审计已封存四折 LOPO：评分口径、跨包漂移、工况覆盖和报警形态。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batfd import console, provenance  # noqa: E402
from batfd.data.conditions import CONDITION_NAMES, compute_conditions  # noqa: E402
from batfd.detect.lof import LOFDetector, apply_persistence  # noqa: E402

_spec = importlib.util.spec_from_file_location("phase5_diagnostic_source", ROOT / "scripts/15_phase5_e2e_lopo.py")
PHASE5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PHASE5)
GROUPS = ("sigma_v", "one_minus_cos", "q3_err")
FEATURE_NAMES = [f"cell_{cell}_{group}" for cell in range(1, 9) for group in GROUPS]
SOURCE_PATHS = (*PHASE5.SOURCE_PATHS, "batfd/data/conditions.py", "scripts/16_phase6_lopo_shift_diagnostic.py")


def distribution_rows(calibration, held_out, names, center=None, scale=None):
    cal, held = np.asarray(calibration, dtype=float), np.asarray(held_out, dtype=float)
    if (cal.ndim != 2 or held.ndim != 2 or cal.shape[1] != held.shape[1]
            or len(names) != cal.shape[1] or not len(cal) or not len(held)
            or not np.isfinite(cal).all() or not np.isfinite(held).all()):
        raise ValueError("分布比较需要同维、非空、有限的校准和留出矩阵")
    center = np.median(cal, axis=0) if center is None else np.asarray(center)
    scale = np.maximum(np.median(np.abs(cal - center), axis=0) * 1.4826,
                       np.maximum(.001 * np.abs(center), 1e-12)) if scale is None else np.asarray(scale)
    if (center.shape != (cal.shape[1],) or scale.shape != center.shape
            or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("校准中心或尺度无效")
    cq, hq = np.quantile(cal, [.01, .5, .99], axis=0), np.quantile(held, [.01, .5, .99], axis=0)
    rows = []
    for j, name in enumerate(names):
        rows.append({
            "name": name, "cal_center": float(center[j]), "cal_scale": float(scale[j]),
            "cal_min": float(cal[:, j].min()), "cal_max": float(cal[:, j].max()),
            "cal_q01": float(cq[0, j]), "cal_median": float(cq[1, j]), "cal_q99": float(cq[2, j]),
            "held_q01": float(hq[0, j]), "held_median": float(hq[1, j]), "held_q99": float(hq[2, j]),
            "median_shift_in_cal_scale": float((hq[1, j] - center[j]) / scale[j]),
            "median_abs_held_deviation_in_cal_scale": float(np.median(np.abs(held[:, j] - center[j])) / scale[j]),
            "held_outside_cal_range_rate": float(((held[:, j] < cal[:, j].min()) | (held[:, j] > cal[:, j].max())).mean()),
            "held_outside_cal_01_99_rate": float(((held[:, j] < cq[0, j]) | (held[:, j] > cq[2, j])).mean()),
        })
    return rows


def alarm_shape(scores, threshold, persistence):
    scores = np.asarray(scores)
    if (scores.ndim != 1 or not scores.size or not np.isfinite(scores).all()
            or not np.isfinite(threshold) or persistence < 1 or int(persistence) != persistence):
        raise ValueError("分数、阈值或持久性参数无效")
    flag = scores > threshold
    confirmed = apply_persistence(flag, int(persistence))
    edges = np.diff(np.r_[False, flag, False].astype(int))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    hits = np.flatnonzero(confirmed)
    shape = {
        "n_windows": int(scores.size), "n_exceeded": int(flag.sum()), "exceedance_rate": float(flag.mean()),
        "n_confirmed": int(confirmed.sum()), "confirmed_rate": float(confirmed.mean()),
        "n_exceedance_runs": int(len(lengths)), "longest_exceedance_run": int(lengths.max()) if lengths.size else 0,
        "n_runs_at_least_persistence": int((lengths >= persistence).sum()),
        "first_confirmed_window": int(hits[0]) if hits.size else None,
    }
    bins = []
    for j, idx in enumerate(np.array_split(np.arange(len(scores)), 10)):
        if not idx.size:
            continue
        bins.append({"bin": j + 1, "start_window": int(idx[0]), "end_window_exclusive": int(idx[-1] + 1),
                     "n_windows": int(idx.size), "n_exceeded": int(flag[idx].sum()),
                     "exceedance_rate": float(flag[idx].mean()), "n_confirmed": int(confirmed[idx].sum()),
                     "confirmed_rate": float(confirmed[idx].mean())})
    return shape, bins


def audit_scores(calibration, held_out, cfg):
    detector = LOFDetector(n_neighbors=int(cfg["detect"]["lof_n_neighbors"]),
                           standardize=bool(cfg["detect"]["score_normalize"]),
                           norm_floor=float(cfg["model"]["norm_floor"])).fit(calibration)
    old_scores = detector.score(calibration).score
    standard_scores = -detector.model_.negative_outlier_factor_
    held_scores = detector.score(held_out).score
    q, persistence = float(cfg["detect"]["threshold_q"]), int(cfg["detect"]["persistence_windows"])
    old_threshold = detector.threshold_from_train(q)
    standard_threshold = float(np.quantile(standard_scores, q))
    old_shape, _ = alarm_shape(held_scores, old_threshold, persistence)
    standard_shape, _ = alarm_shape(held_scores, standard_threshold, persistence)
    return detector, {
        "q": q, "old_threshold": old_threshold, "standard_training_threshold": standard_threshold,
        "threshold_difference": standard_threshold - old_threshold,
        "training_score_max_abs_difference": float(np.max(np.abs(old_scores - standard_scores))),
        "old": old_shape, "standard_training": standard_shape,
        "interpretation": "标准训练口径只用于敏感性对照；不自动替换阈值或选择最终模型",
    }, standard_scores


def nearest_geometry(detector, calibration, held_out, calibration_ids):
    cal, held = detector.transform(calibration), detector.transform(held_out)
    ids = np.asarray(calibration_ids)
    if cal.shape[1] != 24 or held.shape[1] != 24 or ids.shape != (len(cal),):
        raise ValueError("邻居几何需要24维交错特征与校准IDs")
    energy = np.zeros((len(held), 3))
    pack_counts = {int(pid): 0 for pid in np.unique(ids)}
    kth = []
    # 仅查询已由校准数据拟合的邻居索引；分块避免 N×k×24 常驻。
    for start in range(0, len(held), 256):
        query = held[start:start + 256]
        distance, indices = detector.model_.kneighbors(query)
        delta = query[:, None, :] - cal[indices]
        energy[start:start + len(query)] = np.column_stack(
            [np.square(delta[:, :, j::3]).sum(axis=(1, 2)) for j in range(3)])
        kth.extend(distance[:, -1].tolist())
        for pid in pack_counts:
            pack_counts[pid] += int((ids[indices] == pid).sum())
    cal_distance, _ = detector.model_.kneighbors()  # 无X查询排除训练点自身。
    cal_kth = cal_distance[:, -1]
    squared, per_window = energy.sum(axis=0), energy.sum(axis=1)
    shares = np.divide(energy, per_window[:, None], out=np.zeros_like(energy), where=per_window[:, None] > 0)
    total, neighbors = float(squared.sum()), sum(pack_counts.values())
    top_count = max(1, int(np.ceil(.01 * len(held))))
    return {
        "n_neighbors": int(detector.model_.n_neighbors_),
        "median_kth_distance": float(np.median(kth)),
        "calibration_median_kth_distance": float(np.median(cal_kth)),
        "held_kth_above_cal_q99_rate": float((np.asarray(kth) > np.quantile(cal_kth, .99)).mean()),
        "squared_distance_group_share": {name: float(squared[j] / total) if total else 0. for j, name in enumerate(GROUPS)},
        "window_median_group_share": {name: float(np.median(shares[:, j])) for j, name in enumerate(GROUPS)},
        "top_1_percent_distance_energy_share": float(np.sort(per_window)[-top_count:].sum() / total) if total else 0.,
        "neighbor_pack_share": {str(pid): count / neighbors for pid, count in pack_counts.items()},
        "calibration_pack_share": {str(pid): float((ids == pid).mean()) for pid in pack_counts},
        "interpretation": "所有窗口及k邻居平方距离总和之比，可能受极端尾部支配；不是精确LOF贡献或故障定位",
    }


def rank_correlation(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)  # 仅描述性系数，不报告窗口独立样本p值。


def tree_hashes(directory):
    return {p.relative_to(directory).as_posix(): provenance.file_digest(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def checked_inputs(manifest):
    identity = manifest["identity"]
    if provenance.source_digest(PHASE5.SOURCE_PATHS) != identity["implementation_sha256"]:
        raise ValueError("原四折实现指纹已变，不能用新实现重放旧结果")
    directory = Path(identity["inputs"]["directory"]).resolve()
    names = ("signal.npy", "ids.npy", "meta.json")
    for name in names:
        if provenance.file_digest(directory / name) != identity["inputs"]["sha256"][name]:
            raise ValueError(f"独立输入来源哈希不一致：{name}")
    meta = provenance.read_manifest(directory / "meta.json")
    if meta is None or meta.get("name") != "StandTrainData" or meta.get("has_time"):
        raise ValueError("诊断只能使用无时间列的原StandTrainData")
    return directory, identity["config"]


def check_fold_protocol(cfg, summary):
    if (cfg["detect"]["persistence_windows"] != summary["persistence_windows"]
            or cfg["detect"]["threshold_q"] != summary["q"]):
        raise ValueError("根配置与每折保存的阈值分位/持久性协议不一致")


def run_diagnostic(run_dir, output_dir):
    directory, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output == directory or directory in output.parents:
        raise ValueError("诊断输出禁止放入旧四折产物树")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖：{output}")
    old_manifest = PHASE5.verify_run(directory)
    cache_dir, cfg = checked_inputs(old_manifest)
    if output == cache_dir or cache_dir in output.parents:
        raise ValueError("诊断输出禁止放入输入缓存树")
    before = tree_hashes(directory)
    implementation = provenance.source_digest(SOURCE_PATHS)
    ids = np.load(cache_dir / "ids.npy", allow_pickle=False)
    signal = np.load(cache_dir / "signal.npy", mmap_mode="r", allow_pickle=False)
    try:
        if ids.ndim != 1 or signal.shape != (len(ids), 256, 20):
            raise ValueError("原信号或IDs形状不符")
        PHASE5.split_outer_fold(ids, PHASE5.PACKS[0])
        conditions = np.empty((len(ids), 9), dtype=float)
        for start in range(0, len(ids), 256):
            values = signal[start:start + 256]
            if not np.isfinite(values).all():
                raise ValueError("信号含非有限值")
            conditions[start:start + len(values)] = compute_conditions(cfg, values, hidden=None)[:, :9]
    finally:
        signal._mmap.close()  # Windows文件映射必须释放，即使调用方保留切片引用。
    if not np.isfinite(conditions).all():
        raise ValueError("可观测工况含非有限值")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "status": "running", "source_run": str(directory),
                "source_tree_sha256": before, "implementation_sha256": implementation,
                "inputs": {"directory": str(cache_dir), "sha256": {
                    name: old_manifest["identity"]["inputs"]["sha256"][name]
                    for name in ("signal.npy", "ids.npy", "meta.json")}}}
    provenance.write_manifest(output / "manifest.json", manifest)
    packs, feature_rows, condition_rows, bins, geometry_rows = [], [], [], [], []
    try:
        for pid in PHASE5.PACKS:
            print(f"[诊断] 留出pack {pid}；仅拟合校准特征，不训练深度模型", flush=True)
            fold = directory / "folds" / str(pid)
            with np.load(fold / "split.npz", allow_pickle=False) as split:
                cal_idx, held_idx = split["calibration_idx"], split["held_out_idx"]
            expected_cal, expected_held = PHASE5.split_outer_fold(ids, pid)
            if not np.array_equal(cal_idx, expected_cal) or not np.array_equal(held_idx, expected_held):
                raise ValueError("四折索引与独立来源IDs不一致")
            cal = np.load(fold / "calibration_features.npy", allow_pickle=False)
            held = np.load(fold / "held_out_features.npy", allow_pickle=False)
            detector, audit, _ = audit_scores(cal, held, cfg)
            scores = detector.score(held).score
            differences = {}
            for label, replay in (("calibration", detector.score(cal).score), ("held_out", scores)):
                saved = np.load(fold / f"{label}_scores.npy", allow_pickle=False)
                differences[label] = float(np.max(np.abs(replay - saved)))
                if not np.array_equal(saved, replay):
                    raise ValueError(f"pack {pid} {label}原分数未精确重放")
            with np.load(fold / "lof_scaler.npz", allow_pickle=False) as scaler:
                if not np.array_equal(scaler["center"], detector.center_) or not np.array_equal(scaler["scale"], detector.scale_):
                    raise ValueError("原特征scaler未精确重放")
            saved_summary = provenance.read_manifest(fold / "summary.json")
            check_fold_protocol(cfg, saved_summary)
            if audit["old_threshold"] != saved_summary["calibration_threshold"]:
                raise ValueError("原阈值未精确重放")
            rows = distribution_rows(cal, held, FEATURE_NAMES, detector.center_, detector.scale_)
            feature_rows.extend({"pack_id": pid, "cell": j // 3 + 1, "metric": GROUPS[j % 3], **row}
                                for j, row in enumerate(rows))
            geometry = nearest_geometry(detector, cal, held, ids[cal_idx])
            geometry_rows.extend({"pack_id": pid, "metric": group, "squared_distance_share": share,
                                  "window_median_share": geometry["window_median_group_share"][group],
                                  "top_1_percent_energy_share": geometry["top_1_percent_distance_energy_share"]}
                                 for group, share in geometry["squared_distance_group_share"].items())
            threshold = audit["old_threshold"]
            cal_cond, held_cond = conditions[cal_idx], conditions[held_idx]
            crows = distribution_rows(cal_cond, held_cond, CONDITION_NAMES[:9])
            outside_any = np.zeros(len(held), dtype=bool)
            for j, row in enumerate(crows):
                outside = (held_cond[:, j] < row["cal_q01"]) | (held_cond[:, j] > row["cal_q99"])
                outside_any |= outside
                crows[j].update({"score_spearman": rank_correlation(held_cond[:, j], scores),
                    "abs_deviation_score_spearman": rank_correlation(np.abs(held_cond[:, j] - row["cal_center"]), scores),
                    "n_outside": int(outside.sum()),
                    "outside_exceedance_rate": float((scores[outside] > threshold).mean()) if outside.any() else None,
                    "inside_exceedance_rate": float((scores[~outside] > threshold).mean()) if (~outside).any() else None})
            condition_rows.extend({"pack_id": pid, **row} for row in crows)
            for definition, thr in (("old", threshold), ("standard_training", audit["standard_training_threshold"])):
                _, alarm_bins = alarm_shape(scores, thr, int(cfg["detect"]["persistence_windows"]))
                bins.extend({"pack_id": pid, "definition": definition, **row} for row in alarm_bins)
            packs.append({"pack_id": pid, "n_calibration_windows": len(cal), "n_held_out_windows": len(held),
                          "replay_max_abs_error": differences, "score_audit": audit, "geometry": geometry,
                          "condition_outside_any_01_99_rate": float(outside_any.mean()),
                          "condition_outside_exceedance_rate": float((scores[outside_any] > threshold).mean()) if outside_any.any() else None,
                          "condition_inside_exceedance_rate": float((scores[~outside_any] > threshold).mean()) if (~outside_any).any() else None})
        PHASE5.verify_run(directory)
        checked_inputs(old_manifest)
        if before != tree_hashes(directory) or implementation != provenance.source_digest(SOURCE_PATHS):
            raise ValueError("诊断期间旧产物树或源码发生变化")
        result = {"source_run": str(directory), "n_windows": len(ids), "source_unchanged": True, "packs": packs,
                  "protocol": {"hidden_read": False, "soc_used": False, "labels_used": False,
                               "test_data_used": False, "tuned_on_held_out": False, "deep_model_retrained": False,
                               "fit_scope": "每折其他三包校准特征", "order": "原始行顺序，不是真实时间轴"}}
        provenance.write_manifest(output / "summary.json", result)
        for name, rows in (("feature_shift.csv", feature_rows), ("condition_shift.csv", condition_rows),
                           ("alarm_bins.csv", bins), ("geometry.csv", geometry_rows)):
            PHASE5.write_csv(output / name, rows)
        (output / "report.md").write_text(render_report(result, feature_rows, condition_rows), encoding="utf-8")
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
        PHASE5.seal_manifest(output, manifest)
        verify_diagnostic(output)
        return result
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        provenance.write_manifest(output / "manifest.json", manifest)
        raise


def render_report(result, features, conditions):
    rows = ["# 四折 LOPO 只读根因诊断", "", "## 事实：原评分精确重放与训练阈值口径敏感性", "",
            "| 留出包 | 旧越限率 | 标准训练口径越限率 | 旧确认率 | 标准口径确认率 | 最长越限段 |",
            "|---|---:|---:|---:|---:|---:|"]
    for pack in result["packs"]:
        a = pack["score_audit"]
        rows.append(f'| {pack["pack_id"]} | {a["old"]["exceedance_rate"]:.2%} | {a["standard_training"]["exceedance_rate"]:.2%} | {a["old"]["confirmed_rate"]:.2%} | {a["standard_training"]["confirmed_rate"]:.2%} | {a["old"]["longest_exceedance_run"]} |')
    rows += ["", "评分差异仅在校准训练分数口径：旧实现用 novelty score_samples；标准训练LOF用 negative_outlier_factor_。留出评分不变。", "", "## 跨包特征与可观测工况", ""]
    for pack in result["packs"]:
        pid, geometry = pack["pack_id"], pack["geometry"]
        top = sorted((r for r in features if r["pack_id"] == pid), key=lambda r: abs(r["median_shift_in_cal_scale"]), reverse=True)[:4]
        corr = sorted((r for r in conditions if r["pack_id"] == pid and r["score_spearman"] is not None), key=lambda r: abs(r["score_spearman"]), reverse=True)[:3]
        rows += [f'### pack {pid}',
                 "- 校准邻居平方距离分组占比：" + "；".join(f"{name} {value:.2%}" for name, value in geometry["squared_distance_group_share"].items()) + "。",
                 "- 逐窗口分组占比中位数：" + "；".join(f"{name} {value:.2%}" for name, value in geometry["window_median_group_share"].items()) + "（各组中位数不必合计100%）。",
                 f'- 距离能量最高的约1%窗口占总能量：{geometry["top_1_percent_distance_energy_share"]:.2%}；上项总和占比不能解释为多数窗口的贡献。',
                 "- 中位数偏移最大的特征（校准尺度）：" + "；".join(f'{r["name"]} {r["median_shift_in_cal_scale"]:.3f}' for r in top) + "。",
                 f'- 任一工况超出校准1%–99%区间：{pack["condition_outside_any_01_99_rate"]:.2%}。',
                 "- 包内工况与分数描述性Spearman：" + "；".join(f'{r["name"]} {r["score_spearman"]:.3f}' for r in corr) + "。", ""]
    rows += ["## 解释边界与收益优先下一步", "",
             "- 评分口径差异是确定实现问题；标准训练口径对照后仍高越限时，不能将其认定为唯一根因。",
             "- 邻居平方距离不是LOF密度贡献；跨包偏移支持泛化/覆盖假设，但不证明具体故障单体或因果。",
             "- 工况边际范围内不等于联合覆盖充分，任一边际越界率不对应固定2%基准。",
             "- 工况范围越界和秩相关仅描述关联；不报告独立窗口p值，不以留出结果选择最终阈值或模型。",
             "- 越限率不是现场虚警率；窗口确认数不是报警事件数；原行顺序不是真实时间，不换算提前量。",
             "- 优先修正已证实的LOF训练分数口径，独立新目录复算检测链路，无需重训深度模型。",
             "- 随后在预先固定协议下做指标组受控消融（仅sigma_v、仅重建组、全部24维），不按本次留出结果择优定版。",
             "- 再根据覆盖证据决定是否需要重建模型/工况建模实验；不要先抬阈值来压低报警。", ""]
    return "\n".join(rows)


def verify_diagnostic(output_dir):
    output = Path(output_dir).resolve()
    manifest = provenance.read_manifest(output / "manifest.json")
    required = {"summary.json", "feature_shift.csv", "condition_shift.csv", "alarm_bins.csv", "geometry.csv", "report.md"}
    if (manifest is None or manifest.get("status") != "complete" or manifest.get("schema_version") != 1
            or set(manifest.get("files", {})) != required):
        raise ValueError("诊断尚未完整封存")
    actual = tree_hashes(output)
    actual.pop("manifest.json", None)
    if actual != manifest["files"]:
        raise ValueError("诊断产物SHA-256或文件集合不一致")
    if provenance.source_digest(SOURCE_PATHS) != manifest["implementation_sha256"]:
        raise ValueError("诊断实现指纹已变")
    source = Path(manifest["source_run"])
    old_manifest = PHASE5.verify_run(source)
    _, cfg = checked_inputs(old_manifest)
    if tree_hashes(source) != manifest["source_tree_sha256"]:
        raise ValueError("旧四折产物树已变")
    expected_inputs = old_manifest["identity"]["inputs"]
    if (Path(manifest["inputs"]["directory"]).resolve() != Path(expected_inputs["directory"]).resolve()
            or manifest["inputs"]["sha256"] != {name: expected_inputs["sha256"][name] for name in ("signal.npy", "ids.npy", "meta.json")}):
        raise ValueError("诊断输入身份与四折来源不一致")
    summary = provenance.read_manifest(output / "summary.json")
    if (summary is None or summary.get("source_unchanged") is not True
            or summary.get("n_windows") != old_manifest["summary"]["n_windows"]
            or summary.get("source_run") != str(source)
            or [p.get("pack_id") for p in summary.get("packs", [])] != list(PHASE5.PACKS)
            or any(summary.get("protocol", {}).get(key) is not False for key in (
                "hidden_read", "soc_used", "labels_used", "test_data_used", "tuned_on_held_out", "deep_model_retrained"))):
        raise ValueError("诊断摘要四折/窗口/来源/只读协议不符")
    expected_bins = []
    for pack in summary["packs"]:
        fold = source / "folds" / str(pack["pack_id"])
        saved = provenance.read_manifest(fold / "summary.json")
        check_fold_protocol(cfg, saved)
        audit = pack["score_audit"]
        if (pack["n_held_out_windows"] != saved["n_held_out_windows"]
                or pack["n_calibration_windows"] != saved["n_calibration_windows"]
                or audit["old_threshold"] != saved["calibration_threshold"] or audit["q"] != saved["q"]
                or any(value != 0. for value in pack["replay_max_abs_error"].values())):
            raise ValueError("诊断折摘要与已验收原实验不一致")
        scores = np.load(fold / "held_out_scores.npy", allow_pickle=False)
        for definition, thr in (("old", audit["old_threshold"]), ("standard_training", audit["standard_training_threshold"])):
            shape, bins = alarm_shape(scores, thr, saved["persistence_windows"])
            if audit[definition] != shape:
                raise ValueError("诊断报警数值与原分数不一致")
            expected_bins.extend({"pack_id": pack["pack_id"], "definition": definition, **row} for row in bins)
    for name, per_pack in (("feature_shift.csv", 24), ("condition_shift.csv", 9), ("geometry.csv", 3)):
        with (output / name).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 4 * per_pack or any(sum(r["pack_id"] == str(pid) for r in rows) != per_pack for pid in PHASE5.PACKS):
            raise ValueError(f"诊断表折集合/行数不符：{name}")
    with (output / "alarm_bins.csv").open(encoding="utf-8-sig", newline="") as handle:
        bins = list(csv.DictReader(handle))
    if bins != [{key: str(value) for key, value in row.items()} for row in expected_bins]:
        raise ValueError("报警分箱与独立复算数值不一致")
    return manifest


def main():
    console.setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="已完整验收的Phase5四折目录")
    parser.add_argument("--output-dir", type=Path, help="独立新目录，拒绝覆盖或嵌入旧树")
    parser.add_argument("--verify", type=Path, help="只读验证已封存诊断、旧产物和输入来源")
    args = parser.parse_args()
    if args.verify:
        manifest = verify_diagnostic(args.verify)
        print(json.dumps({"status": manifest["status"], "source_run": manifest["source_run"]}, ensure_ascii=False, indent=2))
        return 0
    if not args.run_dir or not args.output_dir:
        parser.error("诊断需要同时指定--run-dir和--output-dir")
    result = run_diagnostic(args.run_dir, args.output_dir)
    print(f'[诊断] 已封存 {len(result["packs"])} 折、{result["n_windows"]} 个留出窗口：{args.output_dir}', flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
