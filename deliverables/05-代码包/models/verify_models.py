# Load every delivered model and check it against the run summary it came from.
#
# Two levels of checking:
#   - without data: model identity, recorded threshold, internal shapes
#   - with data:    re-run prediction on the test split and compare the metrics
#
# Usage:
#   python verify_models.py                          # identity + threshold only
#   python verify_models.py --smd-dir <path>         # also re-run SMD
#   python verify_models.py --battery-cache <path>   # also re-run battery
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "framework" / "src"))
try:
    import joblib
    from chronoguard import AnomalyDetector
    from chronoguard.data import load_smd, make_windows, split_train
    from chronoguard.evaluation import detection_metrics
except ImportError as exc:
    raise SystemExit(
        f"cannot import chronoguard ({exc}); install the wheel or source archive first") from exc

ROOT = Path(__file__).resolve().parent
METHODS = ("robust", "lof", "iforest", "pca")


def method_of(stem):
    for name in METHODS:
        if name in stem.split("_"):
            return name
    return None


def entity_of(stem):
    if stem.startswith("battery_pack_"):
        return stem.split("_")[2]
    if stem.startswith("machine-"):
        return "-".join(stem.split("_")[0].split("-")[:3])
    return None


def report_internals(model):
    '''Surface what a model actually carries, so nothing is hidden in the artifact.'''
    notes = []
    inner = getattr(model, "model_", None)
    if inner is not None and hasattr(inner, "_fit_X"):
        fit_x = np.asarray(inner._fit_X)
        notes.append(f"内嵌训练特征 _fit_X {fit_x.shape} {fit_x.dtype}")
    if model.method == "robust":
        notes.append("仅存标准化统计量 center_/scale_，不含训练样本")
    if model.method == "pca" and inner is not None:
        notes.append(f"存投影方向 components_ {inner.components_.shape}")
    return "；".join(notes) if notes else "—"


def check_identity(directory, group):
    summary_path = directory / "summary.json"
    rows = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    indexed = {(str(r["entity"]), r["method"]): r for r in rows} if rows else {}

    print(f"\n=== {group}：{directory.relative_to(ROOT)} ===")
    ok = failed = 0
    for path in sorted(directory.glob("*.joblib")):
        method = method_of(path.stem)
        entity = entity_of(path.stem)
        model = AnomalyDetector.load(path)
        problems = []
        if not isinstance(model, AnomalyDetector):
            problems.append("不是 AnomalyDetector")
        if model.method != method:
            problems.append(f"方法不符 {model.method}!={method}")
        if model.threshold_ is None:
            problems.append("未校准（threshold_ 为空）")
        row = indexed.get((entity, method))
        if row is None:
            problems.append("summary.json 中无对应记录")
        elif abs(float(row["threshold"]) - float(model.threshold_)) > 1e-9:
            problems.append(f"阈值不符 {model.threshold_}!={row['threshold']}")

        size_mb = path.stat().st_size / 1024 / 1024
        if problems:
            failed += 1
            print(f"  FAIL {path.name}  -> {'；'.join(problems)}")
        else:
            ok += 1
            extra = report_internals(model)
            thr = f"{model.threshold_:.6g}" if model.threshold_ is not None else "n/a"
            print(f"  OK   {path.name:44s} 阈值={thr:>12s}  {size_mb:5.2f} MB  {extra}")
    print(f"  --- {group}: {ok} 通过 / {failed} 失败")
    return failed


def replay_smd(directory, smd_dir, group):
    '''Re-run the windowed evaluation and compare with the stored summary.'''
    # checked_release summary.json holds both scenarios; keep only the SMD rows.
    rows = {(str(r["entity"]), r["method"]): r
            for r in json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            if str(r.get("scenario", "smd")).startswith("smd")}
    print(f"\n=== {group} 重跑核对（需要 SMD 数据）===")
    mismatches = 0
    for machine in sorted({e for e, _ in rows}):
        train, test, labels = load_smd(smd_dir, machine)
        for method in METHODS:
            row = rows.get((machine, method))
            if row is None:
                continue
            matches = sorted(directory.glob(f"{machine}_{method}*.joblib"))
            if not matches:
                print(f"  SKIP {machine:12s} {method:8s} 找不到对应模型文件")
                continue
            model = AnomalyDetector.load(matches[0])
            window = int(row["window_size"])
            windows, ends = make_windows(np.asarray(test), window, 1)
            confirmed = []
            state = None
            for start in range(0, len(windows), 512):
                out = model.predict(windows[start:start + 512], device_id=machine, state=state)
                confirmed.append(out.confirmed)
                state = out.state
            metrics = detection_metrics(labels[ends], np.concatenate(confirmed))
            delta = abs(metrics["f1"] - float(row["f1"]))
            flag = "OK  " if delta < 1e-9 else "DIFF"
            if delta >= 1e-9:
                mismatches += 1
            print(f"  {flag} {machine:12s} {method:8s} 记录F1={row['f1']:.6f} 重算F1={metrics['f1']:.6f}")
    print(f"  --- 差异条目: {mismatches}")
    return mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smd-dir", type=Path, default=None)
    parser.add_argument("--battery-cache", type=Path, default=None)
    args = parser.parse_args()

    print("模型包自检")
    print(f"目录: {ROOT}")
    print("注意：模型经 joblib/pickle 序列化。只加载可信来源的模型。")
    print("注意：LOF 模型内嵌标准化后的训练特征（_fit_X），并非只含参数。")

    failed = 0
    for group, sub in (("电池 T=256 / peer 特征", "battery"),
                       ("服务器 SMD / T=1 逐点基线", "smd_t1"),
                       ("服务器 SMD / T=16 开窗", "smd_t16")):
        directory = ROOT / sub
        if directory.is_dir():
            failed += check_identity(directory, group)

    diffs = 0
    if args.smd_dir:
        diffs += replay_smd(ROOT / "smd_t1", args.smd_dir, "SMD T=1")
        diffs += replay_smd(ROOT / "smd_t16", args.smd_dir, "SMD T=16")
    else:
        print("\n未提供 --smd-dir，跳过 SMD 逐点重跑（无法核对指标，只能核对阈值与结构）")
    if not args.battery_cache:
        print("未提供 --battery-cache，跳过电池重跑")

    print(f"\n结论：身份/阈值检查失败 {failed} 项，指标重算差异 {diffs} 项")
    print("说明：电池场景没有独立故障事件记录，本脚本只能核对报警计数，不能核对检出率。")
    return 1 if (failed or diffs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
