# SMD windowed protocol: same pipeline as benchmark.py, T>1 instead of the T=1 point baseline.
# Deliberately imports benchmark.py so the shared protocol is a structural fact, not a claim.
import argparse
from datetime import datetime
import csv
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

import chronoguard
from chronoguard.data import load_smd, make_windows, file_sha256, split_train

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import FRAMEWORK, ROOT, fit_model, predict_chunks, select_fit, write_json

from chronoguard.detector import METHODS
from chronoguard.evaluation import detection_metrics


def windowed_benchmark(args, out):
    rows = []
    inputs = {}
    for machine in args.machines:
        train, test, labels = load_smd(args.smd_dir, machine)
        fit, cal, val = split_train(train)
        # Window each role separately so no window spans a role boundary.
        fit_w, _ = make_windows(fit, args.window_size, args.stride)
        cal_w, _ = make_windows(cal, args.window_size, args.stride)
        val_w, _ = make_windows(val, args.window_size, args.stride)
        test_w, ends = make_windows(test, args.window_size, args.stride)
        labels_w = labels[ends]
        selected = select_fit(np.arange(len(fit_w)), args.max_fit)
        for method in METHODS:
            begin = time.perf_counter()
            model = fit_model(method, 'independent', fit_w[selected], cal_w)
            predicted, scores = predict_chunks(model, len(test_w), lambda sl: test_w[sl], machine)
            val_pred, _ = predict_chunks(model, len(val_w), lambda sl: val_w[sl], machine)
            metrics = detection_metrics(labels_w, predicted)
            row = dict(scenario='smd_windowed', entity=machine, method=method,
                       fit_available=len(fit_w), fit_used=len(selected), calibration_samples=len(cal_w),
                       validation_samples=len(val_w), test_samples=len(test_w), channels=38,
                       window_size=args.window_size, stride=args.stride,
                       test_warmup_points=args.window_size - args.stride,
                       threshold=model.threshold_, scale_fallback_features=int(model.scale_fallback_.sum()),
                       validation_alarm_rate=float(val_pred.mean()),
                       **metrics, seconds=round(time.perf_counter() - begin, 3))
            rows.append(row)
            print(json.dumps(row), flush=True)
            model.save(out / f'{machine}_{method}_t{args.window_size}_s{args.stride}.joblib')
            np.savez_compressed(out / f'{machine}_{method}_t{args.window_size}_s{args.stride}_predictions.npz',
                                scores=scores, confirmed=predicted, labels=labels_w, endpoints=ends)
        for group in ('train', 'test', 'test_label'):
            name = f'{group}/{machine}.txt'
            inputs['smd/' + name] = file_sha256(args.smd_dir / name)
    return rows, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smd-dir', type=Path, default=ROOT / 'datasets/smd')
    parser.add_argument('--machines', nargs='+', default=['machine-1-1', 'machine-2-1', 'machine-3-1'])
    parser.add_argument('--max-fit', type=int, default=6000,
                        help='Fixed chronological uniform fit sample cap; identical to benchmark.py')
    parser.add_argument('--window-size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.max_fit < 3:
        parser.error('--max-fit must be >=3')
    if args.window_size < 1 or args.stride < 1:
        parser.error('--window-size and --stride must be >=1')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = args.output_dir / 'source'
    source.mkdir()
    source_files = [*sorted((FRAMEWORK / 'src/chronoguard').glob('*.py')),
                    FRAMEWORK / 'examples/benchmark.py', Path(__file__), FRAMEWORK / 'pyproject.toml']
    hashes = {}
    for file in source_files:
        rel = file.relative_to(ROOT)
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
        hashes[rel.as_posix()] = file_sha256(file)
    manifest = dict(
        status='running',
        date=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        protocol=dict(fit_fraction=.6, calibration_fraction=.2, validation_fraction=.2,
                      quantile=.99, persistence=5, seed=42, max_fit=args.max_fit,
                      threshold_from_test_labels=False, point_adjustment=False,
                      smd_training_normality='assumed, not verified by train labels',
                      smd_window_size=args.window_size, smd_stride=args.stride,
                      role_boundary_windows='each role windowed separately; no window spans a boundary',
                      label_alignment='window_end via make_windows endpoints',
                      scale_policy='MAD then fit-std for degenerate varying features',
                      feature_mode='independent (38 heterogeneous channels)',
                      differs_from_benchmark_py='window size and stride only'),
        windowing_caveats=[
            f'adjacent windows share {args.window_size - args.stride} raw points; '
            'per-point metrics are autocorrelated and not independent samples',
            f'the first {args.window_size - args.stride} test points are warm-up and produce no output',
            'stride=1 is a choice, not a property of the data',
        ],
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      sklearn=sklearn.__version__, chronoguard=chronoguard.__version__),
        source_sha256=hashes,
        command=vars(args) | {'smd_dir': str(args.smd_dir), 'output_dir': str(args.output_dir)})
    write_json(args.output_dir / 'manifest.json', manifest)
    begin = time.perf_counter()
    with threadpool_limits(limits=2):
        rows, inputs = windowed_benchmark(args, args.output_dir)
    write_json(args.output_dir / 'summary.json', rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.output_dir / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest.update(status='complete', input_sha256=inputs, seconds=round(time.perf_counter() - begin, 3),
                    artifacts={f.relative_to(args.output_dir).as_posix(): file_sha256(f)
                               for f in sorted(args.output_dir.rglob('*'))
                               if f.is_file() and f.name != 'manifest.json'})
    write_json(args.output_dir / 'manifest.json', manifest)
    print('COMPLETE', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
