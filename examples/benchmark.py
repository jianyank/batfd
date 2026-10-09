# Fixed protocol: no parameter or threshold selection using held-out test labels.
import argparse
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

import chronoguard
from chronoguard import AnomalyDetector, battery_windows
from chronoguard.data import download_smd, load_smd, split_train, file_sha256
from chronoguard.detector import METHODS
from chronoguard.evaluation import detection_metrics, normal_metrics

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _repo_root():
    '''Anchor for the datasets/ directory: the package root when running from the
    source tree, otherwise the current directory so an extracted copy still runs.
    '''
    return PACKAGE_ROOT if (PACKAGE_ROOT / 'datasets').is_dir() else Path.cwd()


ROOT = _repo_root()


def write_json(path, content):
    Path(path).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def select_fit(indices, cap):
    indices = np.asarray(indices)
    return indices if len(indices) <= cap else indices[np.linspace(0, len(indices) - 1, cap, dtype=int)]


def battery_input(signal, indices):
    # Select channels before converting to float64 to avoid materializing unused columns.
    return battery_windows(np.asarray(signal[indices]))


def fit_model(method, mode, fit, cal):
    return AnomalyDetector(method, feature_mode=mode, quantile=.99, persistence=5,
                           random_state=42).fit(fit).calibrate(cal)


def predict_chunks(model, size, getter, device, chunk_size=512):
    flags, scores = [], []
    state = None
    for start in range(0, size, chunk_size):
        out = model.predict(getter(slice(start, min(start + chunk_size, size))), device_id=device, state=state)
        flags.append(out.confirmed); scores.append(out.scores); state = out.state
    return np.concatenate(flags), np.concatenate(scores)


def battery_benchmark(args, out):
    cache = args.battery_cache
    signal = np.load(cache/'signal.npy', mmap_mode='r', allow_pickle=False)
    ids = np.load(cache/'ids.npy', allow_pickle=False)
    if signal.ndim != 3 or signal.shape[1:] != (256,20) or ids.shape != (len(signal),):
        raise ValueError('expected battery cache signal=(N,256,20) and aligned ids')
    packs = sorted(np.unique(ids).tolist())
    if packs != [6,8,9,10]:
        raise ValueError('expected battery packs 6/8/9/10')
    rows=[]
    for held in packs:
        fit_ids, cal_ids, val_ids = [], [], []
        for pack in packs:
            if pack == held: continue
            a,b,c = split_train(np.flatnonzero(ids == pack))
            fit_ids.extend(a); cal_ids.extend(b); val_ids.extend(c)
        selected = select_fit(fit_ids, args.max_fit)
        fit = battery_input(signal, selected)
        cal = battery_input(signal, np.asarray(cal_ids))
        held_ids = np.flatnonzero(ids == held)
        for method in METHODS:
            begin=time.perf_counter()
            model=fit_model(method,'peer',fit,cal)
            predicted,_=predict_chunks(model,len(held_ids),lambda sl: battery_input(signal,held_ids[sl]), str(held))
            validation=[]
            # Each known-normal validation pack is a separate alarm sequence.
            for pack in packs:
                if pack == held: continue
                index=np.asarray(val_ids)[ids[np.asarray(val_ids)] == pack]
                flags,_=predict_chunks(model,len(index),lambda sl: battery_input(signal,index[sl]),str(pack))
                validation.append(flags)
            val_count=sum(len(v) for v in validation)
            row=dict(scenario='battery', entity=str(held), method=method,
                     fit_available=len(fit_ids), fit_used=len(selected), calibration_samples=len(cal),
                     validation_samples=val_count, test_samples=len(held_ids), channels=8, window_size=256,
                     threshold=model.threshold_, scale_fallback_features=int(model.scale_fallback_.sum()), validation_alarm_rate=sum(int(v.sum()) for v in validation)/val_count,
                     test_normal_alarm_rate=float(predicted.mean()),
                     test_alarm_intervals=normal_metrics(predicted)['alarm_intervals'],
                     seconds=round(time.perf_counter()-begin,3))
            rows.append(row)
            print(json.dumps(row),flush=True)
            model.save(out/f'battery_pack_{held}_{method}.joblib')
    return rows, {'battery/'+name:file_sha256(cache/name) for name in ('signal.npy','ids.npy','meta.json')}


def smd_benchmark(args, out):
    if args.download:
        download_smd(args.smd_dir,args.machines)
    rows=[]; inputs={}
    for machine in args.machines:
        train,test,labels=load_smd(args.smd_dir,machine)
        fit,cal,val=split_train(train)
        selected=select_fit(np.arange(len(fit)),args.max_fit)
        for method in METHODS:
            begin=time.perf_counter()
            model=fit_model(method,'independent',fit[selected,None,:],cal[:,None,:])
            predicted,scores=predict_chunks(model,len(test),lambda sl: test[sl,None,:],machine)
            val_pred,_=predict_chunks(model,len(val),lambda sl: val[sl,None,:],machine)
            metrics=detection_metrics(labels,predicted)
            row=dict(scenario='smd',entity=machine,method=method,
                     fit_available=len(fit),fit_used=len(selected),calibration_samples=len(cal),
                     validation_samples=len(val),test_samples=len(test),channels=38,window_size=1,
                     threshold=model.threshold_,scale_fallback_features=int(model.scale_fallback_.sum()),validation_alarm_rate=float(val_pred.mean()),
                     **metrics,seconds=round(time.perf_counter()-begin,3))
            rows.append(row)
            print(json.dumps(row),flush=True)
            model.save(out/f'{machine}_{method}.joblib')
            np.savez_compressed(out/f'{machine}_{method}_predictions.npz',scores=scores,
                                confirmed=predicted,labels=labels)
        for group in ('train','test','test_label'):
            name=f'{group}/{machine}.txt'
            inputs['smd/'+name]=file_sha256(args.smd_dir/name)
    return rows,inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=('battery','smd','both'),default='both')
    parser.add_argument('--battery-cache',type=Path,default=ROOT/'datasets/battery/StandTrainData')
    parser.add_argument('--smd-dir',type=Path,default=ROOT/'datasets/smd')
    parser.add_argument('--machines',nargs='+',default=['machine-1-1','machine-2-1','machine-3-1'])
    parser.add_argument('--download',action='store_true',help='Explicitly download SMD from its author repository')
    parser.add_argument('--max-fit',type=int,default=6000,help='Fixed chronological uniform fit sample cap')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.max_fit < 3: parser.error('--max-fit must be >=3')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    source=args.output_dir/'source'; source.mkdir()
    source_files=[*sorted((FRAMEWORK/'src/chronoguard').glob('*.py')),Path(__file__),FRAMEWORK/'pyproject.toml']
    hashes={}
    for file in source_files:
        rel=file.relative_to(ROOT); target=source/rel; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(file,target); hashes[rel.as_posix()]=file_sha256(file)
    manifest=dict(status='running',date=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  protocol=dict(fit_fraction=.6,calibration_fraction=.2,validation_fraction=.2,
                                quantile=.99,persistence=5,seed=42,max_fit=args.max_fit,
                                threshold_from_test_labels=False,point_adjustment=False,
                                smd_training_normality='assumed, not verified by train labels',
                                smd_window_size=1,scale_policy='MAD then fit-std for degenerate varying features',battery='four normal-pack leave-one-pack-out'),
                  versions=dict(python=platform.python_version(),numpy=np.__version__,
                                sklearn=sklearn.__version__,chronoguard=chronoguard.__version__),
                  source_sha256=hashes,command=vars(args)|{'battery_cache':str(args.battery_cache),
                  'smd_dir':str(args.smd_dir),'output_dir':str(args.output_dir)})
    write_json(args.output_dir/'manifest.json',manifest)
    rows=[]; inputs={}; begin=time.perf_counter()
    with threadpool_limits(limits=2):
        if args.scenario in ('battery','both'):
            new,identity=battery_benchmark(args,args.output_dir); rows.extend(new); inputs.update(identity)
        if args.scenario in ('smd','both'):
            new,identity=smd_benchmark(args,args.output_dir); rows.extend(new); inputs.update(identity)
    write_json(args.output_dir/'summary.json',rows)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (args.output_dir/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    manifest.update(status='complete',input_sha256=inputs,seconds=round(time.perf_counter()-begin,3),
                    artifacts={f.relative_to(args.output_dir).as_posix():file_sha256(f)
                               for f in sorted(args.output_dir.rglob('*')) if f.is_file() and f.name!='manifest.json'})
    write_json(args.output_dir/'manifest.json',manifest)
    print('COMPLETE',args.output_dir,flush=True)


if __name__ == '__main__': main()
