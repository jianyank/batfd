"""Freeze a research-only global candidate; never claim production acceptance."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
from features import subset
from protocol import PACKS, select_candidate, write_json, seal, verify_files, file_hash
from run_benchmark import HOME, CACHE, extract_all, verify_run, write_csv
from system import CandidateSystem

INFERENCE_SOURCES=('features.py','methods.py','system.py','protocol.py','deploy.py','run_benchmark.py',
                   'inject.py','evaluation.py','predict.py','configs/screen.json')


def global_selection(rows):
    development=[r for r in rows if r['stage']=='development']
    if not development: raise ValueError('development evidence is required')
    result=[]
    for method in sorted({r['method'] for r in development}):
        folds=[r for r in development if r['method']==method]
        if sorted(r['pack_id'] for r in folds)!=list(PACKS):
            raise ValueError('exactly one development result per method and outer fold required')
        metrics={key:float(np.mean([r[key] for r in folds]))
                 for key in ('detection','worst_detection','localization','ref_alarm')}
        kinds=sorted(folds[0]['by_kind'])
        if any(sorted(r['by_kind'])!=kinds for r in folds): raise ValueError('development kinds differ')
        row=select_candidate([dict(method=method,complexity=folds[0]['complexity'],**metrics)])
        row.update(fold_count=4,minimum_fold_detection=min(r['detection'] for r in folds),
                   by_kind={kind:float(np.mean([r['by_kind'][kind] for r in folds])) for kind in kinds},
                   selection_scope='macro average of four overlapping development folds; not blind validation')
        result.append(row)
    return sorted(result,key=lambda r:(-r['utility'],r['complexity']))


def build_bundle(signal,ids,run_dir,out,window_samples=256):
    run_dir=Path(run_dir); out=Path(out)
    if out.exists(): raise FileExistsError(out)
    source_manifest=verify_files(run_dir)
    source_summary=json.loads((run_dir/'summary.json').read_text(encoding='utf-8'))
    ranking=global_selection(source_summary['stress_summary'])
    ids=np.asarray(ids)
    if ids.ndim!=1 or set(np.unique(ids))!=set(PACKS) or len(signal)!=len(ids):
        raise ValueError('expected aligned four-pack input')
    if signal.shape[1:]!=(window_samples,20): raise ValueError('window input contract differs')
    source_inputs=source_manifest.get('source_inputs',{})
    array_sources=[]
    if source_inputs:
        for name,values in (('signal.npy',signal),('ids.npy',ids)):
            paths=[(Path(path),digest) for path,digest in source_inputs.items() if Path(path).name==name]
            if len(paths)!=1: raise ValueError('expected one declared source input: '+name)
            path,digest=paths[0]
            if file_hash(path)!=digest: raise ValueError('source input changed: '+name)
            recorded=np.load(path,mmap_mode='r',allow_pickle=False)
            try:
                if recorded.shape!=values.shape or recorded.dtype!=values.dtype:
                    raise ValueError('source array contract differs: '+name)
                chunk=max(1,(8*1024*1024)//max(1,recorded[0:1].nbytes))
                for start in range(0,len(values),chunk):
                    if not np.array_equal(values[start:start+chunk],recorded[start:start+chunk]):
                        raise ValueError('array differs from declared source input: '+name)
            finally: recorded._mmap.close()
            array_sources.append((path,digest))
    fit=[]; calibration=[]; excluded=[]
    for pack in PACKS:
        rows=np.flatnonzero(ids==pack)
        if len(rows)<10: raise ValueError('too few rows per pack')
        a,b=int(.6*len(rows)),int(.8*len(rows))
        fit.extend(rows[:a]); calibration.extend(rows[a:b]); excluded.extend(rows[b:])
    fit,calibration,excluded=(np.asarray(part,dtype=np.int64) for part in (fit,calibration,excluded))
    start=time.perf_counter(); features=extract_all(signal)
    system=CandidateSystem.fit(ranking[0]['method'],subset(features,fit),subset(features,calibration),ids[calibration])
    fit_seconds=time.perf_counter()-start
    for path,digest in array_sources:
        if file_hash(path)!=digest: raise ValueError('source input changed during build: '+path.name)
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'split.npz',fit=fit,calibration=calibration,excluded=excluded,ids=ids)
    system.save(out/'candidate.joblib')
    write_json(out/'development_ranking.json',ranking)
    write_csv(out/'development_ranking.csv',[{k:v for k,v in row.items() if k!='by_kind'} for row in ranking])
    code_hashes={}
    for name in INFERENCE_SOURCES:
        target=out/'source'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(HOME/name,target); code_hashes[name]=file_hash(target)
    # Speed check only: not a performance validation set.
    sample=np.asarray(signal[excluded[:min(512,len(excluded))]])
    times=[]
    for _ in range(3):
        start=time.perf_counter(); system.predict(sample,pack_id=99)
        times.append(time.perf_counter()-start)
    median=float(np.median(times))
    meta=dict(method=system.detector.name,threshold=system.threshold,q=.99,persistence=5,
        n_fit=len(fit),n_calibration=len(calibration),n_excluded=len(excluded),
        input_shape=['N',window_samples,20],channel_units_verified=False,
        fit_seconds=fit_seconds,inference_windows=len(sample),inference_repeats=3,
        median_inference_seconds=median,median_windows_per_second=len(sample)/median,
        model_bytes=(out/'candidate.joblib').stat().st_size,peak_process_memory_measured=False,
        source_run=str(run_dir.resolve()),source_manifest_sha256=file_hash(run_dir/'manifest.json'),
        source_inputs=source_inputs,refit_inputs_verified=bool(array_sources),
        inference_code_sha256=code_hashes,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__),
        selection_scope=ranking[0]['selection_scope'],global_refit_independently_validated=False,
        deployment_approved=False,real_fault_performance_verified=False,
        real_fault_probability_available=False,mode='offline_research_candidate_only')
    write_json(out/'bundle.json',meta)
    seal(out,dict(status='complete',purpose='frozen_global_research_candidate',deployment_approved=False))
    verify_files(out)
    return meta


def predict_bundle(bundle_dir,input_file,out,pack_id):
    bundle_dir=Path(bundle_dir); out=Path(out)
    if out.exists(): raise FileExistsError(out)
    manifest=verify_files(bundle_dir)
    meta=json.loads((bundle_dir/'bundle.json').read_text(encoding='utf-8'))
    for name,digest in meta['inference_code_sha256'].items():
        if file_hash(HOME/name)!=digest: raise ValueError('inference source differs from frozen bundle: '+name)
    digest=file_hash(input_file)
    signal=np.load(input_file,allow_pickle=False)
    if signal.ndim!=3 or list(signal.shape[1:])!=meta['input_shape'][1:]:
        raise ValueError('prediction input contract differs from frozen bundle')
    model_digest=manifest['files']['candidate.joblib']
    system=CandidateSystem.load(bundle_dir/'candidate.joblib',expected_sha256=model_digest)
    result=system.predict(signal,pack_id=pack_id)
    if file_hash(input_file)!=digest: raise ValueError('input changed during prediction')
    out.mkdir(parents=True,exist_ok=False)
    keys=('scores','cell_scores','ranked_cells','exceeded','confirmed','alarm_started','alarm_cleared','ready')
    np.savez_compressed(out/'prediction.npz',**{key:result[key] for key in keys})
    summary=dict(method=system.detector.name,pack_id=int(pack_id),n_windows=len(signal),
        threshold=system.threshold,n_confirmed=int(result['confirmed'].sum()),
        n_alarm_starts=int(result['alarm_started'].sum()),n_not_ready=int((~result['ready']).sum()),
        quality=result['quality'],localization_basis=result['localization_basis'],
        input_file=str(Path(input_file).resolve()),input_sha256=digest,
        model_sha256=model_digest,bundle_manifest_sha256=file_hash(bundle_dir/'manifest.json'),
        real_fault_performance_verified=False,real_fault_probability_available=False,deployment_approved=False)
    write_json(out/'summary.json',summary)
    seal(out,dict(status='complete',purpose='research_prediction',deployment_approved=False))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('build'); build.add_argument('--run-dir',type=Path,required=True)
    build.add_argument('--output-dir',type=Path,required=True)
    build.add_argument('--cache-dir',type=Path,default=CACHE,help='normal cache matching the sealed source run')
    predict=sub.add_parser('predict'); predict.add_argument('--bundle-dir',type=Path,required=True)
    predict.add_argument('--input',type=Path,required=True); predict.add_argument('--pack-id',type=int,required=True)
    predict.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(HOME): parser.error('output must remain in isolated research directory')
    with threadpool_limits(limits=2):
        if args.command=='build':
            verify_run(args.run_dir)
            signal=np.load(args.cache_dir/'signal.npy',mmap_mode='r',allow_pickle=False)
            ids=np.load(args.cache_dir/'ids.npy',allow_pickle=False)
            result=build_bundle(signal,ids,args.run_dir,args.output_dir)
        else: result=predict_bundle(args.bundle_dir,args.input,args.output_dir,args.pack_id)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
