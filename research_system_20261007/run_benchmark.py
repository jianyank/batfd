"""Independent exploratory screening. No independent real-fault labels are used."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
from features import extract, subset
from methods import RAW_METHODS, Detector
from system import CandidateSystem
from protocol import PACKS, split_roles, summarize, select_candidate, file_hash, write_json, seal, verify_files
from inject import fit_amplitude, scenarios, inject
from evaluation import episode_metrics, aggregate_stress

HOME=Path(__file__).resolve().parent
ROOT=HOME.parent
CACHE=ROOT/'outputs/cache/StandTrainData'
FROZEN=ROOT/'outputs/diagnostics/phase5_e2e_lopo_20261002T105204Z'
SOURCE_FILES=('features.py','methods.py','protocol.py','inject.py','system.py','evaluation.py','run_benchmark.py','configs/screen.json','predict.py')


def write_csv(path,rows):
    if not rows: return
    with Path(path).open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def extract_all(signal):
    chunks=[]
    for start in range(0,len(signal),512): chunks.append(extract(signal[start:start+512]))
    return {k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}


def build_episodes(signal,ids,indices,stage,amplitude,length,onset):
    refs=[]; perturbed=[]; specs=[]
    for pack in sorted(set(ids[indices])):
        rows=indices[ids[indices]==pack]
        if len(rows)<length: raise ValueError('episode too long for development/test pack')
        starts=(0,len(rows)-length)
        for spec in scenarios(stage,int(pack)):
            spec=dict(spec,onset=onset)
            idx=rows[starts[spec['segment']]:starts[spec['segment']]+length]
            ref=np.asarray(signal[idx]); altered=inject(ref,spec,amplitude)
            refs.append(extract(ref)); perturbed.append(extract(altered))
            specs.append(dict(spec,pack_id=int(pack),original_indices=idx.tolist(),amplitude_raw=float(amplitude)))
    # Every episode is a separate sequence; repetitions are paired tests, not independent field samples.
    ref={k:np.concatenate([r[k] for r in refs]) for k in refs[0]}
    alt={k:np.concatenate([r[k] for r in perturbed]) for k in perturbed[0]}
    return ref,alt,np.repeat(np.arange(len(specs)),length),specs


def evaluate_stress(model,threshold,episodes,length):
    ref,alt,ids,specs=episodes
    ref_score,_=model.score(ref,ids); alt_score,cells=model.score(alt,ids)
    rows=[dict(episode_metrics(ref_score[i*length:(i+1)*length],alt_score[i*length:(i+1)*length],
               cells[i*length:(i+1)*length],threshold,spec),episode=i,pack_id=spec['pack_id']) for i,spec in enumerate(specs)]
    return rows,dict(reference=ref_score,injected=alt_score,cell_scores=cells),aggregate_stress(rows)


def snapshot_sources(out):
    source=out/'source'; source.mkdir()
    hashes={}
    for name in SOURCE_FILES:
        target=source/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(HOME/name,target); hashes[name]=file_hash(target)
    return hashes


def checked_frozen(ids,input_hashes,cache_dir=CACHE,frozen_dir=FROZEN):
    manifest=json.loads((frozen_dir/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='complete': raise ValueError('frozen source incomplete')
    old=manifest['identity']['inputs']['sha256']
    for name in ('signal.npy','ids.npy','meta.json'):
        if input_hashes[str(cache_dir/name)]!=old[name]: raise ValueError('cache differs from frozen source')
    files={}
    for pack in PACKS:
        fold=frozen_dir/'folds'/str(pack)
        meta=json.loads((fold/'manifest.json').read_text(encoding='utf-8'))
        for name in ('split.npz','calibration_features.npy','held_out_features.npy'):
            file=fold/name; digest=file_hash(file)
            if digest!=meta['files'][name]: raise ValueError('frozen artifact changed: '+str(file))
            files[str(file)]=digest
        with np.load(fold/'split.npz',allow_pickle=False) as split:
            if not np.array_equal(split['calibration_idx'],np.flatnonzero(ids!=pack)) or not np.array_equal(split['held_out_idx'],np.flatnonzero(ids==pack)):
                raise ValueError('frozen split differs from IDs')
    files[str(frozen_dir/'manifest.json')]=file_hash(frozen_dir/'manifest.json')
    return files


def run_frozen(out,raw,ids,frozen_dir=FROZEN):
    rows=[]
    for pack in PACKS:
        fit,cal,dev,held=split_roles(ids,pack)
        fold=frozen_dir/'folds'/str(pack)
        train_idx=np.flatnonzero(ids!=pack)
        a=np.load(fold/'calibration_features.npy',allow_pickle=False)
        b=np.load(fold/'held_out_features.npy',allow_pickle=False)
        if a.shape!=(len(train_idx),24) or b.shape!=(len(held),24) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('frozen feature shape/nonfinite')
        full=np.empty((len(ids),24)); full[train_idx]=a; full[held]=b
        for group,cols in (('all24',list(range(24))),('reconstruction16',[i for i in range(24) if i%3])):
            data=dict(raw,frozen=full[:,cols])
            for algorithm in ('lof','iforest','svm'):
                name=algorithm+'_frozen'; label=group+'_'+algorithm
                print(f'[frozen] held={pack} method={label}',flush=True)
                start=time.perf_counter(); model=Detector(name).fit(subset(data,fit))
                cal_score,_=model.score(subset(data,cal),ids[cal])
                threshold=float(np.quantile(cal_score,.99)); scores,_=model.score(subset(data,held),ids[held])
                folder=out/'frozen'/str(pack)/label; folder.mkdir(parents=True)
                np.savez_compressed(folder/'scores.npz',calibration=cal_score,held=scores,ids=ids[held],indices=held)
                row=dict(pack_id=pack,method=label,group='frozen_replay',**summarize(scores,threshold,ids[held]),
                    elapsed_seconds=time.perf_counter()-start,encoder_calibration_independent=False)
                write_json(folder/'summary.json',row); rows.append(row)
        # Exact legacy scale + corrected LOF semantics, separate from disjoint calibration comparisons.
        center=np.median(a,axis=0); scale=np.maximum(1.4826*np.median(np.abs(a-center),axis=0),np.maximum(.001*np.abs(center),1e-12))
        from sklearn.neighbors import LocalOutlierFactor
        legacy=LocalOutlierFactor(n_neighbors=20,novelty=True).fit((a-center)/scale)
        cal_score=-legacy.negative_outlier_factor_; scores=-legacy.score_samples((b-center)/scale)
        threshold=float(np.quantile(cal_score,.99))
        folder=out/'frozen'/str(pack)/'legacy_all24_lof'; folder.mkdir()
        np.savez_compressed(folder/'scores.npz',calibration=cal_score,held=scores,ids=ids[held],indices=held)
        row=dict(pack_id=pack,method='legacy_all24_lof',group='legacy_in_sample_calibration',
            **summarize(scores,threshold,ids[held]),elapsed_seconds=0.,encoder_calibration_independent=False)
        write_json(folder/'summary.json',row); rows.append(row)
    return rows


def method_table(rows):
    result=[]
    for method in sorted({r['method'] for r in rows}):
        r=[v for v in rows if v['method']==method]; n=sum(v['n_windows'] for v in r)
        result.append(dict(method=method,group=r[0]['group'],n_windows=n,
            macro_exceedance=float(np.mean([v['exceedance_rate'] for v in r])),
            macro_confirmation=float(np.mean([v['confirmation_rate'] for v in r])),
            weighted_exceedance=sum(v['n_exceeded'] for v in r)/n,
            weighted_confirmation=sum(v['n_confirmed'] for v in r)/n,
            packs_with_alarm=sum(v['first_confirmed'] is not None for v in r)))
    return result


def run_raw(signal,ids,out,method_names=RAW_METHODS,episode_length=96,onset=32,input_hashes=None,with_frozen=False,frozen_dir=FROZEN):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    try:
        source_hashes=snapshot_sources(out); features=extract_all(signal)
        folds=[]; stress=[]; selections=[]; selected_stress=[]
        for pack in PACKS:
            fit,cal,dev,held=split_roles(ids,pack)
            folder=out/'raw'/str(pack); folder.mkdir(parents=True)
            np.savez(folder/'split.npz',fit=fit,calibration=cal,development=dev,held=held)
            amplitude=fit_amplitude(signal[fit[np.linspace(0,len(fit)-1,min(512,len(fit)),dtype=int)]])
            development=build_episodes(signal,ids,dev,'development',amplitude,episode_length,onset)
            challenge=build_episodes(signal,ids,held,'challenge',amplitude,episode_length,onset)
            write_json(folder/'development_scenarios.json',development[3]); write_json(folder/'challenge_scenarios.json',challenge[3])
            development_rows=[]; models={}
            # Build/choose before evaluating any held-out challenge performance.
            for complexity,name in enumerate(method_names):
                print(f'[raw fit/dev] held={pack} method={name}',flush=True)
                start=time.perf_counter()
                system=CandidateSystem.fit(name,subset(features,fit),subset(features,cal),ids[cal])
                models[name]=system
                cal_score,_=system.detector.score(subset(features,cal),ids[cal])
                cal_summary=summarize(cal_score,system.threshold,ids[cal])
                cal_summary.update(q99_tie_fraction=float(np.mean(cal_score==system.threshold)))
                method_dir=folder/name; method_dir.mkdir()
                np.save(method_dir/'calibration.npy',cal_score)
                write_json(method_dir/'calibration_summary.json',cal_summary)
                events,arrays,metrics=evaluate_stress(system.detector,system.threshold,development,episode_length)
                np.savez_compressed(method_dir/'development_scores.npz',**arrays)
                write_json(method_dir/'development_events.json',events)
                row=dict(method=name,pack_id=pack,stage='development',complexity=complexity,**metrics)
                development_rows.append(row); stress.append(row)
                write_json(method_dir/'fit.json',dict(elapsed_seconds=time.perf_counter()-start,amplitude_raw=amplitude))
            chosen=select_candidate(development_rows)
            chosen.update(selection_scope='other three packs development injections only',
                          independent_real_validation=False)
            selections.append(chosen); write_json(folder/'selection.json',chosen)
            models[chosen['method']].save(folder/'selected_system.joblib')
            for name in method_names:
                print(f'[raw held/challenge] held={pack} method={name}',flush=True)
                system=models[name]; method_dir=folder/name
                score,cells=system.detector.score(subset(features,held),ids[held])
                np.savez_compressed(method_dir/'held_scores.npz',scores=score,cell_scores=cells,ids=ids[held],indices=held)
                row=dict(pack_id=pack,method=name,group='raw_disjoint_calibration',**summarize(score,system.threshold,ids[held]))
                write_json(method_dir/'held_summary.json',row); folds.append(row)
                events,arrays,metrics=evaluate_stress(system.detector,system.threshold,challenge,episode_length)
                np.savez_compressed(method_dir/'challenge_scores.npz',**arrays)
                write_json(method_dir/'challenge_events.json',events)
                m=dict(method=name,pack_id=pack,stage='challenge',complexity=list(method_names).index(name),**metrics)
                stress.append(m)
                if name==chosen['method']: selected_stress.append(m)
            del models
        frozen_rows=run_frozen(out,features,ids,frozen_dir=frozen_dir) if with_frozen else []
        summary=dict(schema_version=1,status='complete',raw_folds=folds,frozen_folds=frozen_rows,
                     method_summary=method_table(folds+frozen_rows),stress_summary=stress,selections=selections,
                     selected_challenge=selected_stress,real_fault_performance_verified=False,
                     synthetic_events_independent=False,exploratory_previously_observed_packs=True)
        write_json(out/'summary.json',summary); write_csv(out/'method_summary.csv',summary['method_summary'])
        write_csv(out/'raw_folds.csv',folds); write_csv(out/'stress_summary.csv',stress)
        source_inputs=input_hashes or {}
        for filename,digest in source_inputs.items():
            if file_hash(filename)!=digest: raise ValueError('input modified during run: '+filename)
        for filename,digest in source_hashes.items():
            if file_hash(HOME/filename)!=digest: raise ValueError('implementation changed during run')
        manifest=dict(status='complete',schema_version=1,source_inputs=source_inputs,source_code=source_hashes,
            runtime=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__),
            protocol=dict(fit_fraction=.6,calibration_fraction=.2,development_fraction=.2,q=.99,persistence=5,
                          labels='synthetic sensor perturbations only',time_independence_verified=False,
                          deep_features_retrained=False,encoder_calibration_independent=False),
            created_utc=datetime.now(timezone.utc).isoformat())
        seal(out,manifest)
        verify_run(out,check_sources=bool(source_inputs))
        return summary
    except Exception as exc:
        write_json(out/'FAILED.json',dict(status='failed',error=type(exc).__name__+': '+str(exc)))
        raise


def _verify_rows(rows,fields,expected):
    keys=[tuple(row[field] for field in fields) for row in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected):
        raise ValueError('summary coverage mismatch: '+','.join(fields))
    return dict(zip(keys,rows))


def _verify_indices(indices,n):
    indices=np.asarray(indices)
    if (indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer) or
        not len(indices) or np.any(indices<0) or np.any(indices>=n) or len(np.unique(indices))!=len(indices)):
        raise ValueError('invalid role / score / scenario indices')
    return indices


def verify_run(out,check_sources=True):
    out=Path(out); manifest=verify_files(out)
    if check_sources:
        for filename,digest in manifest['source_inputs'].items():
            if file_hash(filename)!=digest: raise ValueError('source input changed: '+filename)
    for name,digest in manifest['source_code'].items():
        if manifest['files'].get('source/'+name)!=digest: raise ValueError('source snapshot mismatch')
    cfg=json.loads((out/'source/configs/screen.json').read_text(encoding='utf-8'))
    contract=manifest['protocol']
    if (manifest['schema_version']!=1 or cfg['schema_version']!=1 or cfg['q']!=contract['q'] or
        cfg['persistence']!=contract['persistence'] or cfg['q']!=.99 or cfg['persistence']!=5 or
        cfg['roles']!=dict(fit=contract['fit_fraction'],calibration=contract['calibration_fraction'],
                           development=contract['development_fraction']) or
        cfg['roles']!=dict(fit=.6,calibration=.2,development=.2)):
        raise ValueError('unsupported or inconsistent sealed protocol')
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    if summary['schema_version']!=1 or summary['status']!='complete': raise ValueError('invalid summary status/schema')
    # Inventory supplies executed candidates; sealed config supplies the versioned pool.
    pairs={group:{(int(parts[1]),parts[2]) for name in manifest['files']
                  if len(parts:=Path(name).parts)==4 and parts[0]==group and parts[3]==marker}
           for group,marker in (('raw','calibration.npy'),('frozen','scores.npz'))}
    methods={p:{m for pack,m in pairs['raw'] if pack==p} for p in PACKS}
    if (set(p for p,m in pairs['raw'])!=set(PACKS) or not methods[PACKS[0]] or
        any(methods[p]!=methods[PACKS[0]] for p in PACKS) or
        not methods[PACKS[0]]<=set(cfg['raw_methods'])):
        raise ValueError('candidate inventory differs from sealed pool/folds')
    if pairs['frozen'] and (set(p for p,m in pairs['frozen'])!=set(PACKS) or
        any({m for p,m in pairs['frozen'] if p==pack}!=
            {m for p,m in pairs['frozen'] if p==PACKS[0]} for pack in PACKS)):
        raise ValueError('frozen fold coverage mismatch')
    _verify_rows(summary['raw_folds'],('pack_id','method'),pairs['raw'])
    _verify_rows(summary['frozen_folds'],('pack_id','method'),pairs['frozen'])
    stress=_verify_rows(summary['stress_summary'],('pack_id','method','stage'),
                       {(p,m,s) for p,m in pairs['raw'] for s in ('development','challenge')})
    _verify_rows(summary['selections'],('pack_id',),{(p,) for p in PACKS})
    splits={}
    for pack in PACKS:
        with np.load(out/'raw'/str(pack)/'split.npz',allow_pickle=False) as data:
            splits[pack]={role:data[role] for role in ('fit','calibration','development','held')}
    held=np.concatenate([splits[p]['held'] for p in PACKS]); n=len(held)
    _verify_indices(held,n)
    if not np.array_equal(np.sort(held),np.arange(n)): raise ValueError('held roles do not cover input')
    ids=np.empty(n,dtype=np.int64)
    for pack in PACKS: ids[splits[pack]['held']]=pack
    id_inputs=[name for name in manifest['source_inputs'] if Path(name).name=='ids.npy']
    if check_sources and id_inputs:
        if len(id_inputs)!=1: raise ValueError('ambiguous source ids')
        source_ids=np.load(id_inputs[0],allow_pickle=False)
        if not np.array_equal(source_ids,ids): raise ValueError('split differs from source pack IDs')
    specs_by_stage={}
    for pack,split in splits.items():
        expected=split_roles(ids,pack)
        for (role,indices),part in zip(split.items(),expected):
            _verify_indices(indices,n)
            if not np.array_equal(indices,part): raise ValueError('role partition mismatch: '+role)
        for stage,role in (('development','development'),('challenge','held')):
            specs=json.loads((out/'raw'/str(pack)/(stage+'_scenarios.json')).read_text(encoding='utf-8'))
            if not specs: raise ValueError('empty scenarios')
            length=len(specs[0]['original_indices'])
            for spec in specs:
                idx=_verify_indices(spec['original_indices'],n)
                if (len(idx)!=length or not np.isin(idx,split[role]).all() or
                    not np.all(ids[idx]==spec['pack_id']) or spec['stage']!=stage or
                    not isinstance(spec['onset'],int) or not 1<=spec['onset']<length or
                    not spec['cells'] or any(not isinstance(c,int) or c not in range(8) for c in spec['cells'])):
                    raise ValueError('scenario role/shape mismatch')
            specs_by_stage[pack,stage]=specs
        if {s['seed'] for s in specs_by_stage[pack,'development']} & {s['seed'] for s in specs_by_stage[pack,'challenge']}:
            raise ValueError('development/challenge seeds overlap')
    thresholds={}
    for row in summary['raw_folds']+summary['frozen_folds']:
        pack=row['pack_id']; method=row['method']; frozen=(pack,method) in pairs['frozen']
        folder=out/('frozen' if frozen else 'raw')/str(pack)/method
        if not frozen: cal=np.load(folder/'calibration.npy',allow_pickle=False)
        with np.load(folder/('scores.npz' if frozen else 'held_scores.npz'),allow_pickle=False) as scores:
            if frozen: cal=scores['calibration']
            count=n-len(splits[pack]['held']) if row['group']=='legacy_in_sample_calibration' else len(splits[pack]['calibration'])
            if cal.shape!=(count,) or not np.isfinite(cal).all(): raise ValueError('calibration shape/nonfinite')
            threshold=float(np.quantile(cal,cfg['q']))
            held_score=scores['held' if frozen else 'scores']
            if (held_score.shape!=(len(splits[pack]['held']),) or
                not np.array_equal(scores['indices'],splits[pack]['held']) or
                not np.array_equal(scores['ids'],ids[splits[pack]['held']])):
                raise ValueError('held score indices/IDs/shape mismatch')
            _verify_indices(scores['indices'],n)
            if not frozen and (scores['cell_scores'].shape!=(len(held_score),8) or not np.isfinite(scores['cell_scores']).all()):
                raise ValueError('held cell score shape/nonfinite')
            actual=summarize(held_score,threshold,scores['ids'],cfg['persistence'])
        saved=json.loads((folder/('summary.json' if frozen else 'held_summary.json')).read_text(encoding='utf-8'))
        if saved!=row or any(actual[k]!=row[k] for k in actual): raise ValueError('held summary mismatch')
        if not frozen: thresholds[pack,method]=threshold
    for row in summary['stress_summary']:
        pack=row['pack_id']; method=row['method']; stage=row['stage']
        folder=out/'raw'/str(pack)/method
        events=json.loads((folder/(stage+'_events.json')).read_text(encoding='utf-8'))
        specs=specs_by_stage[pack,stage]; length=len(specs[0]['original_indices']); total=len(specs)*length
        with np.load(folder/(stage+'_scores.npz'),allow_pickle=False) as arrays:
            if (arrays['reference'].shape!=(total,) or arrays['injected'].shape!=(total,) or
                arrays['cell_scores'].shape!=(total,8) or
                any(not np.isfinite(arrays[k]).all() for k in ('reference','injected','cell_scores'))):
                raise ValueError('episode score shape/nonfinite')
            expected=[dict(episode_metrics(arrays['reference'][i*length:(i+1)*length],
                       arrays['injected'][i*length:(i+1)*length],arrays['cell_scores'][i*length:(i+1)*length],
                       thresholds[pack,method],spec),episode=i,pack_id=spec['pack_id']) for i,spec in enumerate(specs)]
        if expected!=events: raise ValueError('episode evidence inconsistent with saved scores')
        actual=aggregate_stress(events)
        if any(actual[k]!=row[k] for k in actual): raise ValueError('stress summary mismatch')
        if row['complexity']!=stress[pack,method,'development']['complexity']: raise ValueError('candidate complexity mismatch')
    selected=[]
    for row in summary['selections']:
        dev=[r for r in summary['stress_summary'] if r['stage']=='development' and r['pack_id']==row['pack_id']]
        chosen=select_candidate(dev)
        saved=json.loads((out/'raw'/str(row['pack_id'])/'selection.json').read_text(encoding='utf-8'))
        if saved!=row or any(chosen[k]!=row[k] for k in chosen): raise ValueError('selection mismatch')
        selected.append(stress[row['pack_id'],chosen['method'],'challenge'])
    if selected!=summary['selected_challenge']: raise ValueError('selected challenge mismatch')
    if method_table(summary['raw_folds']+summary['frozen_folds'])!=summary['method_summary']:
        raise ValueError('method aggregate mismatch')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--verify',type=Path)
    parser.add_argument('--include-frozen',action='store_true')
    parser.add_argument('--cache-dir',type=Path,default=CACHE,help='existing normal NumPy cache directory')
    parser.add_argument('--frozen-dir',type=Path,default=FROZEN,help='optional historical feature artifacts; not shipped')
    args=parser.parse_args()
    if args.verify:
        summary=verify_run(args.verify)
        print('VERIFIED: artifact hashes, source inputs, q99, persistence, summaries and development-only selection')
        print('raw folds:',len(summary['raw_folds']),'frozen folds:',len(summary['frozen_folds']))
        return
    if not args.output_dir: parser.error('--output-dir is required; existing directories are rejected')
    if not args.output_dir.resolve().is_relative_to(HOME): parser.error('output must be inside isolated research directory')
    signal=np.load(args.cache_dir/'signal.npy',mmap_mode='r',allow_pickle=False)
    ids=np.load(args.cache_dir/'ids.npy',allow_pickle=False)
    meta=json.loads((args.cache_dir/'meta.json').read_text(encoding='utf-8'))
    counts={int(p):int(np.sum(ids==p)) for p in PACKS}
    if counts!={6:9127,8:11147,9:5093,10:1141} or signal.shape!=(26508,256,20) or meta['has_time']:
        raise ValueError('real-data contract differs from frozen plan')
    cfg=json.loads((HOME/'configs/screen.json').read_text(encoding='utf-8'))
    if cfg['q']!=.99 or cfg['persistence']!=5 or tuple(cfg['raw_methods'])!=RAW_METHODS or not cfg['synthetic_allowed']:
        raise ValueError('config does not match implementation protocol')
    hashes={str(args.cache_dir/n):file_hash(args.cache_dir/n) for n in ('signal.npy','ids.npy','meta.json')}
    if args.include_frozen: hashes.update(checked_frozen(ids,hashes,cache_dir=args.cache_dir,frozen_dir=args.frozen_dir))
    with threadpool_limits(limits=2):
        result=run_raw(signal,ids,args.output_dir,input_hashes=hashes,with_frozen=args.include_frozen,frozen_dir=args.frozen_dir)
    print('COMPLETE:',args.output_dir,'selected:',[(s['pack_id'],s['method']) for s in result['selections']])


if __name__=='__main__': main()
