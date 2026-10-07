"""Read-only diagnostic of sequence-reset effects, not healthy/fault performance."""
import argparse
import json
from pathlib import Path
import numpy as np
from protocol import confirm, verify_files, write_json, file_hash, seal
from run_benchmark import HOME, write_csv


def longest_run(flags):
    flags=np.asarray(flags,dtype=bool)
    edges=np.flatnonzero(np.diff(np.r_[False,flags,False].astype(int)))
    return int(np.max(edges[1::2]-edges[::2])) if len(edges) else 0


def compare_episode(scores,indices,reference,original_indices,threshold):
    scores,indices,reference,original=(np.asarray(v) for v in (scores,indices,reference,original_indices))
    if scores.ndim!=1 or indices.shape!=scores.shape or np.any(np.diff(indices)<=0):
        raise ValueError('continuous indices must be aligned and strictly increasing')
    if reference.ndim!=1 or original.shape!=reference.shape or not len(reference):
        raise ValueError('episode indices and scores must align')
    at=np.searchsorted(indices,original)
    if np.any(at>=len(indices)) or not np.array_equal(indices[at],original):
        raise ValueError('episode contains windows absent from continuous sequence')
    if not np.isfinite(scores).all() or not np.isfinite(reference).all(): raise ValueError('nonfinite scores')
    continuous=confirm(scores>threshold,np.ones(len(scores)))[at]
    episodic=confirm(reference>threshold,np.ones(len(reference)))
    return dict(max_absolute_score_difference=float(np.max(np.abs(scores[at]-reference))),
        median_absolute_score_difference=float(np.median(np.abs(scores[at]-reference))),
        episode_confirmed_windows=int(episodic.sum()),continuous_confirmed_windows=int(continuous.sum()),
        confirmation_disagreement_windows=int((episodic!=continuous).sum()))


def diagnose(run_dir,out):
    run_dir=Path(run_dir); out=Path(out)
    if out.exists(): raise FileExistsError(out)
    verify_files(run_dir)
    summary=json.loads((run_dir/'summary.json').read_text(encoding='utf-8'))
    rows=[]; comparisons=[]
    for row in summary['raw_folds']:
        fold=run_dir/'raw'/str(row['pack_id']); method=fold/row['method']
        with np.load(method/'held_scores.npz',allow_pickle=False) as arrays:
            scores=arrays['scores']; indices=arrays['indices']
        confirmed=confirm(scores>row['threshold'],np.ones(len(scores)))
        quarters=[float(np.mean(part)) for part in np.array_split(confirmed,4)]
        rows.append(dict(method=row['method'],pack_id=row['pack_id'],n_windows=len(scores),
            longest_confirmation_windows=longest_run(confirmed),confirmation_rate=float(confirmed.mean()),
            first_quarter_confirmation=quarters[0],second_quarter_confirmation=quarters[1],
            third_quarter_confirmation=quarters[2],last_quarter_confirmation=quarters[3]))
        specs=json.loads((fold/'challenge_scenarios.json').read_text(encoding='utf-8'))
        with np.load(method/'challenge_scores.npz',allow_pickle=False) as arrays: reference=arrays['reference']
        seen=set(); length=len(reference)//len(specs)
        for episode,spec in enumerate(specs):
            key=tuple(spec['original_indices'])
            if key in seen: continue
            seen.add(key)
            comparison=compare_episode(scores,indices,reference[episode*length:(episode+1)*length],key,row['threshold'])
            comparisons.append(dict(method=row['method'],pack_id=row['pack_id'],segment=spec['segment'],
                original_first_window=key[0],original_last_window=key[-1],n_windows=length,**comparison))
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'continuous_behavior.csv',rows)
    write_csv(out/'reference_reset_comparison.csv',comparisons)
    result=dict(status='complete',n_method_folds=len(rows),n_unique_reference_comparisons=len(comparisons),
        continuous_behavior=rows,reference_reset_comparison=comparisons,
        conclusion_scope='same saved input windows; change of scoring/confirmation initial state only',
        independent_real_validation=False,reference_health_verified=False,
        source_run=str(run_dir.resolve()),source_manifest_sha256=file_hash(run_dir/'manifest.json'))
    write_json(out/'summary.json',result)
    seal(out,dict(status='complete',purpose='read_only_sequence_reset_diagnostic',
                  source_code_sha256=file_hash(Path(__file__))))
    verify_files(out)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True); parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(HOME): parser.error('output must remain inside isolated research directory')
    result=diagnose(args.run_dir,args.output_dir)
    print('SEQUENCE_DIAGNOSTIC_COMPLETE',result['n_method_folds'],result['n_unique_reference_comparisons'])


if __name__=='__main__': main()
