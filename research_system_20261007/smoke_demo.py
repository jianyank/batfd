"""Fixed paired input demo and streaming replay; not a performance estimate."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from deploy import predict_bundle
from inject import inject
from protocol import file_hash, write_json, verify_files, seal
from run_benchmark import HOME, CACHE
from system import CandidateSystem


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,default=CACHE,help='normal cache used to fit this bundle')
    args=parser.parse_args(); out=args.output_dir
    if not out.resolve().is_relative_to(HOME): parser.error('demo must remain in isolated research directory')
    if out.exists(): raise FileExistsError(out)
    verify_files(args.bundle_dir)
    digest=file_hash(args.cache_dir/'signal.npy')
    ids=np.load(args.cache_dir/'ids.npy',allow_pickle=False)
    signal=np.load(args.cache_dir/'signal.npy',mmap_mode='r',allow_pickle=False)
    rows=np.flatnonzero(ids==10)[-96:]
    reference=np.asarray(signal[rows])
    fit=np.load(args.bundle_dir/'split.npz',allow_pickle=False)
    from inject import fit_amplitude
    fit_rows=fit['fit']; amplitude=fit_amplitude(signal[fit_rows[np.linspace(0,len(fit_rows)-1,min(512,len(fit_rows)),dtype=int)]])
    spec=dict(kind='offset',cells=[2],onset=32,strength=5.,seed=314159,control=False)
    altered=inject(reference,spec,amplitude)
    out.mkdir(parents=True,exist_ok=False)
    np.save(out/'reference.npy',reference); np.save(out/'injected.npy',altered)
    write_json(out/'spec.json',dict(spec,amplitude_raw=amplitude,original_indices=rows.tolist(),
                                  source_sha256=digest,selection='pack10 final96 fixed before viewing demo scores'))
    results=[]
    with threadpool_limits(limits=2):
        for name in ('reference','injected'):
            results.append(predict_bundle(args.bundle_dir,out/(name+'.npy'),out/(name+'_prediction'),10))
        model=CandidateSystem.load(args.bundle_dir/'candidate.joblib')
        full=model.predict(altered,10); first=model.predict(altered[:47],10)
        second=model.predict(altered[47:],10,state=first['state'])
        for key in ('scores','cell_scores','ranked_cells','exceeded','confirmed','alarm_started','alarm_cleared','ready'):
            joined=np.concatenate((first[key],second[key]))
            if np.issubdtype(full[key].dtype,np.floating): np.testing.assert_allclose(full[key],joined)
            else: np.testing.assert_array_equal(full[key],joined)
    if file_hash(args.cache_dir/'signal.npy')!=digest: raise ValueError('source cache changed')
    result=dict(status='passed',streaming_replay_verified=True,original_cache_unchanged=True,
                performance_estimate=False,real_fault_performance_verified=False,predictions=results)
    write_json(out/'demo.json',result)
    seal(out,dict(status='complete',purpose='fixed_paired_smoke_demo'))
    verify_files(out); print(result)


if __name__=='__main__': main()
