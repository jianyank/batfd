"""Predict with one fold candidate; verify local artifacts before trusted joblib load."""
import argparse
from pathlib import Path
import numpy as np
from protocol import verify_files, file_hash, write_json, seal
from system import CandidateSystem


def predict_file(run_dir,fold,input_file,output_dir,pack_id):
    run_dir=Path(run_dir); manifest=verify_files(run_dir)
    model_path=run_dir/'raw'/str(fold)/'selected_system.joblib'
    digest=file_hash(input_file)
    signal=np.load(input_file,allow_pickle=False)
    if file_hash(input_file)!=digest: raise ValueError('input changed during loading')
    model_digest=manifest['files'][model_path.relative_to(run_dir).as_posix()]
    system=CandidateSystem.load(model_path,expected_sha256=model_digest)
    prediction=system.predict(signal,pack_id=pack_id)
    if file_hash(input_file)!=digest: raise ValueError('input changed during prediction')
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=False)
    keys=('scores','cell_scores','ranked_cells','exceeded','confirmed','alarm_started','alarm_cleared','ready')
    np.savez_compressed(out/'prediction.npz',**{k:prediction[k] for k in keys})
    summary=dict(n_windows=len(signal),pack_id=int(pack_id),source_fold=int(fold),method=system.detector.name,
        threshold=system.threshold,n_confirmed=int(prediction['confirmed'].sum()),
        n_alarm_starts=int(prediction['alarm_started'].sum()),n_not_ready=int((~prediction['ready']).sum()),
        quality=prediction['quality'],
        localization_basis=prediction['localization_basis'],real_fault_probability_available=False,
        real_fault_performance_verified=False,input_file=str(Path(input_file).resolve()),
        input_sha256=digest,model_sha256=model_digest)
    write_json(out/'summary.json',summary); seal(out,dict(status='complete',purpose='offline_candidate_prediction'))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--fold',type=int,choices=(6,8,9,10),required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--pack-id',type=int,required=True)
    args=parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(Path(__file__).resolve().parent):
        parser.error('output must remain in isolated research directory')
    print(predict_file(args.run_dir,args.fold,args.input,args.output_dir,args.pack_id))


if __name__=='__main__': main()
