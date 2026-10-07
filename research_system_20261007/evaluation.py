"""Paired perturbation metrics, explicitly not real-fault performance."""
import numpy as np
from protocol import confirm


def episode_metrics(reference,injected,cells,threshold,spec):
    ids=np.ones(len(reference)); start=spec['onset']
    ref=confirm(np.asarray(reference)>threshold,ids)
    inj=confirm(np.asarray(injected)>threshold,ids)
    new=np.flatnonzero(inj[start:] & ~ref[start:])+start
    first=int(new[0]) if len(new) else None
    # Evaluate localization at the strongest post-onset injected cell evidence,
    # even without an alarm; conditional alarm-localization is reported separately.
    at=first if first is not None else int(start+np.argmax(cells[start:].max(1)))
    ranks=np.argsort(-cells[at]); targets=set(spec['cells'])
    return dict(kind=spec['kind'],strength=spec['strength'],seed=spec['seed'],cells=spec['cells'],
        onset=start,control=spec.get('control',False),new_detection=first is not None,
        injected_alarm=bool(inj[start:].any()),reference_alarm=bool(ref[start:].any()),
        delay_windows=first-start if first is not None else None,
        top1=bool(int(ranks[0]) in targets),top3=bool(set(ranks[:3]) & targets),
        all_targets_top3=bool(targets <= set(ranks[:3])),
        reference_score_median=float(np.median(reference[start:])),
        score_delta_median=float(np.median(injected[start:]-reference[start:])))


def aggregate_stress(rows):
    targeted=[r for r in rows if not r['control']]
    if not targeted: raise ValueError('no targeted scenarios')
    kinds=sorted({r['kind'] for r in targeted})
    per_kind={k:float(np.mean([r['new_detection'] for r in targeted if r['kind']==k])) for k in kinds}
    delays=[r['delay_windows'] for r in targeted if r['delay_windows'] is not None]
    detected=[r for r in targeted if r['new_detection']]
    quiet=[r for r in targeted if not r['reference_alarm']]
    controls=[r for r in rows if r['control']]
    return dict(n_events=len(targeted),detection=float(np.mean([r['new_detection'] for r in targeted])),
        worst_detection=min(per_kind.values()),by_kind=per_kind,
        localization=float(np.mean([r['top1'] for r in targeted])),
        localization_top3=float(np.mean([r['top3'] for r in targeted])),
        conditional_top1=float(np.mean([r['top1'] for r in detected])) if detected else None,
        ref_alarm=float(np.mean([r['reference_alarm'] for r in targeted])),
        n_reference_quiet=len(quiet),quiet_detection=float(np.mean([r['injected_alarm'] for r in quiet])) if quiet else None,
        median_delay_windows=float(np.median(delays)) if delays else None,
        common_mode_new_confirmation=float(np.mean([r['new_detection'] for r in controls])) if controls else None,
        n_controls=len(controls))
