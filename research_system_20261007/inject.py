"""Paired input-copy sensor perturbations; not electrochemical fault models."""
import numpy as np


def fit_amplitude(signal):
    v=np.asarray(signal)[:, :, :16:2]
    peer=v-np.median(v,axis=2,keepdims=True)
    scale=float(np.median(np.std(peer,axis=1)))
    return max(scale,1e-5)


def scenarios(stage, pack):
    if stage not in ('development','challenge'): raise ValueError(stage)
    strengths=(2.,5.) if stage=='development' else (3.,6.)
    kinds=('offset','drift','noise') if stage=='development' else (
        'offset','drift','noise','stuck','intermittent','two_cell','common_shift','common_noise')
    result=[]
    for i,kind in enumerate(kinds):
        for j,strength in enumerate(strengths):
            for segment in range(2):
                cell=(i+j+segment+pack)%8
                result.append(dict(kind=kind,strength=strength,cells=[cell,(cell+3)%8] if kind=='two_cell' else [cell],
                    onset=32,seed=(100000 if stage=='development' else 200000)+pack*1000+i*100+j*10+segment,
                    segment=segment,stage=stage,control=kind.startswith('common_')))
    return result


def inject(signal,spec,amplitude):
    x=np.array(signal,copy=True)
    onset=spec['onset']; cells=spec['cells']; kind=spec['kind']
    if onset<1 or onset>=len(x) or any(c not in range(8) for c in cells) or amplitude<=0:
        raise ValueError('invalid injection specification')
    rng=np.random.default_rng(spec['seed']); a=spec['strength']*amplitude
    count=len(x)-onset; t=x.shape[1]; cols=[2*c for c in cells]
    direction=1. if spec['seed']%2 else -1.
    for col in cols:
        if kind in ('offset','two_cell'): x[onset:,:,col]+=direction*a
        elif kind=='drift': x[onset:,:,col]+=direction*a*np.linspace(0.,1.,count)[:,None]
        elif kind=='noise': x[onset:,:,col]+=rng.normal(0,a,(count,t))
        elif kind=='stuck': x[onset:,:,col]=x[onset-1,-1,col]
        elif kind=='intermittent':
            for start in range(onset,len(x),16): x[start:start+8,:,col]+=direction*a
        elif kind not in ('common_shift','common_noise'): raise ValueError(kind)
    if kind=='common_shift': x[onset:,:, :16:2]+=direction*a
    if kind=='common_noise': x[onset:,:, :16:2]+=rng.normal(0,a,(count,t,1))
    return x
