"""Replayable candidate: frozen fit, threshold and pack-local causal state."""
from pathlib import Path
import hashlib
import io
import joblib
import numpy as np
from features import extract
from methods import Detector
from protocol import confirm


class CandidateSystem:
    @classmethod
    def fit(cls,name,fit_features,cal_features,cal_ids,q=.99,persistence=5):
        if name.endswith('_frozen'):
            raise ValueError('raw CandidateSystem cannot infer frozen features; use Detector feature replay')
        if name=='peer_anchor':
            ids=np.asarray(cal_ids)
            boundaries=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1,len(ids)]
            if not np.any(np.diff(boundaries)>16):
                raise ValueError('peer_anchor calibration has no ready rows after 16-window warmup')
        self=cls(); self.detector=Detector(name).fit(fit_features)
        cal_scores,_=self.detector.score(cal_features,cal_ids)
        self.threshold=float(np.quantile(cal_scores,q)); self.q=q; self.persistence=persistence
        return self

    def predict(self,signal,pack_id,state=None):
        if state is not None and state['pack_id']!=pack_id:
            raise ValueError('state belongs to another pack; reset explicitly')
        state={} if state is None else state
        ids=np.full(len(signal),pack_id)
        scores,cells,model_state=self.detector.score_with_state(extract(signal),ids,state.get('detector'))
        ready=np.ones(len(scores),dtype=bool)
        if self.detector.name=='peer_anchor':
            warm_count=state.get('detector',{}).get('warm_count',0)
            ready=np.arange(len(scores))+warm_count>=16
        exceeded=(scores>self.threshold) & ready
        run=int(state.get('run',0)); confirmed=confirm(exceeded,ids,self.persistence,run)
        for flag in exceeded: run=run+1 if flag else 0
        initial_confirmed=bool(state.get('confirmed',False))
        onset=confirmed & ~np.r_[initial_confirmed,confirmed[:-1]]
        cleared=~confirmed & np.r_[initial_confirmed,confirmed[:-1]]
        return dict(scores=scores,cell_scores=cells,ranked_cells=np.argsort(-cells,axis=1)+1,
                    exceeded=exceeded,confirmed=confirmed,alarm_started=onset,alarm_cleared=cleared,
                    ready=ready,reference_health_verified=False,
                    quality='finite_raw_cache_semantics_unverified',threshold=self.threshold,
                    state=dict(pack_id=pack_id,detector=model_state,run=run,confirmed=bool(confirmed[-1])),
                    localization_basis='PCA per-cell reconstruction contribution' if self.detector.name.startswith('pca_')
                    else 'temporal peer-residual change contributions' if self.detector.name in ('peer_anchor','peer_contrast','peer_multiscale')
                    else 'independent robust peer-residual head; not causal attribution of the global detector')

    def save(self,path):
        path=Path(path)
        if path.exists(): raise FileExistsError(path)
        joblib.dump(self,path)

    @staticmethod
    def load(path,expected_sha256=None):
        # Hash verification binds bytes, not trust: never load an untrusted pickle-backed artifact.
        if expected_sha256 is None: return joblib.load(path)
        content=Path(path).read_bytes()
        if hashlib.sha256(content).hexdigest()!=expected_sha256:
            raise ValueError('model differs from expected SHA256')
        return joblib.load(io.BytesIO(content))
