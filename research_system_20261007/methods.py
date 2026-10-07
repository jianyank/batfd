"""Small, fixed candidate pool; fitting never sees evaluation rows."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from features import robust_scale

RAW_METHODS=('peer_spread','peer_ewma','peer_cusum','peer_robust','peer_multiscale',
             'pca_absolute','pca_peer','mahal_absolute','iforest_absolute','iforest_peer',
             'lof_absolute','lof_peer','svm_peer','peer_anchor','peer_contrast')


def smooth(values,ids,alpha,previous=None):
    out=np.empty_like(values,dtype=float); state=previous
    for i,value in enumerate(values):
        if state is None or (i and ids[i]!=ids[i-1]): state=np.array(value,copy=True)
        else: state=alpha*value+(1-alpha)*state
        out[i]=state
    return out,state


class Detector:
    def __init__(self,name):
        if name not in RAW_METHODS and name not in ('lof_frozen','iforest_frozen','svm_frozen'):
            raise ValueError(name)
        self.name=name

    def fit(self,features):
        self.peer_center,self.peer_scale=robust_scale(features['peer'])
        key='absolute' if self.name.endswith('absolute') else 'peer'
        if self.name.endswith('frozen'): key='frozen'
        self.key=key; self.center,self.scale=robust_scale(features[key])
        z=(features[key]-self.center)/self.scale
        self.spread_center,self.spread_scale=robust_scale(features['spread'])
        self.model=None
        if self.name.startswith('pca_'):
            self.model=PCA(n_components=.95,svd_solver='full').fit(z)
        elif self.name.startswith('mahal_'): self.model=LedoitWolf().fit(z)
        elif self.name.startswith('iforest_'):
            self.model=IsolationForest(n_estimators=100,max_samples=min(256,len(z)),random_state=42,n_jobs=1).fit(z)
        elif self.name.startswith('lof_'):
            self.model=LocalOutlierFactor(n_neighbors=20,novelty=True,n_jobs=1).fit(z)
        elif self.name.startswith('svm_'):
            # Fixed deterministic cap, only fit-domain rows. No test-directed parameter search.
            idx=np.linspace(0,len(z)-1,min(len(z),3000),dtype=int)
            self.model=OneClassSVM(kernel='rbf',nu=.05,gamma=1./z.shape[1]).fit(z[idx])
        return self

    def score(self,features,ids,state=None):
        scores,cells,_=self.score_with_state(features,ids,state)
        return scores,cells

    def score_with_state(self,features,ids,state=None):
        ids=np.asarray(ids); state={} if state is None else dict(state)
        if len(ids) and 'last_id' in state and state['last_id']!=ids[0]: state={}
        peer=(features['peer']-self.peer_center)/self.peer_scale
        cells=np.max(np.abs(peer.reshape(-1,8,4)),axis=2)
        spread=features['spread']
        if self.name=='peer_spread': score=spread
        elif self.name=='peer_ewma':
            score,state['ewma']=smooth(spread,ids,.1,state.get('ewma'))
        elif self.name=='peer_cusum':
            z=(spread-self.spread_center)/self.spread_scale
            score=np.empty(len(z)); run=float(state.get('cusum',0.))
            for i,value in enumerate(z):
                if i and ids[i]!=ids[i-1]: run=0.
                run=max(0.,run+value-.5); score[i]=run
            state['cusum']=run
        elif self.name=='peer_anchor':
            changes=np.zeros_like(peer)
            count=int(state.get('warm_count',0)); total=state.get('warm_sum',np.zeros(peer.shape[1]))
            anchor=state.get('anchor'); previous=state.get('anchored_ewma')
            for i,value in enumerate(peer):
                if i and ids[i]!=ids[i-1]:
                    count=0; total=np.zeros(peer.shape[1]); anchor=None; previous=None
                if count<16:
                    total=total+value; count+=1
                    if count==16: anchor=total/16.
                    continue
                delta=value-anchor
                previous=delta if previous is None else .2*delta+.8*previous
                changes[i]=previous
            state.update(warm_count=count,warm_sum=total,anchor=anchor,anchored_ewma=previous)
            cells=np.max(np.abs(changes).reshape(-1,8,4),axis=2)
            score=cells.max(1)
        elif self.name=='peer_contrast':
            fast,state['fast']=smooth(peer,ids,.2,state.get('fast'))
            slow,state['slow']=smooth(peer,ids,.02,state.get('slow'))
            contrast=np.abs(fast-slow).reshape(-1,8,4)
            # A separate instantaneous variability head retains injected noise evidence.
            noise=np.maximum(peer.reshape(-1,8,4)[:,:,1],peer.reshape(-1,8,4)[:,:,3])
            cells=np.maximum(contrast.max(2),np.maximum(noise,0.))
            score=cells.max(1)
        elif self.name=='peer_robust': score=cells.max(1)
        elif self.name=='peer_multiscale':
            fast,state['fast']=smooth(peer,ids,.2,state.get('fast'))
            slow,state['slow']=smooth(peer,ids,.02,state.get('slow'))
            cells=np.max(np.maximum(np.abs(fast),np.abs(slow)).reshape(-1,8,4),axis=2)
            score=cells.max(1)
        else:
            z=(features[self.key]-self.center)/self.scale
            if self.name.startswith('pca_'):
                residual=z-self.model.inverse_transform(self.model.transform(z))
                score=np.mean(residual**2,axis=1)
                cells=np.mean(residual.reshape(len(z),8,-1)**2,axis=2)
            elif self.name.startswith('mahal_'): score=self.model.mahalanobis(z)
            else: score=-self.model.score_samples(z)
        if not np.isfinite(score).all() or not np.isfinite(cells).all():
            raise ValueError('nonfinite score')
        if len(ids): state['last_id']=ids[-1].item()
        return score,cells,state
