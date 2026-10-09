# Normal-reference fitting, independent calibration and explicit streaming state.
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import uuid

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .features import WindowFeatures, validate_windows

METHODS = ('robust', 'lof', 'iforest', 'pca')


@dataclass(frozen=True)
class PredictionOutput:
    '''Aligned per-window outputs; ranked_channels uses zero-based input-channel indices.'''
    scores: np.ndarray
    channel_evidence: np.ndarray
    ranked_channels: np.ndarray
    exceeded: np.ndarray
    confirmed: np.ndarray
    alarm_started: np.ndarray
    alarm_cleared: np.ndarray
    threshold: float
    state: dict


class AnomalyDetector:
    '''Fit normal-reference windows, calibrate separately, then score unseen windows.

    A fitted model fixes both window length and channel order. Each predict call
    contains one contiguous device sequence. Pass its returned state to the next
    chunk; pass state=None at a gap or device boundary. Channel evidence is not
    causal attribution for LOF or Isolation Forest.
    '''
    def __init__(self, method='robust', feature_mode='independent', quantile=.99,
                 persistence=5, random_state=42):
        if method not in METHODS:
            raise ValueError(f'unknown method: {method}')
        self.features = WindowFeatures(feature_mode)
        if not np.isfinite(quantile) or not 0 < quantile < 1:
            raise ValueError('quantile must be finite and between 0 and 1')
        if isinstance(persistence, bool) or not isinstance(persistence, (int, np.integer)) or persistence < 1:
            raise ValueError('persistence must be a positive integer')
        self.method = method
        self.quantile = float(quantile)
        self.persistence = int(persistence)
        self.random_state = random_state

    def fit(self, windows):
        '''Fit (N,T,C) windows without labels; discard earlier calibration/state.'''
        x = validate_windows(windows)
        f = self.features.transform(x)
        if len(f) < 3:
            raise ValueError('at least three fit windows are required')
        flat = f.reshape(len(f), -1)
        center = np.median(flat, axis=0)
        mad = 1.4826 * np.median(np.abs(flat - center), axis=0)
        std = flat.std(axis=0)
        # Sparse varying features can have zero MAD without being constant.
        fallback = (mad < 1e-7) & (std >= 1e-7)
        scale = np.maximum(np.where(fallback, std, mad), 1e-7)
        z = (flat - center) / scale
        if not np.isfinite(z).all():
            raise ValueError('nonfinite standardized fit features')
        if self.method == 'lof':
            model = LocalOutlierFactor(n_neighbors=min(20, len(z) - 1), novelty=True, n_jobs=1).fit(z)
        elif self.method == 'iforest':
            model = IsolationForest(n_estimators=100, max_samples=min(256, len(z)),
                                    random_state=self.random_state, n_jobs=1).fit(z)
        elif self.method == 'pca':
            # Constant fit data has no identifiable PCA subspace: use full residual.
            model = PCA(n_components=.95, svd_solver='full').fit(z) if np.any(np.var(z, axis=0) > 0) else None
        else:
            model = None
        self.model_ = model
        self.center_, self.scale_ = center, scale
        self.scale_fallback_ = fallback
        self.input_shape_ = x.shape[1:]
        self.n_channels_, self.n_features_ = f.shape[1:]
        self.threshold_ = None
        self._model_token = uuid.uuid4().hex
        return self

    def _require_fit(self):
        if not hasattr(self, 'input_shape_'):
            raise RuntimeError('fit must be called before scoring')

    def _score(self, windows):
        self._require_fit()
        x = validate_windows(windows)
        if x.shape[1:] != self.input_shape_:
            raise ValueError('channel count or window length differs from the fitted model')
        f = self.features.transform(x)
        z = (f.reshape(len(f), -1) - self.center_) / self.scale_
        if not np.isfinite(z).all():
            raise ValueError('nonfinite standardized features')
        evidence = np.max(np.abs(z.reshape(len(z), self.n_channels_, self.n_features_)), axis=2)
        if self.method == 'robust':
            scores = evidence.max(axis=1)
        elif self.method == 'pca':
            if self.model_ is None:
                residual = z
            else:
                # Fixed per-row reductions avoid BLAS batch-size rounding at threshold ties.
                components = self.model_.components_
                projection = np.einsum('nf,kf->nk', z - self.model_.mean_, components, optimize=False)
                reconstructed = np.einsum('nk,kf->nf', projection, components, optimize=False) + self.model_.mean_
                residual = z - reconstructed
            evidence = np.mean(residual.reshape(len(z), self.n_channels_, self.n_features_) ** 2, axis=2)
            scores = evidence.mean(axis=1)
        else:
            scores = -self.model_.score_samples(z)
        if not np.isfinite(scores).all() or not np.isfinite(evidence).all():
            raise ValueError('nonfinite anomaly score')
        return scores, evidence

    def score_samples(self, windows):
        '''Return larger-is-more-anomalous scores; calibration is not required.'''
        return self._score(windows)[0]

    def calibrate(self, windows):
        '''Freeze a score quantile from separate normal-reference windows.'''
        scores = self.score_samples(windows)
        if len(scores) < 2:
            raise ValueError('at least two calibration windows are required')
        self.threshold_ = float(np.quantile(scores, self.quantile))
        self._model_token = uuid.uuid4().hex
        return self

    def predict(self, windows, device_id='default', state=None):
        '''Return scores, channel evidence, causal alarm edges and reusable state.'''
        self._require_fit()
        if self.threshold_ is None:
            raise RuntimeError('calibrate must be called before predict')
        if not isinstance(device_id, (str, int)) or isinstance(device_id, bool):
            raise ValueError('device_id must be a string or integer')
        run, was_confirmed = 0, False
        if state is not None:
            if state.get('model_token') != self._model_token or state.get('device_id') != device_id:
                raise ValueError('state belongs to another device or model/calibration')
            run = state['run']
            was_confirmed = state['confirmed']
            if not isinstance(run, int) or isinstance(run, bool) or run < 0:
                raise ValueError('invalid state run')
            if not isinstance(was_confirmed, bool) or was_confirmed != (run >= self.persistence):
                raise ValueError('invalid state confirmed flag')
        scores, evidence = self._score(windows)
        exceeded = scores > self.threshold_
        confirmed = np.zeros(len(scores), dtype=bool)
        for i, flag in enumerate(exceeded):
            run = run + 1 if flag else 0
            confirmed[i] = run >= self.persistence
        previous = np.r_[was_confirmed, confirmed[:-1]]
        next_state = dict(model_token=self._model_token, device_id=device_id,
                          run=run, confirmed=bool(confirmed[-1]))
        return PredictionOutput(scores=scores, channel_evidence=evidence,
                                ranked_channels=np.argsort(-evidence, axis=1, kind='stable'),
                                exceeded=exceeded, confirmed=confirmed,
                                alarm_started=confirmed & ~previous, alarm_cleared=~confirmed & previous,
                                threshold=self.threshold_, state=next_state)

    def save(self, path):
        '''Save without overwriting an existing file; return its SHA256 digest.'''
        self._require_fit()
        with Path(path).open('xb') as stream:
            joblib.dump(self, stream)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def load(path, expected_sha256=None):
        '''Load trusted joblib bytes, optionally checking a separately trusted digest.'''
        # joblib uses pickle: load only trusted artifacts; a hash is not a trust decision.
        content = Path(path).read_bytes()
        if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError('model differs from expected SHA256')
        model = joblib.load(io.BytesIO(content))
        if not isinstance(model, AnomalyDetector):
            raise ValueError('artifact is not an AnomalyDetector')
        return model
