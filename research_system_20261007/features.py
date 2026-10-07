"""Raw-cache voltage features; no inferred physical-unit conversion."""
import numpy as np


def extract(signal):
    x = np.asarray(signal)
    if x.ndim != 3 or x.shape[1] < 4 or x.shape[2] != 20 or not len(x):
        raise ValueError('signal must be nonempty (N,T,20), T>=4')
    if not np.isfinite(x).all():
        raise ValueError('signal contains nonfinite values')
    v = np.asarray(x[:, :, :16:2], dtype=np.float64)
    peer = v - np.median(v, axis=2, keepdims=True)
    t = np.linspace(-1., 1., v.shape[1])
    slope = np.einsum('ntc,t->nc', v, t) / np.dot(t, t)
    absolute = np.stack((v.mean(1), v.std(1), np.ptp(v, axis=1), slope,
                         np.diff(v, axis=1).std(1)), axis=2)
    local = np.stack((peer.mean(1), peer.std(1), np.quantile(np.abs(peer), .95, axis=1),
                      np.diff(peer, axis=1).std(1)), axis=2)
    means = v.mean(1)
    return {'absolute': absolute.reshape(len(x), -1), 'peer': local.reshape(len(x), -1),
            'spread': np.max(np.abs(means - np.median(means, axis=1, keepdims=True)), axis=1)}


def subset(features, indices):
    return {key: value[indices] for key, value in features.items()}


def robust_scale(x):
    x = np.asarray(x, dtype=float)
    center = np.median(x, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(x-center), axis=0), 1e-7)
    return center, scale
