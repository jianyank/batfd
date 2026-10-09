# Per-channel window statistics; physical units are never inferred.
import numpy as np


def validate_windows(windows):
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) < 1:
        raise ValueError('windows must be a nonempty (N,T,C) array')
    if not np.isfinite(x).all():
        raise ValueError('windows contains nonfinite values')
    return x


def battery_windows(signal, channel_indices=None):
    '''Select comparable cell channels; default mapping is legacy columns 0,2,...,14.'''
    x = validate_windows(signal)
    if channel_indices is None:
        if x.shape[2] != 20:
            raise ValueError('default battery mapping requires 20 columns')
        indices = np.arange(0, 16, 2)
    else:
        indices = np.asarray(channel_indices)
    if indices.ndim != 1 or len(indices) < 2 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('channel_indices must contain at least two integer columns')
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= x.shape[2]):
        raise ValueError('channel_indices must be unique and within the input columns')
    return x[:, :, indices]


class WindowFeatures:
    '''Produce (N,C,F) statistics; peer mode requires physically comparable channels.'''
    def __init__(self, mode='independent'):
        if mode not in ('independent', 'peer'):
            raise ValueError(f'unknown feature mode: {mode}')
        self.mode = mode

    def transform(self, windows):
        x = validate_windows(windows)
        if self.mode == 'peer':
            if x.shape[2] < 2:
                raise ValueError('peer features require at least two comparable channels')
            peer = x - np.median(x, axis=2, keepdims=True)
            delta = np.diff(peer, axis=1).std(axis=1) if x.shape[1] > 1 else np.zeros_like(peer[:, 0])
            return np.stack((peer.mean(axis=1), peer.std(axis=1),
                             np.quantile(np.abs(peer), .95, axis=1), delta), axis=2)
        t = np.linspace(-1., 1., x.shape[1])
        slope = np.einsum('ntc,t->nc', x, t) / np.dot(t, t) if x.shape[1] > 1 else np.zeros_like(x[:, 0])
        delta = np.diff(x, axis=1).std(axis=1) if x.shape[1] > 1 else np.zeros_like(x[:, 0])
        return np.stack((x.mean(axis=1), x.std(axis=1), np.ptp(x, axis=1), slope, delta), axis=2)
