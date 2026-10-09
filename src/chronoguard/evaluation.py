# Strict sample-level metrics: never expand one hit to a whole anomaly event.
import numpy as np


def _binary(values):
    x = np.asarray(values)
    if x.ndim != 1 or not np.isin(x, (0, 1)).all():
        raise ValueError('expected a one-dimensional binary array')
    return x.astype(bool)


def alarm_intervals(flags):
    x = _binary(flags)
    changes = np.diff(np.r_[False, x, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def normal_metrics(predicted):
    p = _binary(predicted)
    if not len(p):
        raise ValueError('prediction is empty')
    return {'n_samples': len(p), 'alarm_samples': int(p.sum()),
            'alarm_rate': float(p.mean()), 'alarm_intervals': len(alarm_intervals(p))}


def detection_metrics(labels, predicted):
    y, p = _binary(labels), _binary(predicted)
    if y.shape != p.shape or not len(y):
        raise ValueError('labels and predictions must be nonempty and aligned')
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    events = alarm_intervals(y)
    delays = [int(np.flatnonzero(p[a:b])[0]) for a, b in events if p[a:b].any()]
    false_intervals = sum(not y[a:b].any() for a, b in alarm_intervals(p))
    return dict(n_samples=len(y), anomaly_samples=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1,
                normal_false_positive_rate=fp / (fp + tn) if fp + tn else None,
                events_total=len(events), events_detected=len(delays),
                event_recall=len(delays) / len(events) if events else None,
                detected_event_delays=delays,
                mean_detected_event_delay=float(np.mean(delays)) if delays else None,
                false_alarm_intervals=int(false_intervals))
