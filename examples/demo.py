# Synthetic interface demonstration, NOT a real-fault effectiveness benchmark.
import json
import numpy as np
from chronoguard import AnomalyDetector
from chronoguard.evaluation import detection_metrics, alarm_intervals


def main():
    rng = np.random.default_rng(42)
    fit = rng.normal(size=(240, 16, 6))
    cal = rng.normal(size=(120, 16, 6))
    test = rng.normal(size=(160, 16, 6))
    test[50:90, :, 2] += 8
    labels = np.zeros(160, dtype=bool); labels[50:90] = True
    model = AnomalyDetector('lof').fit(fit).calibrate(cal)
    out = model.predict(test, device_id='demo-device')
    first = model.predict(test[:60], device_id='demo-device')
    second = model.predict(test[60:], device_id='demo-device', state=first.state)
    np.testing.assert_array_equal(np.r_[first.confirmed, second.confirmed], out.confirmed)
    print(json.dumps({'scenario': 'synthetic shift; not field fault data',
                      'method': model.method, 'threshold': out.threshold,
                      'streaming_matches_batch': True,
                      'alarm_intervals': alarm_intervals(out.confirmed),
                      'channel_index_base': 0,
                      'top_channel_during_shift': int(out.ranked_channels[70, 0]),
                      'metrics': detection_metrics(labels, out.confirmed)}, indent=2))


if __name__ == '__main__': main()
