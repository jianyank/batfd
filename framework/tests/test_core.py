import tempfile
import unittest
from pathlib import Path

import numpy as np
from chronoguard import AnomalyDetector, WindowFeatures, battery_windows


class FeatureTests(unittest.TestCase):
    def test_shapes_for_variable_channels_and_single_point(self):
        for c in (1, 3, 8, 38):
            for t in (1, 4, 16):
                f = WindowFeatures().transform(np.ones((6, t, c)))
                self.assertEqual(f.shape, (6, c, 5))
                self.assertTrue(np.isfinite(f).all())

    def test_peer_matches_existing_battery_features(self):
        import importlib.util
        file = Path(__file__).resolve().parents[2] / 'research_system_20261007/features.py'
        if not file.is_file():
            # The legacy implementation ships only alongside the source repository.
            self.skipTest('legacy features.py not present; equivalence check skipped')
        spec = importlib.util.spec_from_file_location('legacy_features', file)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        x = np.random.default_rng(2).normal(size=(30, 16, 20))
        expected = legacy.extract(x)['peer']
        actual = WindowFeatures('peer').transform(battery_windows(x)).reshape(30, -1)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_battery_mapping_is_explicit_and_unique(self):
        x = np.arange(48).reshape(2, 4, 6)
        np.testing.assert_array_equal(battery_windows(x, [0, 3, 5]), x[:, :, [0, 3, 5]])
        for columns in ([0,0], [0,6], [-1,2], [1], [0.,2.]):
            with self.assertRaises(ValueError): battery_windows(x, columns)
        with self.assertRaises(ValueError): battery_windows(x)

    def test_independent_does_not_mix_channel_units(self):
        x = np.random.default_rng(4).normal(size=(8, 10, 3))
        y = x.copy(); y[:, :, 2] *= 1000
        np.testing.assert_array_equal(WindowFeatures().transform(x)[:, :2],
                                      WindowFeatures().transform(y)[:, :2])

    def test_invalid_inputs(self):
        for x in ([], np.ones((3, 4)), np.ones((0, 4, 2)), np.ones((3, 0, 2)),
                  np.ones((3, 4, 0)), np.full((3, 4, 2), np.nan),
                  np.full((3, 4, 2), np.inf)):
            with self.subTest(shape=np.shape(x)), self.assertRaises(ValueError):
                WindowFeatures().transform(x)
        with self.assertRaises(ValueError):
            WindowFeatures('peer').transform(np.ones((3, 4, 1)))
        with self.assertRaises(ValueError): WindowFeatures('nonsense')


class DetectorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        self.fit = rng.normal(size=(100, 8, 3))
        self.cal = rng.normal(size=(70, 8, 3))
        self.test = rng.normal(size=(80, 8, 3))
        self.test[20:40, :, 1] += 10

    def model(self, method='robust'):
        return AnomalyDetector(method, persistence=3).fit(self.fit).calibrate(self.cal)

    def test_all_methods_finite_and_variable_channels(self):
        for method in ('robust', 'lof', 'iforest', 'pca'):
            for channels in (1, 3, 38):
                with self.subTest(method=method, channels=channels):
                    rng = np.random.default_rng(1)
                    model = AnomalyDetector(method).fit(rng.normal(size=(50, 4, channels)))
                    model.calibrate(rng.normal(size=(40, 4, channels)))
                    out = model.predict(rng.normal(size=(12, 4, channels)), device_id='a')
                    self.assertEqual(out.scores.shape, (12,))
                    self.assertEqual(out.channel_evidence.shape, (12, channels))
                    self.assertTrue(np.isfinite(out.scores).all())

    def test_calibration_is_separate_and_training_scale_is_frozen(self):
        model = AnomalyDetector().fit(self.fit)
        scores = model.score_samples(self.cal)
        center = model.center_.copy(); scale = model.scale_.copy()
        model.calibrate(self.cal)
        self.assertAlmostEqual(model.threshold_, float(np.quantile(scores, .99)))
        np.testing.assert_array_equal(model.center_, center)
        np.testing.assert_array_equal(model.scale_, scale)

    def test_unfitted_and_uncalibrated_rejected(self):
        with self.assertRaises(RuntimeError): AnomalyDetector().predict(self.test)
        with self.assertRaises(RuntimeError): AnomalyDetector().calibrate(self.cal)
        with self.assertRaises(RuntimeError): AnomalyDetector().fit(self.fit).predict(self.test)

    def test_streaming_equals_whole_batch_for_all_methods(self):
        for method in ('robust', 'lof', 'iforest', 'pca'):
            model = self.model(method)
            full = model.predict(self.test, device_id='a')
            state = None; parts = []
            for chunk in np.array_split(self.test, [1, 21, 23, 39, 55]):
                out = model.predict(chunk, device_id='a', state=state)
                parts.append(out); state = out.state
            for attr in ('scores', 'exceeded', 'confirmed', 'alarm_started', 'alarm_cleared',
                         'channel_evidence', 'ranked_channels'):
                actual = np.concatenate([getattr(p, attr) for p in parts])
                np.testing.assert_allclose(actual, getattr(full, attr), atol=1e-12)
            self.assertEqual(state, full.state)

    def test_device_and_model_states_are_not_interchangeable(self):
        model = self.model()
        state = model.predict(self.test[:10], device_id='a').state
        with self.assertRaises(ValueError): model.predict(self.test[10:], device_id='b', state=state)
        with self.assertRaises(ValueError): self.model().predict(self.test[10:], device_id='a', state=state)
        model.calibrate(self.cal)
        with self.assertRaises(ValueError): model.predict(self.test, device_id='a', state=state)

    def test_refit_invalidates_calibration_and_old_state(self):
        model = self.model()
        state = model.predict(self.test[:2]).state
        model.fit(self.fit)
        with self.assertRaises(RuntimeError): model.predict(self.test)
        model.calibrate(self.cal)
        with self.assertRaises(ValueError): model.predict(self.test, state=state)

    def test_persistence_and_alarm_edges(self):
        model = AnomalyDetector(persistence=3).fit(np.zeros((30, 4, 2)))
        model.calibrate(np.zeros((20, 4, 2)))
        x = np.zeros((8, 4, 2)); x[1:5, :, 1] = 10
        out = model.predict(x)
        np.testing.assert_array_equal(out.confirmed, [0,0,0,1,1,0,0,0])
        np.testing.assert_array_equal(out.alarm_started, [0,0,0,1,0,0,0,0])
        np.testing.assert_array_equal(out.alarm_cleared, [0,0,0,0,0,1,0,0])
        self.assertEqual(out.ranked_channels[3, 0], 1)

    def test_reset_explicitly_starts_new_run(self):
        model = AnomalyDetector(persistence=3).fit(np.zeros((30, 4, 2))).calibrate(np.zeros((20, 4, 2)))
        first = model.predict(np.ones((2, 4, 2)))
        self.assertFalse(first.confirmed.any())
        self.assertTrue(model.predict(np.ones((1, 4, 2)), state=first.state).confirmed[0])
        self.assertFalse(model.predict(np.ones((1, 4, 2)), state=None).confirmed[0])

    def test_scores_and_alarm_prefix_do_not_depend_on_future_windows(self):
        for method in ('robust','lof','iforest','pca'):
            model = self.model(method)
            prefix = model.predict(self.test[:30], device_id='a')
            altered = self.test.copy(); altered[30:] += 100
            out = model.predict(altered, device_id='a')
            np.testing.assert_allclose(prefix.scores, out.scores[:30], atol=1e-12)
            np.testing.assert_array_equal(prefix.confirmed, out.confirmed[:30])

    def test_peer_mode_detects_shift_and_exposes_channel_evidence(self):
        model = AnomalyDetector('lof',feature_mode='peer',persistence=3).fit(self.fit).calibrate(self.cal)
        out = model.predict(self.test)
        self.assertTrue(out.confirmed[25:40].any())
        self.assertEqual(int(out.ranked_channels[30,0]),1)

    def test_failed_refit_and_calibration_do_not_destroy_a_working_model(self):
        model = self.model()
        baseline = model.predict(self.test)
        with self.assertRaises(ValueError): model.fit(self.fit[:1])
        with self.assertRaises(ValueError): model.calibrate(self.cal[:1])
        out = model.predict(self.test)
        np.testing.assert_array_equal(out.scores, baseline.scores)
        self.assertEqual(out.state['model_token'], baseline.state['model_token'])

    def test_invalid_streaming_state_and_device_are_rejected(self):
        model = self.model()
        state = model.predict(self.test[:10]).state
        for value in (-1,1.5,True):
            with self.assertRaises(ValueError): model.predict(self.test,state=dict(state,run=value))
        with self.assertRaises(ValueError): model.predict(self.test,device_id=[])

    def test_mismatched_channels_and_window_length_rejected(self):
        model = self.model()
        for shape in ((8, 8, 4), (8, 9, 3)):
            with self.assertRaises(ValueError): model.predict(np.zeros(shape))

    def test_constructor_validation(self):
        for kwargs in ({'method':'unknown'}, {'quantile':0}, {'quantile':1},
                       {'quantile':np.nan}, {'persistence':0}, {'persistence':1.5},
                       {'persistence':True}, {'feature_mode':'unknown'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): AnomalyDetector(**kwargs)

    def test_too_few_fit_or_calibration_rows_rejected(self):
        with self.assertRaises(ValueError): AnomalyDetector('lof').fit(self.fit[:2])
        with self.assertRaises(ValueError): self.model().calibrate(self.cal[:1])

    def test_save_load_and_digest(self):
        model = self.model('lof')
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/'model.joblib'
            digest = model.save(file)
            with self.assertRaises(FileExistsError): model.save(file)
            with self.assertRaises(ValueError): AnomalyDetector.load(file, expected_sha256='0'*64)
            restored = AnomalyDetector.load(file, expected_sha256=digest)
            np.testing.assert_array_equal(restored.predict(self.test).scores, model.predict(self.test).scores)
            state = model.predict(self.test[:20]).state
            restored.predict(self.test[20:], state=state)

    def test_sparse_varying_features_do_not_use_constant_floor(self):
        x = np.zeros((100, 1, 2)); x[60:, 0, 0] = 1
        model = AnomalyDetector().fit(x)
        self.assertGreater(model.scale_[0], .1)
        self.assertLess(float(model.score_samples(x).max()), 3.)
        scaled = x.copy(); scaled[:, :, 0] *= 1000
        other = AnomalyDetector().fit(scaled)
        np.testing.assert_allclose(model.score_samples(x), other.score_samples(scaled))
        self.assertEqual(int(model.scale_fallback_.sum()), 1)

    def test_pca_threshold_ties_are_chunk_invariant(self):
        rng = np.random.default_rng(1)
        fit = rng.normal(size=(60, 4, 3))
        window = rng.normal(size=(1, 4, 3))
        factors = np.r_[np.linspace(.8, .99, 38), 1., 1.]
        cal = window * factors[:, None, None]
        model = AnomalyDetector('pca').fit(fit).calibrate(cal)
        test = np.repeat(window, 10, axis=0)
        whole = model.predict(test)
        chunks, state = [], None
        for i in range(len(test)):
            out = model.predict(test[i:i + 1], state=state)
            chunks.append(out)
            state = out.state
        for field in ('scores', 'channel_evidence', 'ranked_channels',
                      'exceeded', 'confirmed', 'alarm_started', 'alarm_cleared'):
            np.testing.assert_array_equal(np.concatenate([getattr(c, field) for c in chunks]),
                                          getattr(whole, field), err_msg=field)
        features = model.features.transform(test).reshape(len(test), -1)
        z = (features - model.center_) / model.scale_
        reference = z - model.model_.inverse_transform(model.model_.transform(z))
        np.testing.assert_allclose(whole.scores, np.mean(reference ** 2, axis=1),
                                   rtol=1e-12, atol=1e-12)

    def test_constant_fit_is_finite_for_all_methods(self):
        for method in ('robust','lof','iforest','pca'):
            with self.subTest(method=method):
                model = AnomalyDetector(method).fit(np.zeros((30,4,2))).calibrate(np.zeros((20,4,2)))
                self.assertTrue(np.isfinite(model.predict(np.ones((10,4,2))).scores).all())


if __name__ == '__main__':
    unittest.main()
