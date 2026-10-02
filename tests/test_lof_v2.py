"""隔离 LOF v2 的训练评分修正，锁定其余旧行为。"""
import unittest

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from batfd.detect.lof import LOFDetector
from batfd.detect.lof_v2 import LOFDetectorV2


class LOFDetectorV2Tests(unittest.TestCase):
    def setUp(self):
        self.calibration = np.array([
            [0., 0.], [.1, .2], [.2, -.1], [.3, .4],
            [1., 1.], [4., 4.], [5., 4.], [9., -2.],
        ])
        self.held_out = np.array([[.15, .15], [8., 9.]])
        center = np.median(self.calibration, axis=0)
        scale = np.maximum(
            np.median(np.abs(self.calibration - center), axis=0) * 1.4826,
            np.maximum(.001 * np.abs(center), 1e-12),
        )
        self.reference = LocalOutlierFactor(n_neighbors=3, novelty=True).fit(
            (self.calibration - center) / scale)
        self.held_scaled = (self.held_out - center) / scale

    def test_versioned_api_inherits_old_detector_and_fit_returns_self(self):
        detector = LOFDetectorV2(n_neighbors=3)
        self.assertIsInstance(detector, LOFDetector)
        original = self.calibration.copy()
        self.assertIs(detector.fit(self.calibration), detector)
        np.testing.assert_array_equal(self.calibration, original)

    def test_training_scores_use_standard_lof_not_training_novelty_queries(self):
        detector = LOFDetectorV2(n_neighbors=3).fit(self.calibration)
        expected = -self.reference.negative_outlier_factor_
        np.testing.assert_array_equal(detector._train_score, expected)
        self.assertGreater(np.max(np.abs(expected - detector.score(
            self.calibration).score)), 1.)
        legacy = LOFDetector(n_neighbors=3).fit(self.calibration)
        np.testing.assert_array_equal(legacy._train_score,
                                      legacy.score(self.calibration).score)

    def test_training_threshold_uses_standard_quantiles(self):
        detector = LOFDetectorV2(n_neighbors=3).fit(self.calibration)
        expected = -self.reference.negative_outlier_factor_
        for q in (0., .5, .9, .99, 1.):
            with self.subTest(q=q):
                self.assertEqual(detector.threshold_from_train(q),
                                 float(np.quantile(expected, q)))
        self.assertIsInstance(detector.threshold_from_train(), float)
        for q in (-.1, 1.1):
            with self.subTest(q=q), self.assertRaises(ValueError):
                detector.threshold_from_train(q)

    def test_new_sample_scores_are_identical_to_old_and_sklearn(self):
        detector = LOFDetectorV2(n_neighbors=3).fit(self.calibration)
        legacy = LOFDetector(n_neighbors=3).fit(self.calibration)
        result = detector.score(self.held_out)
        np.testing.assert_array_equal(result.score, legacy.score(self.held_out).score)
        np.testing.assert_array_equal(result.score,
                                      -self.reference.score_samples(self.held_scaled))
        self.assertIsNone(result.threshold)
        self.assertIsNone(result.flag)
        self.assertEqual(result.meta, {"mode": "train_novelty"})

    def test_score_threshold_is_strict_and_preserves_result_metadata(self):
        detector = LOFDetectorV2(n_neighbors=3).fit(self.calibration)
        threshold = float(-self.reference.score_samples(self.held_scaled)[0])
        result = detector.score(self.held_out, threshold=threshold)
        self.assertEqual(result.threshold, threshold)
        np.testing.assert_array_equal(result.flag, [False, True])
        self.assertEqual(result.meta, {"mode": "train_novelty"})

    def test_robust_scaler_and_both_floors_are_unchanged(self):
        calibration = np.array([
            [100., 0., 0., -1.], [100., 0., 1e-15, 0.],
            [100., 0., 2e-15, 1.], [100., 0., 3e-15, 2.],
        ])
        for norm_floor, constant_scale in ((.001, .1), (.1, 10.)):
            with self.subTest(norm_floor=norm_floor):
                detector = LOFDetectorV2(n_neighbors=2, norm_floor=norm_floor).fit(calibration)
                legacy = LOFDetector(n_neighbors=2, norm_floor=norm_floor).fit(calibration)
                np.testing.assert_allclose(detector.center_, [100., 0., 1.5e-15, .5])
                np.testing.assert_array_equal(detector.scale_,
                                              [constant_scale, 1e-12, 1e-12, 1.4826])
                np.testing.assert_array_equal(detector.center_, legacy.center_)
                np.testing.assert_array_equal(detector.scale_, legacy.scale_)

    def test_disabled_standardization_preserves_raw_novelty_scoring(self):
        detector = LOFDetectorV2(n_neighbors=3, standardize=False).fit(self.calibration)
        reference = LocalOutlierFactor(n_neighbors=3, novelty=True).fit(self.calibration)
        self.assertIsNone(detector.center_)
        self.assertIsNone(detector.scale_)
        self.assertIs(detector.transform(self.held_out), self.held_out)
        np.testing.assert_array_equal(detector._train_score, -reference.negative_outlier_factor_)
        np.testing.assert_array_equal(detector.score(self.held_out).score,
                                      -reference.score_samples(self.held_out))

    def test_held_out_scoring_cannot_change_calibration_or_batch_neighbors(self):
        detector = LOFDetectorV2(n_neighbors=3).fit(self.calibration)
        center, scale = detector.center_.copy(), detector.scale_.copy()
        train_score = detector._train_score.copy()
        threshold = detector.threshold_from_train()
        original = detector.score(self.held_out).score
        with_extreme = detector.score(np.vstack([self.held_out, [1e9, -1e9]])).score
        np.testing.assert_array_equal(with_extreme[:2], original)
        np.testing.assert_array_equal(detector.center_, center)
        np.testing.assert_array_equal(detector.scale_, scale)
        np.testing.assert_array_equal(detector._train_score, train_score)
        self.assertEqual(detector.threshold_from_train(), threshold)
        self.assertEqual(detector.model_.n_samples_fit_, len(self.calibration))

    def test_neighbor_count_is_clipped_for_small_calibration_sets(self):
        for n in (2, 3):
            with self.subTest(n=n):
                detector = LOFDetectorV2(n_neighbors=20).fit(self.calibration[:n])
                self.assertTrue(detector.model_.novelty)
                self.assertEqual(detector.model_.n_neighbors_, n - 1)
                self.assertTrue(np.all(np.isfinite(detector._train_score)))
                self.assertTrue(np.all(np.isfinite(detector.score(self.held_out).score)))

    def test_per_pack_mode_and_small_block_behavior_are_unchanged(self):
        detector = LOFDetectorV2(n_neighbors=3, mode="per_pack").fit(self.calibration)
        legacy = LOFDetector(n_neighbors=3, mode="per_pack").fit(self.calibration)
        self.assertIsNone(detector.model_)
        self.assertIsNone(detector._train_score)
        with self.assertRaises(RuntimeError):
            detector.threshold_from_train()
        for n in (2, len(self.calibration)):
            with self.subTest(n=n):
                result = detector.score_self_referential(self.calibration[:n], q=.9)
                expected = legacy.score_self_referential(self.calibration[:n], q=.9)
                np.testing.assert_array_equal(result.score, expected.score)
                self.assertEqual(result.threshold, expected.threshold)
                np.testing.assert_array_equal(result.flag, expected.flag)
                self.assertEqual(result.meta, expected.meta)

    def test_existing_validation_rejects_nonfinite_and_too_small_inputs(self):
        for standardize in (True, False):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(standardize=standardize, value=value):
                    calibration = self.calibration.copy()
                    calibration[0, 0] = value
                    with self.assertRaisesRegex(ValueError, "非有限值"):
                        LOFDetectorV2(standardize=standardize).fit(calibration)
        for calibration in (np.empty((0, 2)), self.calibration[:1]):
            with self.subTest(n=len(calibration)), self.assertRaisesRegex(ValueError, "太少"):
                LOFDetectorV2(standardize=False).fit(calibration)
        with self.assertRaises(ValueError):
            LOFDetectorV2(mode="unsupported")

    def test_unfitted_errors_are_preserved(self):
        detector = LOFDetectorV2()
        with self.assertRaises(RuntimeError):
            detector.threshold_from_train()
        with self.assertRaises(RuntimeError):
            detector.score(self.held_out)
        with self.assertRaises(RuntimeError):
            detector.transform(self.held_out)

    def test_refit_replaces_training_scores_and_scaler(self):
        detector = LOFDetectorV2(n_neighbors=3).fit(self.calibration)
        second = self.calibration[:6] * [2., 3.] + [20., -10.]
        second[-1] = [100., -10.]
        detector.fit(second)
        fresh = LOFDetectorV2(n_neighbors=3).fit(second)
        np.testing.assert_array_equal(detector.center_, fresh.center_)
        np.testing.assert_array_equal(detector.scale_, fresh.scale_)
        np.testing.assert_array_equal(detector._train_score, fresh._train_score)
        np.testing.assert_array_equal(detector.score(self.held_out).score,
                                      fresh.score(self.held_out).score)


if __name__ == "__main__":
    unittest.main()
