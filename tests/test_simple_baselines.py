import unittest

import numpy as np

from batfd.baselines.simple import (
    cusum,
    ewma,
    peer_spread_from_means,
    robust_center_scale,
    window_cell_mean,
)


class SimpleBaselineTests(unittest.TestCase):
    def test_window_cell_mean_and_peer_spread(self):
        signal = np.zeros((2, 3, 4), dtype=np.float32)
        signal[0, :, 0] = 1.0
        signal[0, :, 2] = 1.2
        signal[1, :, 0] = 1.0
        signal[1, :, 2] = 3.0
        means = window_cell_mean(signal, [0, 2])
        np.testing.assert_allclose(means, [[1.0, 1.2], [1.0, 3.0]])
        np.testing.assert_allclose(peer_spread_from_means(means), [0.1, 1.0], atol=1e-6)

    def test_ewma_is_causal(self):
        result = ewma(np.array([0.0, 1.0, 1.0]), alpha=0.5)
        np.testing.assert_allclose(result, [0.0, 0.5, 0.75])

    def test_cusum_resets_after_negative_evidence(self):
        result = cusum(np.array([0.0, 2.0, 0.0, 2.0]), center=0.0, scale=1.0, slack=0.5)
        np.testing.assert_allclose(result, [0.0, 1.5, 1.0, 2.5])

    def test_robust_center_scale_has_floor(self):
        center, scale = robust_center_scale(np.array([2.0, 2.0, 2.0]), floor=0.25)
        self.assertEqual(center, 2.0)
        self.assertEqual(scale, 0.25)

    def test_invalid_parameters_fail_loudly(self):
        with self.assertRaises(ValueError):
            ewma(np.ones(3), alpha=0.0)
        with self.assertRaises(ValueError):
            cusum(np.ones(3), center=0.0, scale=0.0)


if __name__ == '__main__':
    unittest.main()
