"""隔离回归：不使用真实训练数据或历史实验产物。"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "15_phase5_e2e_lopo.py"
    spec = importlib.util.spec_from_file_location("phase5_e2e_lopo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GuardedHidden:
    def __init__(self, array, forbidden):
        self.array = array
        self.shape = array.shape
        self.forbidden = set(forbidden)

    def __getitem__(self, indices):
        if self.forbidden.intersection(np.asarray(indices).ravel().tolist()):
            raise AssertionError("读取了留出包物理目标")
        return self.array[indices]


class Phase5IsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_outer_split_preserves_order_and_excludes_held_out(self):
        ids = np.array([8, 6, 9, 8, 10, 6, 9, 10])
        calibration, held = self.module.split_outer_fold(ids, 8)
        np.testing.assert_array_equal(calibration, [1, 2, 4, 5, 6, 7])
        np.testing.assert_array_equal(held, [0, 3])
        self.assertEqual(set(ids[calibration]), {6, 9, 10})
        self.assertFalse(np.intersect1d(calibration, held).size)

    def test_unknown_missing_and_nonvector_ids_rejected(self):
        for ids, held in [(np.array([6, 8, 9]), 10),
                          (np.array([6, 8, 9, 10, 17]), 8),
                          (np.array([[6, 8, 9, 10]]), 8),
                          (np.array([6, 8, 9, 10]), 17),
                          (np.array([6, 8, 9, 10.5]), 8)]:
            with self.subTest(ids=ids, held=held), self.assertRaises(ValueError):
                self.module.split_outer_fold(ids, held)

    def test_training_and_inference_never_index_held_hidden(self):
        ids = np.repeat([6, 8, 9, 10], 3)
        cache = {"name": "synthetic", "ids": ids,
                 "signal": np.arange(12 * 4 * 2).reshape(12, 4, 2),
                 "hidden": GuardedHidden(np.arange(12 * 9).reshape(12, 9), [3, 4, 5]),
                 "time": None}
        calibration, held = self.module.split_outer_fold(ids, 8)
        train = self.module.build_fold_cache(cache, calibration, include_hidden=True)
        inference = self.module.build_fold_cache(cache, held, include_hidden=False)
        self.assertEqual(set(train["ids"]), {6, 9, 10})
        np.testing.assert_array_equal(train["hidden"], cache["hidden"].array[calibration])
        self.assertIsNone(inference["hidden"])
        np.testing.assert_array_equal(inference["signal"], cache["signal"][held])
        self.assertIs(cache["hidden"].__class__, GuardedHidden)

    def test_persistence_is_confirmed_on_fifth_not_first_window(self):
        scores = np.array([2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2])
        summary, windows = self.module.summarize_scores(scores, 1.0, 8,
                                                       np.arange(100, 111), persistence=5)
        self.assertEqual(summary["n_exceedance_windows"], 10)
        self.assertEqual(summary["n_persistence_confirmed_windows"], 2)
        self.assertEqual(summary["first_confirmed_alarm_window"], 4)
        self.assertEqual([w["window_index"] for w in windows if w["persistence_confirmed"]], [4, 5])
        self.assertEqual(windows[4]["original_index"], 104)
        self.assertFalse(windows[6]["exceeded_threshold"])

    def test_macro_and_weighted_rates_are_distinct(self):
        rows = [{"n_held_out_windows": 10, "n_exceedance_windows": 10,
                 "n_persistence_confirmed_windows": 6, "alarm_ever": True},
                {"n_held_out_windows": 90, "n_exceedance_windows": 0,
                 "n_persistence_confirmed_windows": 0, "alarm_ever": False}]
        summary = self.module.aggregate_summaries(rows)
        self.assertAlmostEqual(summary["macro_exceedance_rate"], .5)
        self.assertAlmostEqual(summary["weighted_exceedance_rate"], .1)
        self.assertAlmostEqual(summary["macro_persistence_confirmed_rate"], .3)
        self.assertAlmostEqual(summary["weighted_persistence_confirmed_rate"], .06)
        self.assertEqual(summary["n_packs_with_alarm"], 1)

    def test_bad_scores_and_bad_persistence_rejected(self):
        for scores, threshold, persistence in [(np.array([np.nan]), 1, 5),
                                                (np.array([1.0]), np.nan, 5),
                                                (np.array([1.0]), 1, 0)]:
            with self.subTest(), self.assertRaises(ValueError):
                self.module.summarize_scores(scores, threshold, 8, np.array([1]), persistence=persistence)


if __name__ == "__main__":
    unittest.main()
