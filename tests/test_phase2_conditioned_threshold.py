import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "12_phase2_conditioned_threshold.py"
    spec = importlib.util.spec_from_file_location("phase2_conditioned_threshold", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phase2ConditionedThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_observable_conditions_excludes_mean_soc(self):
        cache = {
            "cond": np.arange(30, dtype=float).reshape(3, 10),
            "meta": {
                "condition_names": [
                    "mean_abs_i", "std_i", "p95_abs_i", "disch_frac", "mean_v",
                    "std_v", "range_v", "mean_t", "std_t", "mean_soc",
                ]
            },
        }
        cond, names = self.module.observable_conditions(cache)
        self.assertEqual(cond.shape, (3, 9))
        self.assertEqual(names, list(self.module.OBSERVABLE_CONDITION_NAMES))
        self.assertNotIn("mean_soc", names)

    def test_conditioned_stat_uses_zero_decision(self):
        score = np.linspace(0.0, 1.0, 200)
        cond = np.column_stack([np.linspace(0.0, 1.0, 200)] * 9)
        models = self.module.fit_models(score, cond, 0.99, 10)
        self.assertEqual(models["quantile_regression"].decision, 0.0)
        self.assertEqual(models["binned_quantile"].decision, 0.0)
        stat = models["quantile_regression"].stat(score[:3], cond[:3])
        np.testing.assert_equal(stat.shape, (3,))

    def test_condition_coverage_identifies_out_of_range_values(self):
        train_cond = np.zeros((3, 9), dtype=float)
        train_cond[:, 0] = [0.0, 1.0, 2.0]
        test_cond = np.zeros((2, 9), dtype=float)
        test_cond[:, 0] = [-1.0, 1.0]
        cache = {"ids": np.array([6, 6]), "time": None}
        rows = self.module.condition_coverage_rows(
            train_cond, cache, test_cond, list(self.module.OBSERVABLE_CONDITION_NAMES)
        )
        row = next(item for item in rows if item["condition"] == "mean_abs_i")
        self.assertEqual(row["n_test_outside_train_range"], 1)
        self.assertAlmostEqual(row["test_outside_train_range_fraction"], 0.5)

    def test_diagnostic_csv_is_idempotent_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "comparison.csv"
            self.module.write_csv(path, [{"method": "fixed", "count": 1}])
            original = path.read_bytes()
            self.module.write_csv(path, [{"method": "fixed", "count": 1}])
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
                self.module.write_csv(path, [{"method": "fixed", "count": 2}])
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
