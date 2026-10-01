import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from batfd.detect.lof import LOFDetector, apply_persistence

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "14_phase4_lopo_calibration.py"


def load_module():
    spec = importlib.util.spec_from_file_location("phase4_lopo_calibration", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phase4LopoCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_calibration_excludes_held_out_pack_and_uses_training_quantile(self):
        rng = np.random.default_rng(20261001)
        ids = np.repeat(np.array([6, 8, 9, 10]), 12)
        features = np.vstack([
            rng.normal(loc=pack_id / 10, scale=0.7, size=(12, 24))
            for pack_id in (6, 8, 9, 10)
        ])

        summary, windows = self.module.calibrate_pack(
            features, ids, 8, q=0.95, persistence=3,
            n_neighbors=4, standardize=True, norm_floor=1e-3,
        )

        train = features[ids != 8]
        held_out = features[ids == 8]
        reference = LOFDetector(
            n_neighbors=4, mode="train_novelty", standardize=True, norm_floor=1e-3
        ).fit(train)
        expected_threshold = reference.threshold_from_train(0.95)
        expected_scores = reference.score(held_out).score
        expected_exceedance = expected_scores > expected_threshold
        expected_confirmed = apply_persistence(expected_exceedance, 3)

        self.assertEqual(summary["pack_id"], 8)
        self.assertEqual(summary["n_calibration_windows"], 36)
        self.assertEqual(summary["n_held_out_windows"], 12)
        self.assertAlmostEqual(summary["calibration_threshold"], expected_threshold)
        self.assertEqual(summary["n_exceedance_windows"], int(expected_exceedance.sum()))
        self.assertEqual(summary["n_persistence_confirmed_windows"], int(expected_confirmed.sum()))
        self.assertEqual(
            summary["first_confirmed_alarm_window"],
            int(np.flatnonzero(expected_confirmed)[0]) if expected_confirmed.any() else None,
        )
        self.assertEqual([row["window_index"] for row in windows], list(range(12)))
        self.assertTrue(all(row["pack_id"] == 8 for row in windows))

    def test_calibration_rejects_misaligned_arrays_and_missing_pack(self):
        features = np.arange(96, dtype=float).reshape(4, 24)
        ids = np.array([6, 6, 9, 9])
        with self.assertRaisesRegex(ValueError, "行数一致"):
            self.module.calibrate_pack(
                features[:-1], ids, 6, q=0.99, persistence=5,
                n_neighbors=2, standardize=True, norm_floor=1e-3,
            )
        with self.assertRaisesRegex(ValueError, "留出包"):
            self.module.calibrate_pack(
                features, ids, 8, q=0.99, persistence=5,
                n_neighbors=2, standardize=True, norm_floor=1e-3,
            )

    def test_diagnostic_csv_refuses_to_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            self.module.write_csv(path, [{"pack_id": 8, "rate": 0.25}])
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.module.write_csv(path, [{"pack_id": 8, "rate": 0.5}])
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
