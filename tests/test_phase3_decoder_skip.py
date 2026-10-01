import copy
import importlib.util
import unittest
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "13_decoder_current_skip_ablation.py"
    spec = importlib.util.spec_from_file_location("phase3_decoder_skip", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phase3DecoderSkipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_variant_configs_change_only_current_skip(self):
        base = {
            "model": {"decoder_current_skip": True, "cell_resolved": True},
            "train": {"seed": 42, "lambda2": 0.1, "lambda3": 1e-3},
        }
        original = copy.deepcopy(base)
        variants = self.module.build_variant_configs(base)
        self.assertEqual(set(variants), {"skip_on", "skip_off"})
        self.assertTrue(variants["skip_on"]["model"]["decoder_current_skip"])
        self.assertFalse(variants["skip_off"]["model"]["decoder_current_skip"])
        for variant in variants.values():
            self.assertEqual(variant["model"]["cell_resolved"], True)
            self.assertEqual(variant["train"], base["train"])
        self.assertEqual(base, original)
        self.assertIsNot(variants["skip_on"], variants["skip_off"])

    def test_best_validation_metrics_uses_selected_epoch(self):
        result = {
            "best_epoch": 2,
            "best_val_loss": 0.25,
            "history": [
                {"epoch": 1, "val_loss": 0.3, "val_recon": 0.1},
                {"epoch": 2, "val_loss": 0.25, "val_recon": 0.08},
            ],
        }
        metrics = self.module.best_validation_metrics(result)
        self.assertEqual(metrics, {"best_epoch": 2, "best_val_loss": 0.25, "best_val_recon": 0.08})

    def test_evaluation_preserves_unlabelled_packs_as_unknown(self):
        caches = {
            "StandTestData1": {"ids": np.array([6, 6, 6, 9, 9]), "time": None},
        }
        statistics = {"StandTestData1": np.array([1.0, 1.0, -1.0, -1.0, -1.0])}
        labels = {6: {"onset_point": 2, "n_train": 0}}
        rows, summaries = self.module.evaluate_statistics(
            statistics, caches, labels, persistence=2
        )
        by_pack = {row["pack_id"]: row for row in rows}
        self.assertTrue(by_pack[6]["early_detected"])
        self.assertEqual(by_pack[6]["alarm_index"], 1)
        self.assertIsNone(by_pack[9]["early_detected"])
        self.assertEqual(summaries["StandTestData1"]["n_early_detected"], 1)
        self.assertEqual(summaries["StandTestData1"]["n_packs_with_onset_label"], 1)


if __name__ == "__main__":
    unittest.main()
