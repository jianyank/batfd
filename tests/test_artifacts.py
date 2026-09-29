import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import script


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        from batfd import experiments
        self.api = experiments
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = {"paths": {"outputs_dir": self.root}, "detect": {"threshold_q": .99}}
        d = self.root / "cache/StandTrainData"
        d.mkdir(parents=True)
        np.save(d / "signal.npy", np.array([.3], dtype=np.float32))
        p = self.root / "runs/ours_full/best.pt"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"checkpoint-A")
        self.exp = self.api.Experiment.create(self.cfg, "ours", "ours_full")

    def write(self, exp=None, stage="far_dualtrack", parameters=None, value=.1):
        return (exp or self.exp).write_table(stage, [{"model": "ours", "tag": "ours_full", "rate": value}],
                                             parameters or {"methods": ["fixed"]})

    def test_deterministic_identity(self):
        self.assertEqual(self.exp.id, self.api.Experiment.create(self.cfg, "ours", "ours_full").id)

    def test_checkpoint_change_creates_new_experiment(self):
        (self.root / "runs/ours_full/best.pt").write_bytes(b"checkpoint-B")
        new = self.api.Experiment.create(self.cfg, "ours", "ours_full")
        self.assertNotEqual(self.exp.id, new.id)

    def test_data_change_creates_new_experiment(self):
        np.save(self.root / "cache/StandTrainData/signal.npy", np.array([.4], dtype=np.float32))
        self.assertNotEqual(self.exp.id, self.api.Experiment.create(self.cfg, "ours", "ours_full").id)

    def test_config_change_creates_new_experiment(self):
        self.cfg["detect"]["threshold_q"] = .95
        self.assertNotEqual(self.exp.id, self.api.Experiment.create(self.cfg, "ours", "ours_full").id)

    def test_labels_change_creates_new_experiment(self):
        (self.root / "tables").mkdir()
        (self.root / "tables/labels.csv").write_text("pack_id,onset_point\n6,10\n", encoding="utf-8")
        self.assertNotEqual(self.exp.id, self.api.Experiment.create(self.cfg, "ours", "ours_full").id)

    def test_parameters_and_results_cannot_overwrite(self):
        first = self.write()
        original = first.read_bytes()
        second = self.write(parameters={"methods": ["pack_baseline"]})
        third = self.write(value=.2)
        self.assertEqual(len({first, second, third}), 3)
        self.assertEqual(first.read_bytes(), original)

    def test_legacy_tables_untouched(self):
        legacy = self.root / "tables/far_dualtrack.csv"
        legacy.parent.mkdir()
        legacy.write_text("historical-result", encoding="utf-8")
        output = self.write()
        self.assertNotEqual(output, legacy)
        self.assertEqual(legacy.read_text(encoding="utf-8"), "historical-result")

    def test_table_rows_bound_to_manifest(self):
        output = self.write()
        rows = self.api.read_table(output)
        self.assertEqual(rows[0]["experiment_id"], self.exp.id)
        self.assertTrue(rows[0]["artifact_id"])
        self.assertTrue(output.with_suffix(".json").is_file())

    def test_tampered_table_is_rejected(self):
        output = self.write()
        output.write_text("rate\n99\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "校验|损坏"):
            self.api.read_table(output)

    def test_missing_manifest_not_trusted(self):
        output = self.write()
        output.with_suffix(".json").unlink()
        with self.assertRaises(ValueError):
            self.api.read_table(output)

    def test_legacy_requires_explicit_opt_in(self):
        p = self.root / "old.csv"
        p.write_text("detected,lead_days\nTrue,2\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.api.read_table(p)
        self.assertEqual(self.api.read_table(p, allow_legacy=True)[0]["detected"], "True")

    def test_figure_loader_rejects_duplicate_variants(self):
        figures = script("10_figures")
        rows = [{"tag": "ours_full", "lof_mode": "train_novelty", "q": .99,
                 "persistence": 1, "trigger_rate_before_onset": .1, "detection_rate": .5}]
        self.exp.write_table("tradeoff", rows, {"q": .99})
        self.exp.write_table("tradeoff", rows, {"q": .98})
        with self.assertRaisesRegex(ValueError, "多个|歧义|重复"):
            figures.load_tradeoff(self.exp.directory / "tables")

    def test_manifest_parameter_tampering_is_rejected(self):
        output = self.write()
        sidecar = output.with_suffix(".json")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        meta["parameters"] = {"methods": ["fake"]}
        sidecar.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "校验|身份"):
            self.api.read_table(output)

    def test_wrong_experiment_directory_is_rejected(self):
        import shutil
        output = self.write()
        self.cfg["detect"]["threshold_q"] = .9
        other = self.api.Experiment.create(self.cfg, "ours", "ours_full")
        target = other.directory / "tables" / output.name
        target.parent.mkdir(parents=True)
        shutil.copyfile(output, target)
        shutil.copyfile(output.with_suffix(".json"), target.with_suffix(".json"))
        with self.assertRaises(ValueError):
            self.api.read_table(target)

    def test_localize_help_exits_successfully(self):
        import contextlib
        import io
        import sys
        from unittest.mock import patch
        localize = script("07_localize")
        with patch.object(sys, "argv", ["07_localize.py", "--help"]), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                localize.main()
        self.assertEqual(result.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
