"""Synthetic, temporary-output integration check; not experimental evidence."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")

from batfd import config, experiments
from batfd.baselines import lfaae
from batfd.data import cache, channels
from helpers import ROOT, script


class PipelineSmokeTests(unittest.TestCase):
    def test_real_downstream_pipeline_preserves_history(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as tmp, torch.random.fork_rng(devices=[]):
            out = Path(tmp)
            cfg = config.Config(yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8")))
            cfg["paths"] = {"outputs_dir": out, "data_dir": out / "data"}
            cfg["detect"].update(lof_n_neighbors=5, threshold_q=.95, persistence_windows=2)
            (out / "tables").mkdir()
            (out / "figures").mkdir()
            history = out / "tables/far_dualtrack.csv"
            history.write_text("historical-result\n", encoding="utf-8")
            old_figure = out / "figures/fig6_dualtrack_en.png"
            old_figure.write_bytes(b"historical-figure")
            labels = "pack_id,onset_point,n_train,boundary_ok,cells_expected\n"
            labels += "".join(f"{pid},52,32,True,[1]\n" for pid in (6, 8, 9, 10))
            (out / "tables/labels.csv").write_text(labels, encoding="utf-8")
            protected = {p: p.read_bytes() for p in (history, old_figure, out / "tables/labels.csv")}

            rng = np.random.default_rng(42)
            training = None
            datasets = (("StandTrainData", (6, 8, 9, 10)),
                        ("StandTestData1", (6, 8, 9, 10)),
                        ("StandTestData2", (5,)), ("StandTestData3", (2,)))
            for name, packs in datasets:
                ids = np.repeat(packs, 32).astype(np.int16)
                signal = rng.normal(.8, .05, (len(ids), 256, 20)).astype(np.float32)
                if name == "StandTrainData":
                    training = signal
                else:
                    # Ensure the case study includes a genuine score excursion.
                    for start in range(0, len(ids), 32):
                        signal[start + 24:start + 32, :, 0] += 1
                time = np.datetime64("2020-01-01", "ns") + np.tile(np.arange(32), len(packs)).astype("timedelta64[h]")
                ds = SimpleNamespace(name=name, path=out / f"{name}.mat", signal=signal,
                                     ids=ids, time=time, hidden=None, meta={}, time_status="synthetic")
                cache.save_cache(cfg, ds)

            torch.manual_seed(42)
            model = lfaae.build(cfg)
            x = torch.from_numpy(training[:, :, channels.input_cols(cfg)].transpose(0, 2, 1).copy())
            v = torch.from_numpy(training[:, :, channels.cell_voltage_cols(cfg)].transpose(0, 2, 1).copy())
            model.fit_normalizers(x, v)
            checkpoint = out / "runs/lfaae/best.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save({"model_state": model.state_dict()}, checkpoint)
            checkpoint_bytes = checkpoint.read_bytes()

            def run(name, args):
                module = script(name)
                output = io.StringIO()
                with patch.object(config, "load", return_value=cfg), \
                     patch("sys.argv", [name, *args]), \
                     patch("torch.cuda.is_available", return_value=False), \
                     contextlib.redirect_stdout(output):
                    result = module.main()
                self.assertEqual(result, 0, output.getvalue())

            detect_args = ["--model", "lfaae", "--methods", "fixed", "--no-progress"]
            run("05_detect", detect_args)
            run("06_far_dualtrack", ["--model", "lfaae", "--methods", "fixed", "--no-progress"])
            run("07_localize", ["--model", "lfaae"])
            run("08_tradeoff", ["--model", "lfaae", "--qs", ".9", ".95", "--persistence", "1", "2"])
            run("09_pack5_case", ["--model", "lfaae", "--top", "3"])

            exp = experiments.Experiment.create(cfg, "lfaae", "lfaae")
            self.assertEqual(len(list((out / "experiments").iterdir())), 1)
            tables = exp.directory / "tables"
            paths = sorted(tables.glob("*.csv"))
            self.assertEqual(len(paths), 6)
            snapshots = {p: p.read_bytes() for p in tables.iterdir()}
            rows_by_stage = {}
            for stage in ("detection", "far_dualtrack", "localization", "tradeoff", "pack5_case", "pack5_series"):
                matches = list(tables.glob(f"{stage}_*.csv"))
                self.assertEqual(len(matches), 1)
                rows = experiments.read_table(matches[0])
                self.assertTrue(rows, stage)
                self.assertTrue(all(r["experiment_id"] == exp.id for r in rows))
                rows_by_stage[stage] = rows
            detection = rows_by_stage["detection"]
            self.assertEqual(len(detection), 6)
            self.assertTrue(all("alarm_ever" in r and "early_detected" in r for r in detection))
            self.assertTrue(all("detected" not in r and "far_per_window" not in r for r in detection))
            self.assertTrue(all(r["early_detected"] == "" for r in detection if r["pack_id"] in ("2", "5")))
            tradeoff = rows_by_stage["tradeoff"]
            self.assertEqual(len(tradeoff), 8)
            self.assertTrue(all("early_detection_rate" in r and "detection_rate" not in r for r in tradeoff))
            self.assertEqual(len(rows_by_stage["pack5_case"]), 3)
            self.assertEqual(len(rows_by_stage["pack5_series"]), 32)

            # An ordinary rerun reuses reconstruction and the identical artifact.
            generations = {p: p.read_bytes() for p in (out / "runs/lfaae/recon").glob("*/reconstruction.json")}
            self.assertEqual(len(generations), 4)
            run("05_detect", detect_args)
            self.assertEqual(len(list(tables.glob("detection_*.csv"))), 1)
            for p, content in generations.items():
                self.assertEqual(p.read_bytes(), content)
            # Force rebuild creates a fresh generation without replacing old tables.
            run("05_detect", detect_args + ["--force-recon"])
            self.assertEqual(len(list(tables.glob("detection_*.csv"))), 2)
            for p, content in generations.items():
                self.assertNotEqual(p.read_bytes(), content)
            for p, content in snapshots.items():
                self.assertEqual(p.read_bytes(), content)

            far_id = rows_by_stage["far_dualtrack"][0]["artifact_id"]
            run("10_figures", ["--experiment-ids", exp.id, "--artifact-ids", far_id,
                               "--only", "6", "--langs", "en", "--no-pdf"])
            selections = list((out / "figures/selections").glob("*/selection.json"))
            self.assertEqual(len(selections), 1)
            selection = json.loads(selections[0].read_text(encoding="utf-8"))
            self.assertEqual(selection["experiment_ids"], [exp.id])
            self.assertFalse(selection["legacy_unverified"])
            self.assertEqual(selection["skipped"], [])
            self.assertEqual(len(selection["written"]), 1)
            image = Path(selection["written"][0])
            self.assertTrue(image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
            for p, content in protected.items():
                self.assertEqual(p.read_bytes(), content)
