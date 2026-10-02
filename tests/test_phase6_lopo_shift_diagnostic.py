"""只读诊断回归；真实实验结论不来自合成夹具。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml
from sklearn.neighbors import LocalOutlierFactor

from batfd import config, provenance
from batfd.detect.lof import LOFDetector
from helpers import ROOT, script


class Phase6DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = script("16_phase6_lopo_shift_diagnostic")

    def test_distribution_uses_only_calibration_scale_and_detects_both_tails(self):
        # 留出极值不能参与尺度拟合；绝对偏离和有向偏移不同。
        rows = self.module.distribution_rows(
            np.array([[0.], [1.], [2.]]), np.array([[-10.], [10.]]), ["x"],
            center=np.array([1.]), scale=np.array([2.]))
        self.assertEqual(rows[0]["held_median"], 0.)
        self.assertEqual(rows[0]["median_shift_in_cal_scale"], -.5)
        self.assertEqual(rows[0]["held_outside_cal_range_rate"], 1.)
        self.assertEqual(rows[0]["held_outside_cal_01_99_rate"], 1.)
        with self.assertRaises(ValueError):
            self.module.distribution_rows(np.ones((3, 2)), np.ones((2, 1)), ["x"])

    def test_calibration_scale_is_not_changed_by_held_out_and_constant_scores_are_valid(self):
        cal = np.array([[0.], [1.], [2.], [3.]])
        a = self.module.distribution_rows(cal, np.array([[10.]]), ["x"])[0]
        b = self.module.distribution_rows(cal, np.array([[1e9]]), ["x"])[0]
        self.assertEqual(a["cal_scale"], b["cal_scale"])
        self.assertEqual(a["cal_center"], b["cal_center"])
        shape, _ = self.module.alarm_shape(np.zeros(3), 1., 5)
        self.assertEqual(shape["n_exceedance_runs"], 0)
        self.assertIsNone(shape["first_confirmed_window"])
        with self.assertRaises(ValueError):
            self.module.alarm_shape(np.array([np.nan]), 1., 5)

    def test_alarm_runs_do_not_reset_confirmation_at_bin_boundary(self):
        # 六连击有两个确认窗口，不是六个事件；第5个窗口才确认。
        shape, bins = self.module.alarm_shape(np.array([0, 2, 2, 2, 2, 2, 2, 0.]), 1., 5)
        self.assertEqual(shape["n_exceedance_runs"], 1)
        self.assertEqual(shape["longest_exceedance_run"], 6)
        self.assertEqual(shape["n_runs_at_least_persistence"], 1)
        self.assertEqual(shape["n_confirmed"], 2)
        self.assertEqual(shape["first_confirmed_window"], 5)
        self.assertEqual(sum(row["n_confirmed"] for row in bins), 2)
        self.assertEqual(sum(row["n_windows"] for row in bins), 8)
        equality, _ = self.module.alarm_shape(np.ones(8), 1., 5)
        self.assertEqual(equality["n_exceeded"], 0)

    def test_score_audit_replays_old_definition_and_independently_checks_standard_lof(self):
        cal = np.array([[0., 0.], [0.1, 0.2], [0.2, -0.1], [0.3, 0.4],
                        [1., 1.], [4., 4.], [5., 4.], [9., -2.]])
        held = np.array([[0.15, 0.15], [8., 9.]])
        cfg = {"detect": {"lof_n_neighbors": 3, "score_normalize": True,
                          "threshold_q": .9, "persistence_windows": 5},
               "model": {"norm_floor": .001}}
        detector, audit, train_scores = self.module.audit_scores(cal, held, cfg)
        center = np.median(cal, axis=0)
        scale = np.maximum(np.median(np.abs(cal-center), axis=0)*1.4826,
                           np.maximum(.001*np.abs(center), 1e-12))
        reference = LocalOutlierFactor(n_neighbors=3, novelty=True).fit((cal-center)/scale)
        np.testing.assert_allclose(train_scores, -reference.negative_outlier_factor_)
        self.assertAlmostEqual(audit["standard_training_threshold"],
                               np.quantile(-reference.negative_outlier_factor_, .9))
        self.assertAlmostEqual(audit["old_threshold"],
                               np.quantile(-reference.score_samples((cal-center)/scale), .9))
        self.assertNotAlmostEqual(audit["old_threshold"], audit["standard_training_threshold"])
        np.testing.assert_array_equal(detector.center_, center)

    def test_geometry_queries_only_calibration_and_uses_interleaved_metric_groups(self):
        cal = np.zeros((4, 24))
        held = np.zeros((1, 24)); held[:, 1::3] = 1.
        detector = LOFDetector(n_neighbors=2, standardize=False).fit(cal)
        result = self.module.nearest_geometry(detector, cal, held, np.array([6, 6, 9, 9]))
        self.assertAlmostEqual(result["squared_distance_group_share"]["one_minus_cos"], 1.)
        self.assertAlmostEqual(result["squared_distance_group_share"]["sigma_v"], 0.)
        self.assertAlmostEqual(result["squared_distance_group_share"]["q3_err"], 0.)
        self.assertAlmostEqual(sum(result["neighbor_pack_share"].values()), 1.)
        self.assertAlmostEqual(result["median_kth_distance"], np.sqrt(8.))

    def test_geometry_reports_tail_concentration_separately_from_typical_window(self):
        cal = np.zeros((4, 24))
        held = np.zeros((2, 24))
        held[0, 0::3] = 1.
        held[1, 1::3] = 10.
        detector = LOFDetector(n_neighbors=2, standardize=False).fit(cal)
        geometry = self.module.nearest_geometry(detector, cal, held, np.array([6, 6, 9, 9]))
        self.assertAlmostEqual(geometry["squared_distance_group_share"]["one_minus_cos"], 100/101)
        self.assertAlmostEqual(geometry["window_median_group_share"]["one_minus_cos"], .5)
        self.assertAlmostEqual(geometry["top_1_percent_distance_energy_share"], 100/101)

    def test_real_four_fold_diagnostic_preserves_source_and_refuses_nested_or_existing_output(self):
        p5 = script("15_phase5_e2e_lopo")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache_dir = directory / "cache"; cache_dir.mkdir()
            cfg = config.Config(yaml.safe_load((ROOT/"configs/base.yaml").read_text(encoding="utf-8")))
            cfg["train"].update(device="cpu", max_epochs=1, patience=1, batch_size=4, amp=False)
            cfg["model"].update(cell_latent=2, pack_latent=2, cell_conv=[4,4,4],
                                pack_conv=[4,4,4], decoder_hidden=4)
            ids = np.repeat(np.array([6,8,9,10], dtype=np.int16), 8)
            rng = np.random.default_rng(42)
            signal = rng.normal(.8, .04, (32,256,20)).astype(np.float32)
            hidden = rng.normal(.3, .02, (32,9)).astype(np.float32)
            np.save(cache_dir/"signal.npy", signal); np.save(cache_dir/"ids.npy", ids)
            provenance.write_manifest(cache_dir/"meta.json", {"name":"StandTrainData", "has_time":False})
            identity = {"config": {key:cfg[key] for key in ("data","channels","hidden","split","model","train","detect")},
                        "implementation_sha256":provenance.source_digest(p5.SOURCE_PATHS),
                        "inputs": {"directory":str(cache_dir), "sha256":{name:provenance.file_digest(cache_dir/name)
                                  for name in ("signal.npy","ids.npy","meta.json")}}}
            source = directory/"source"
            threads = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                with contextlib.redirect_stdout(io.StringIO()):
                    p5.run_experiment(cfg, {"name":"StandTrainData", "ids":ids, "signal":signal,
                                          "hidden":hidden, "time":None}, source, identity=identity)
            finally:
                torch.set_num_threads(threads)
            before = {p.relative_to(source).as_posix():provenance.file_digest(p) for p in source.rglob("*") if p.is_file()}
            output = directory/"diagnostic"
            original_load = np.load
            original_conditions = self.module.compute_conditions
            def guarded_load(filename, *args, **kwargs):
                self.assertNotIn(Path(filename).name, ("hidden.npy", "cond.npy", "time.npy"))
                return original_load(filename, *args, **kwargs)
            def guarded_conditions(cfg, values, hidden=None):
                self.assertIsNone(hidden)
                return original_conditions(cfg, values, hidden=hidden)
            with patch.object(np, "load", side_effect=guarded_load), patch.object(
                    self.module, "compute_conditions", side_effect=guarded_conditions):
                result = self.module.run_diagnostic(source, output)
            self.assertEqual([row["pack_id"] for row in result["packs"]], [6,8,9,10])
            self.assertTrue(result["source_unchanged"])
            self.assertEqual(result["protocol"]["hidden_read"], False)
            self.assertEqual(result["protocol"]["tuned_on_held_out"], False)
            self.assertEqual(result["n_windows"], 32)
            saved = provenance.read_manifest(output/"manifest.json")
            self.assertEqual(saved["status"], "complete")
            self.assertEqual(self.module.verify_diagnostic(output)["status"], "complete")
            for name, expected in saved["files"].items():
                self.assertEqual(provenance.file_digest(output/name), expected)
            after = {p.relative_to(source).as_posix():provenance.file_digest(p) for p in source.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertFalse((cache_dir/"hidden.npy").exists())
            original_manifest = (output/"manifest.json").read_text(encoding="utf-8")
            original_summary = (output/"summary.json").read_text(encoding="utf-8")
            for key, value in (("packs", []), ("n_windows", 0), ("source_unchanged", False),
                               ("protocol", {**result["protocol"], "hidden_read": True})):
                altered = json.loads(original_summary); altered[key] = value
                provenance.write_manifest(output/"summary.json", altered)
                updated = json.loads(original_manifest)
                updated["files"]["summary.json"] = provenance.file_digest(output/"summary.json")
                provenance.write_manifest(output/"manifest.json", updated)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.module.verify_diagnostic(output)
            (output/"summary.json").write_text(original_summary, encoding="utf-8")
            updated = json.loads(original_manifest); updated["schema_version"] = 999
            provenance.write_manifest(output/"manifest.json", updated)
            with self.assertRaises(ValueError):
                self.module.verify_diagnostic(output)
            (output/"summary.json").unlink()
            updated = json.loads(original_manifest); updated["files"].pop("summary.json")
            provenance.write_manifest(output/"manifest.json", updated)
            with self.assertRaises(ValueError):
                self.module.verify_diagnostic(output)
            (output/"summary.json").write_text(original_summary, encoding="utf-8")
            (output/"manifest.json").write_text(original_manifest, encoding="utf-8")
            report = output/"report.md"
            report.write_text(report.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.module.verify_diagnostic(output)
            with self.assertRaises(FileExistsError):
                self.module.run_diagnostic(source, output)
            with self.assertRaises(ValueError):
                self.module.run_diagnostic(source, source/"nested")
            self.assertFalse((source/"nested").exists())
            root_manifest = source/"manifest.json"
            original_root = root_manifest.read_text(encoding="utf-8")
            altered_root = json.loads(original_root)
            altered_root["identity"]["config"]["detect"]["persistence_windows"] = 1
            provenance.write_manifest(root_manifest, altered_root)
            with self.assertRaises(ValueError):
                self.module.run_diagnostic(source, directory/"bad-persistence")
            root_manifest.write_text(original_root, encoding="utf-8")
            # hashes匹配但来源不对仍不可接受；IDS缓存独立来源必须核验。
            np.save(cache_dir/"ids.npy", ids[::-1])
            with self.assertRaises(ValueError):
                self.module.run_diagnostic(source, directory/"tampered")


if __name__ == "__main__":
    unittest.main()
