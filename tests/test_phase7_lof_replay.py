"""保存特征检测复算：真实小四折来源、隔离及独立语义验收。"""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from batfd import config, provenance
from helpers import ROOT, script


class Phase7ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = script("17_phase7_lof_replay")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.cache = cls.base / "cache"
        cls.cache.mkdir()
        cls.cfg = config.Config(yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8")))
        cls.cfg["train"].update(device="cpu", max_epochs=1, patience=1, batch_size=4, amp=False)
        cls.cfg["model"].update(cell_latent=2, pack_latent=2, cell_conv=[4,4,4],
                                pack_conv=[4,4,4], decoder_hidden=4)
        cls.ids = np.tile(np.array([6,8,9,10], dtype=np.int16), 8)
        rng = np.random.default_rng(20261002)
        signal = rng.normal(.8, .04, (32,256,20)).astype(np.float32)
        hidden = rng.normal(.3, .02, (32,9)).astype(np.float32)
        np.save(cls.cache / "signal.npy", signal)
        np.save(cls.cache / "ids.npy", cls.ids)
        provenance.write_manifest(cls.cache / "meta.json", {"name":"StandTrainData", "has_time":False})
        p5 = cls.module.PHASE5
        identity = {"config": {k:cls.cfg[k] for k in ("data","channels","hidden","split","model","train","detect")},
                    "implementation_sha256":provenance.source_digest(p5.SOURCE_PATHS),
                    "inputs": {"directory":str(cls.cache), "sha256":{n:provenance.file_digest(cls.cache/n)
                               for n in ("signal.npy","ids.npy","meta.json")}}}
        cls.source = cls.base / "source"
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with contextlib.redirect_stdout(io.StringIO()):
                p5.run_experiment(cls.cfg, {"name":"StandTrainData", "ids":cls.ids, "signal":signal,
                                          "hidden":hidden, "time":None}, cls.source, identity=identity)
        finally:
            torch.set_num_threads(threads)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def fresh_output(self):
        return self.base / (self.id().split(".")[-1])

    def test_csv_none_is_empty_as_written_by_dictwriter(self):
        path = self.fresh_output().with_suffix(".csv")
        rows = [{"pack_id":6, "first_confirmed_alarm_window":None}]
        self.module.PHASE5.write_csv(path, rows)
        self.module.check_csv(path, rows)

    def test_interleaved_groups_are_fixed_and_not_contiguous(self):
        x = np.arange(48).reshape(2,24)
        np.testing.assert_array_equal(self.module.select_features(x, "sigma_v"), x[:,0::3])
        np.testing.assert_array_equal(self.module.select_features(x, "reconstruction"), x[:,[j for j in range(24) if j%3]])
        np.testing.assert_array_equal(self.module.select_features(x, "all"), x)
        for bad in (np.zeros((2,23)), np.full((2,24), np.nan)):
            with self.assertRaises(ValueError):
                self.module.select_features(bad, "all")
        with self.assertRaises(ValueError):
            self.module.select_features(x, "winner")

    def test_fit_is_calibration_only_and_held_does_not_change_threshold(self):
        cal = np.random.default_rng(2).normal(size=(30,24))
        held = np.ones((7,24))
        real_fit = self.module.LOFDetectorV2.fit
        seen = []
        def checked_fit(detector, values):
            seen.append(values.copy())
            return real_fit(detector, values)
        with patch.object(self.module.LOFDetectorV2, "fit", checked_fit):
            a, scores, threshold = self.module.fit_scores(cal, held, self.cfg, "sigma_v")
            b, _, changed_threshold = self.module.fit_scores(cal, held * 1e6, self.cfg, "sigma_v")
        self.assertEqual(len(seen), 2)
        for values in seen:
            np.testing.assert_array_equal(values, cal[:,0::3])
        np.testing.assert_array_equal(a.center_, b.center_)
        self.assertEqual(threshold, changed_threshold)
        np.testing.assert_array_equal(a._train_score, -a.model_.negative_outlier_factor_)
        self.assertEqual(scores.shape, (7,))

    def test_real_saved_feature_replay_preserves_source_and_reads_no_hidden(self):
        before = self.module.PHASE6.tree_hashes(self.source)
        original_load = np.load
        def guarded_load(filename, *args, **kwargs):
            self.assertNotIn(Path(filename).name, ("hidden.npy","signal.npy","cond.npy","time.npy","labels.csv"))
            return original_load(filename, *args, **kwargs)
        output = self.fresh_output()
        with patch.object(np, "load", side_effect=guarded_load), patch.object(
                self.module.PHASE5.train_mod, "train", side_effect=AssertionError("不得重训")):
            result = self.module.run_replay(self.source, output, ablation=True)
            self.assertEqual(self.module.verify_replay(output)["status"], "complete")
        self.assertEqual(result["variants"], ["all","sigma_v","reconstruction"])
        self.assertEqual(result["n_windows"], 32)
        self.assertEqual(len(result["packs"]), 12)
        self.assertFalse(result["protocol"]["hidden_read"])
        self.assertFalse(result["protocol"]["tuned_on_held_out"])
        self.assertFalse(result["protocol"]["deep_retrained"])
        self.assertEqual(before, self.module.PHASE6.tree_hashes(self.source))
        for row in result["packs"]:
            pid, variant = row["pack_id"], row["variant"]
            base = output / variant / "folds" / str(pid)
            scores = np.load(base / "calibration_scores.npy")
            self.assertEqual(np.quantile(scores, .99), row["calibration_threshold"])
            if variant == "all":
                np.testing.assert_array_equal(np.load(base / "held_out_scores.npy"),
                                              np.load(self.source / "folds" / str(pid) / "held_out_scores.npy"))

    def test_refuses_existing_and_source_or_cache_tree_overlap(self):
        for output in (self.source, self.source/"nested", self.base,
                       self.cache, self.cache/"nested"):
            with self.subTest(output=output), self.assertRaises((ValueError, FileExistsError)):
                self.module.run_replay(self.source, output)
        output = self.fresh_output()
        output.mkdir()
        with self.assertRaises(FileExistsError):
            self.module.run_replay(self.source, output)

    def test_rehashed_summary_and_manifest_protocol_tampering_is_rejected(self):
        output = self.fresh_output()
        self.module.run_replay(self.source, output)
        manifest_text = (output/"manifest.json").read_text(encoding="utf-8")
        summary_text = (output/"summary.json").read_text(encoding="utf-8")
        original = json.loads(summary_text)
        for key, value in (("packs", []), ("n_windows",0), ("source_unchanged",False),
                           ("protocol",{**original["protocol"],"hidden_read":True})):
            altered = json.loads(summary_text)
            altered[key] = value
            provenance.write_manifest(output/"summary.json", altered)
            manifest = json.loads(manifest_text)
            manifest["files"]["summary.json"] = provenance.file_digest(output/"summary.json")
            provenance.write_manifest(output/"manifest.json", manifest)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.module.verify_replay(output)
        (output/"summary.json").write_text(summary_text, encoding="utf-8")
        manifest = json.loads(manifest_text)
        manifest["identity"]["protocol"]["training_score"] = "novelty_score_on_training"
        provenance.write_manifest(output/"manifest.json", manifest)
        with self.assertRaises(ValueError):
            self.module.verify_replay(output)

    def test_rehashed_scores_and_missing_file_inventory_are_rejected(self):
        output = self.fresh_output()
        self.module.run_replay(self.source, output)
        manifest_text = (output/"manifest.json").read_text(encoding="utf-8")
        path = output/"all/folds/6/calibration_scores.npy"
        content = path.read_bytes()
        scores = np.load(path)
        scores[0] += 1
        np.save(path, scores)
        manifest = json.loads(manifest_text)
        manifest["files"][path.relative_to(output).as_posix()] = provenance.file_digest(path)
        provenance.write_manifest(output/"manifest.json", manifest)
        with self.assertRaises(ValueError):
            self.module.verify_replay(output)
        path.write_bytes(content)
        manifest = json.loads(manifest_text)
        manifest["files"].pop("all/folds/6/windows.csv")
        provenance.write_manifest(output/"manifest.json", manifest)
        with self.assertRaises(ValueError):
            self.module.verify_replay(output)

    def test_rehashed_scaler_and_window_index_tampering_is_rejected(self):
        output = self.fresh_output()
        self.module.run_replay(self.source, output)
        manifest_text = (output/"manifest.json").read_text(encoding="utf-8")
        for relative in ("all/folds/6/lof_scaler.npz", "all/folds/6/windows.csv"):
            path = output/relative
            content = path.read_bytes()
            if path.suffix == ".npz":
                with np.load(path) as values:
                    center, scale = values["center"].copy(), values["scale"].copy()
                scale[0] *= 2
                np.savez(path, center=center, scale=scale)
            else:
                text = path.read_text(encoding="utf-8-sig")
                lines = text.splitlines()
                row = lines[1].split(",")
                row[2] = "999999"
                lines[1] = ",".join(row)
                path.write_text("\n".join(lines)+"\n", encoding="utf-8-sig")
            manifest = json.loads(manifest_text)
            manifest["files"][relative] = provenance.file_digest(path)
            provenance.write_manifest(output/"manifest.json", manifest)
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                self.module.verify_replay(output)
            path.write_bytes(content)
            (output/"manifest.json").write_text(manifest_text, encoding="utf-8")

    def test_output_status_files_never_replace_existing_targets(self):
        output = self.fresh_output()
        original_replace = Path.replace
        def refuse_locked_existing(path, target):
            target = Path(target)
            if output in target.parents and target.exists():
                raise PermissionError("Windows占用文件：禁止替换既有清单")
            return original_replace(path, target)
        with patch.object(Path, "replace", refuse_locked_existing):
            self.module.run_replay(self.source, output)
        self.assertEqual(self.module.verify_replay(output)["status"], "complete")

    def test_cli_run_and_verify_both_return_zero(self):
        output = self.fresh_output()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.module.main(["--run-dir",str(self.source),"--output-dir",str(output)]), 0)
            self.assertEqual(self.module.main(["--verify",str(output)]), 0)

    def test_fit_failure_marks_new_run_failed(self):
        output = self.fresh_output()
        with patch.object(self.module, "fit_scores", side_effect=RuntimeError("测试拟合失败")):
            with self.assertRaises(RuntimeError):
                self.module.run_replay(self.source, output)
        self.assertEqual(provenance.read_manifest(output/"manifest.json")["status"], "failed")
        with self.assertRaises(ValueError):
            self.module.verify_replay(output)


if __name__ == "__main__":
    unittest.main()
