"""Real CPU four-fold LOPO checks on synthetic data, not experimental evidence."""
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from batfd import config
from batfd.data import channels
from batfd.detect.lof import LOFDetector, apply_persistence
from batfd.features.fault_metric import paper_metrics
from batfd.models import inference
from batfd.models.cell_ae import RobustNormalizer
from batfd.models.losses import TargetScaler
from helpers import ROOT, script


PACKS = (6, 8, 9, 10)
ARTIFACTS = (
    "split.npz", "training/model/best.pt", "training/model/last.pt",
    "training/model/history.csv", "calibration_features.npy",
    "held_out_features.npy", "calibration_scores.npy", "held_out_scores.npy",
    "windows.csv", "summary.json",
)
SUMMARY_FIELDS = {
    "pack_id", "n_calibration_windows", "n_held_out_windows", "best_epoch",
    "best_val_loss", "best_val_recon", "epochs_run", "calibration_threshold",
    "exceedance_rate", "persistence_confirmed_rate", "alarm_ever",
    "first_confirmed_alarm_window",
}


class Phase5E2eLopoIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.module = script("15_phase5_e2e_lopo")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        self.cfg = config.Config(yaml.safe_load(
            (ROOT / "configs/base.yaml").read_text(encoding="utf-8")
        ))
        self.cfg["paths"] = {"outputs_dir": self.out, "data_dir": self.out / "data"}
        self.cfg["train"].update(
            device="cpu", batch_size=4, max_epochs=1, patience=1, amp=False,
        )
        self.cfg["model"].update(
            cell_latent=2, pack_latent=2, cell_conv=[4, 4, 4],
            pack_conv=[4, 4, 4], decoder_hidden=4,
        )
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        rng_state = torch.random.fork_rng(devices=[])
        rng_state.__enter__()
        self.addCleanup(rng_state.__exit__, None, None, None)
        # Unequal counts and interleaved rows expose local-to-original index bugs.
        counts = dict(zip(PACKS, (8, 9, 10, 11)))
        ids = np.array([pid for row in range(11) for pid in PACKS
                        if row < counts[pid]], dtype=np.int16)
        rng = np.random.default_rng(20261002)
        signal = rng.normal(.8, .04, (len(ids), 256, 20)).astype(np.float32)
        signal += ids[:, None, None] * .01
        signal += np.arange(len(ids), dtype=np.float32)[:, None, None] * .002
        hidden = rng.normal(.3, .015, (len(ids), 9)).astype(np.float32)
        hidden += ids[:, None] * .01
        hidden += np.arange(len(ids), dtype=np.float32)[:, None] * .003
        self.cache = {
            "name": "StandTrainData", "signal": signal, "ids": ids,
            "hidden": hidden, "time": None,
            "meta": {"n_windows": len(ids), "signal_shape": list(signal.shape),
                     "id_counts": {str(pid): counts[pid] for pid in PACKS},
                     "has_time": False, "has_hidden": True,
                     "time_status": "missing"},
        }

    def test_locked_public_api_is_available(self):
        for name in ("split_outer_fold", "build_fold_cache", "run_fold",
                     "run_experiment", "verify_run", "summarize_scores",
                     "aggregate_summaries"):
            with self.subTest(api=name):
                self.assertTrue(callable(getattr(self.module, name, None)), name)

    def test_fold_cache_isolates_rows_and_drops_inference_physical_targets(self):
        original = {key: value.copy() for key, value in self.cache.items()
                    if isinstance(value, np.ndarray)}
        for held_out_pack in PACKS:
            with self.subTest(pack=held_out_pack):
                calibration, held_out = self.module.split_outer_fold(
                    self.cache["ids"], held_out_pack,
                )
                np.testing.assert_array_equal(
                    calibration, np.flatnonzero(self.cache["ids"] != held_out_pack),
                )
                np.testing.assert_array_equal(
                    held_out, np.flatnonzero(self.cache["ids"] == held_out_pack),
                )
                train_cache = self.module.build_fold_cache(
                    self.cache, calibration, include_hidden=True,
                )
                infer_cache = self.module.build_fold_cache(
                    self.cache, held_out, include_hidden=False,
                )
                for key in ("signal", "ids", "hidden"):
                    np.testing.assert_array_equal(train_cache[key], original[key][calibration])
                for key in ("signal", "ids"):
                    np.testing.assert_array_equal(infer_cache[key], original[key][held_out])
                self.assertIsNone(infer_cache.get("hidden"))
                self.assertIsNone(train_cache.get("time"))
                self.assertIsNone(infer_cache.get("time"))
        for key, value in original.items():
            np.testing.assert_array_equal(self.cache[key], value)

    def _check_manifest_hashes(self, directory, required):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        files = manifest["files"]
        self.assertTrue(files)
        self.assertTrue(set(required).issubset(files), set(required) - set(files))
        for relative, expected in files.items():
            with self.subTest(artifact=str(directory / relative)):
                path = directory / relative
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        return manifest

    def _check_training_isolation(self, fold_dir, pack_id, split):
        ids = self.cache["ids"]
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        for pid in PACKS:
            if pid == pack_id:
                continue
            original = np.flatnonzero(ids == pid)
            n_val = max(1, int(round(self.cfg["split"]["val_ratio"] * len(original))))
            np.testing.assert_array_equal(train_idx[ids[train_idx] == pid], original[:-n_val])
            np.testing.assert_array_equal(val_idx[ids[val_idx] == pid], original[-n_val:])
        raw_train = self.cache["signal"][train_idx]
        floor = self.cfg["model"]["norm_floor"]
        x = torch.from_numpy(raw_train[:, :, channels.input_cols(self.cfg)].transpose(0, 2, 1).copy())
        v = torch.from_numpy(raw_train[:, :, channels.cell_voltage_cols(self.cfg)].transpose(0, 2, 1).copy())
        input_norm = RobustNormalizer(x.shape[1], floor=floor).fit(x)
        voltage_norm = RobustNormalizer(v.shape[1], floor=floor).fit(v)
        hidden = self.cache["hidden"][train_idx]
        res_cols = np.array(self.cfg["hidden"]["resistance_channels"]) - 1
        target = TargetScaler.fit(
            torch.from_numpy(hidden[:, res_cols].copy()),
            torch.from_numpy(hidden[:, self.cfg["hidden"]["soc_channel"] - 1].copy()),
            floor=floor,
        ).state_dict()
        for name in ("best.pt", "last.pt"):
            checkpoint = torch.load(fold_dir / "training/model" / name,
                                    map_location="cpu", weights_only=False)
            per_id = checkpoint["split_per_id"]
            self.assertEqual({int(pid) for pid in per_id}, set(PACKS) - {pack_id})
            for pid, counts in per_id.items():
                self.assertEqual(counts["n_train"], int(np.sum(ids[train_idx] == int(pid))))
                self.assertEqual(counts["n_val"], int(np.sum(ids[val_idx] == int(pid))))
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertTrue(checkpoint["model_state"])
            for prefix, norm in (("normalizer", input_norm), ("cell_norm", voltage_norm)):
                self.assertTrue(bool(checkpoint["model_state"][prefix + ".fitted"]))
                for statistic in ("median", "scale"):
                    torch.testing.assert_close(checkpoint["model_state"][prefix + "." + statistic],
                                               getattr(norm, statistic))
            for key, expected in target.items():
                torch.testing.assert_close(checkpoint["target_scaler"][key], expected)
        with (fold_dir / "training/model/history.csv").open(encoding="utf-8-sig", newline="") as fh:
            history = list(csv.DictReader(fh))
        self.assertEqual(len(history), 1)
        self.assertTrue(all(np.isfinite(float(history[0][key]))
                            for key in ("train_loss", "val_loss", "val_recon")))
        return history[0]

    def test_real_four_fold_run_isolated_artifacts_hashes_and_no_overwrite(self):
        run_dir = self.out / "four_fold_run"
        real_run_fold = self.module.run_fold
        real_reconstruct = inference.reconstruct_all
        order, reconstructed = [], []
        current_pack = None
        input_bytes = {key: value.tobytes() for key, value in self.cache.items()
                       if isinstance(value, np.ndarray)}

        def checked_fold(cfg, cache, held_out_pack, output_dir):
            nonlocal current_pack
            current_pack = held_out_pack
            order.append(held_out_pack)
            self.assertEqual(Path(output_dir), run_dir / "folds" / str(held_out_pack))
            self.assertFalse(Path(output_dir).exists())
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "running")
            with self.assertRaises(ValueError):
                self.module.verify_run(run_dir)
            return real_run_fold(cfg, cache, held_out_pack, output_dir)

        def checked_reconstruct(model, cfg, cache, **kwargs):
            self.assertIsNone(cache.get("hidden"), "Inference must not read Hiddall")
            self.assertEqual(next(model.parameters()).device.type, "cpu")
            result = real_reconstruct(model, cfg, cache, **kwargs)
            v_meas, v_rec, ids = result
            reconstructed.append((current_pack, np.asarray(ids).copy(),
                                  paper_metrics(v_meas, v_rec).oriented()))
            return result

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(self.module, "run_fold", side_effect=checked_fold))
            # Cover both module-qualified calls and a direct imported alias, preserving real computation.
            if getattr(self.module, "reconstruct_all", None) is real_reconstruct:
                stack.enter_context(patch.object(self.module, "reconstruct_all", side_effect=checked_reconstruct))
            stack.enter_context(patch.object(inference, "reconstruct_all", side_effect=checked_reconstruct))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result = self.module.run_experiment(self.cfg, self.cache, run_dir)
        self.assertEqual(order, list(PACKS))
        self.assertIsInstance(result, dict)
        self.assertTrue(result)
        self.assertTrue(reconstructed, "The real reconstruct_all must execute")
        required_run_files = []
        for pack_id in PACKS:
            with self.subTest(pack=pack_id):
                fold_dir = run_dir / "folds" / str(pack_id)
                for name in (*ARTIFACTS, "manifest.json"):
                    self.assertTrue((fold_dir / name).is_file(), name)
                    required_run_files.append(f"folds/{pack_id}/{name}")
                self._check_manifest_hashes(fold_dir, ARTIFACTS)
                with np.load(fold_dir / "split.npz", allow_pickle=False) as archive:
                    split = {key: archive[key].copy() for key in
                             ("calibration_idx", "held_out_idx", "train_idx", "val_idx")}
                calibration, held_out = split["calibration_idx"], split["held_out_idx"]
                for name, indices in split.items():
                    self.assertTrue(np.issubdtype(indices.dtype, np.integer), name)
                    self.assertEqual(np.unique(indices).size, indices.size, name)
                np.testing.assert_array_equal(calibration, np.flatnonzero(self.cache["ids"] != pack_id))
                np.testing.assert_array_equal(held_out, np.flatnonzero(self.cache["ids"] == pack_id))
                np.testing.assert_array_equal(np.sort(np.concatenate([calibration, held_out])),
                                              np.arange(len(self.cache["ids"])))
                self.assertEqual(np.intersect1d(calibration, held_out).size, 0)
                self.assertEqual(np.intersect1d(split["train_idx"], split["val_idx"]).size, 0)
                np.testing.assert_array_equal(
                    np.sort(np.concatenate([split["train_idx"], split["val_idx"]])), calibration,
                )
                history = self._check_training_isolation(fold_dir, pack_id, split)
                features, scores = {}, {}
                for label, indices in (("calibration", calibration), ("held_out", held_out)):
                    features[label] = np.load(fold_dir / f"{label}_features.npy", allow_pickle=False)
                    scores[label] = np.load(fold_dir / f"{label}_scores.npy", allow_pickle=False)
                    self.assertEqual(features[label].shape, (len(indices), 24))
                    self.assertEqual(scores[label].shape, (len(indices),))
                    self.assertTrue(np.isfinite(features[label]).all())
                    self.assertTrue(np.isfinite(scores[label]).all())
                    actual = [feature[ids == pack_id] if label == "held_out" else feature[ids != pack_id]
                              for pid, ids, feature in reconstructed if pid == pack_id]
                    actual = [feature for feature in actual if len(feature)]
                    np.testing.assert_allclose(features[label], np.concatenate(actual), rtol=1e-6, atol=1e-7)
                detector = LOFDetector(
                    n_neighbors=self.cfg["detect"]["lof_n_neighbors"], mode="train_novelty",
                    standardize=self.cfg["detect"]["score_normalize"],
                    norm_floor=self.cfg["model"]["norm_floor"],
                ).fit(features["calibration"])
                threshold = detector.threshold_from_train(self.cfg["detect"]["threshold_q"])
                for label in ("calibration", "held_out"):
                    np.testing.assert_allclose(scores[label], detector.score(features[label]).score)
                summary = json.loads((fold_dir / "summary.json").read_text(encoding="utf-8"))
                self.assertTrue(SUMMARY_FIELDS.issubset(summary))
                self.assertEqual(summary["pack_id"], pack_id)
                self.assertEqual(summary["n_calibration_windows"], len(calibration))
                self.assertEqual(summary["n_held_out_windows"], len(held_out))
                self.assertEqual(summary["best_epoch"], 1)
                self.assertEqual(summary["epochs_run"], 1)
                self.assertAlmostEqual(summary["best_val_loss"], float(history["val_loss"]))
                self.assertAlmostEqual(summary["best_val_recon"], float(history["val_recon"]))
                self.assertAlmostEqual(summary["calibration_threshold"], threshold)
                exceeded = scores["held_out"] > threshold
                confirmed = apply_persistence(exceeded, self.cfg["detect"]["persistence_windows"])
                self.assertAlmostEqual(summary["exceedance_rate"], float(exceeded.mean()))
                self.assertAlmostEqual(summary["persistence_confirmed_rate"], float(confirmed.mean()))
                self.assertEqual(summary["alarm_ever"], bool(confirmed.any()))
                self.assertEqual(summary["first_confirmed_alarm_window"],
                                 int(np.flatnonzero(confirmed)[0]) if confirmed.any() else None)
                with (fold_dir / "windows.csv").open(encoding="utf-8-sig", newline="") as fh:
                    windows = list(csv.DictReader(fh))
                self.assertEqual(len(windows), len(held_out))
                self.assertEqual([int(row["original_index"]) for row in windows], held_out.tolist())
                self.assertEqual([int(row["window_index"]) for row in windows], list(range(len(held_out))))
                np.testing.assert_allclose([float(row["score"]) for row in windows], scores["held_out"])
                for row, raw_flag, confirmed_flag in zip(windows, exceeded, confirmed):
                    self.assertEqual(row["exceeded_threshold"].lower(), str(bool(raw_flag)).lower())
                    self.assertEqual(row["persistence_confirmed"].lower(), str(bool(confirmed_flag)).lower())
        manifest = self._check_manifest_hashes(run_dir, required_run_files)
        self.assertEqual(manifest["status"], "complete")
        self.assertIsInstance(self.module.verify_run(run_dir), dict)
        for key, value in input_bytes.items():
            self.assertEqual(self.cache[key].tobytes(), value)
        snapshots = {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
        with self.assertRaises(FileExistsError):
            self.module.run_experiment(self.cfg, self.cache, run_dir)
        with self.assertRaises(FileExistsError):
            self.module.run_fold(self.cfg, self.cache, 6, run_dir / "folds/6")
        self.assertEqual(snapshots, {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()})
        # Change bytes without changing size: verification must recompute SHA-256.
        damaged = run_dir / "folds/8/held_out_features.npy"
        original = damaged.read_bytes()
        changed = bytearray(original)
        changed[-1] ^= 1
        damaged.write_bytes(changed)
        with self.assertRaises(ValueError):
            self.module.verify_run(run_dir)
        damaged.write_bytes(original)
        self.assertIsInstance(self.module.verify_run(run_dir), dict)
        # Re-seal deliberately corrupted indices: valid hashes must not hide leakage.
        split_path = run_dir / "folds/8/split.npz"
        fold_manifest_path = run_dir / "folds/8/manifest.json"
        root_manifest_path = run_dir / "manifest.json"
        protected = {path: path.read_bytes() for path in
                     (split_path, fold_manifest_path, root_manifest_path)}
        try:
            with np.load(split_path, allow_pickle=False) as archive:
                corrupted = {key: archive[key].copy() for key in archive.files}
            corrupted["train_idx"][0] = corrupted["held_out_idx"][0]
            np.savez(split_path, **corrupted)
            fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
            fold_manifest["files"]["split.npz"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
            fold_manifest_path.write_text(json.dumps(fold_manifest), encoding="utf-8")
            root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
            for relative in ("folds/8/split.npz", "folds/8/manifest.json"):
                root_manifest["files"][relative] = hashlib.sha256(
                    (run_dir / relative).read_bytes()
                ).hexdigest()
            root_manifest_path.write_text(json.dumps(root_manifest), encoding="utf-8")
            # Both manifests really match the modified bytes before structural verification.
            self._check_manifest_hashes(run_dir / "folds/8", ARTIFACTS)
            self._check_manifest_hashes(run_dir, required_run_files)
            with self.assertRaisesRegex(ValueError, "索引"):
                self.module.verify_run(run_dir)
        finally:
            for path, content in protected.items():
                path.write_bytes(content)
        self.assertIsInstance(self.module.verify_run(run_dir), dict)

    def test_failed_fold_marks_manifest_failed_and_cannot_be_verified(self):
        run_dir = self.out / "failed_run"
        def fail_fold(cfg, cache, held_out_pack, output_dir):
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "running")
            self.assertEqual(held_out_pack, PACKS[0])
            raise RuntimeError("injected fold failure")
        with patch.object(self.module, "run_fold", side_effect=fail_fold):
            with self.assertRaisesRegex(RuntimeError, "injected fold failure"):
                self.module.run_experiment(self.cfg, self.cache, run_dir)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertNotEqual(manifest["status"], "complete")
        with self.assertRaises(ValueError):
            self.module.verify_run(run_dir)
        snapshot = (run_dir / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.module.run_experiment(self.cfg, self.cache, run_dir)
        self.assertEqual((run_dir / "manifest.json").read_bytes(), snapshot)


if __name__ == "__main__":
    unittest.main()
