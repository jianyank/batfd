import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from batfd.models import inference
from helpers import ROOT, script

DETECT = script("05_detect")


class IdentityNorm(torch.nn.Module):
    def inverse(self, x):
        return x


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.0))
        self.cell_norm = IdentityNorm()

    def forward(self, x):
        return x[:, :1].expand(-1, 8, -1) * self.gain


class CacheIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
        self.cfg["paths"]["outputs_dir"] = self.tmp.name
        self.model = TinyModel()
        self.cache = {"signal": np.full((3, 256, 20), .3, dtype=np.float32),
                      "ids": np.array([6, 6, 8], dtype=np.int16)}
        self.d = inference.reconstruction_dir(self.cfg, "tiny") / "small"

    def recon(self, **kwargs):
        return inference.reconstruct_dataset(self.model, self.cfg, self.cache,
                                            "tiny", "small", mmap=False, **kwargs)

    def feature(self, meas, rec, **kwargs):
        arrays = DETECT.get_metrics(self.cfg, "tiny", "small", meas, rec, **kwargs)
        copies = tuple(np.array(a) for a in arrays)
        for a in arrays:
            if isinstance(a, np.memmap):
                a._mmap.close()
        return copies

    def test_changed_reconstruction_invalidates_features(self):
        meas = np.full((3, 8, 256), .3, dtype=np.float32)
        self.feature(meas, meas)
        q3 = self.feature(meas, np.zeros_like(meas))[3]
        np.testing.assert_allclose(q3, .3)

    def test_same_inputs_hit_both_caches(self):
        first = self.recon()
        feat = self.feature(*first[:2])
        with patch.object(inference, "reconstruct_all", side_effect=AssertionError("cache miss")):
            second = self.recon()
        with patch.object(DETECT.fault_metric, "paper_metrics", side_effect=AssertionError("cache miss")):
            feat2 = self.feature(*second[:2])
        for a, b in zip(first + feat, second + feat2):
            np.testing.assert_array_equal(a, b)

    def test_weight_change_invalidates_reconstruction(self):
        self.recon()
        with torch.no_grad():
            self.model.gain.fill_(2)
        np.testing.assert_allclose(self.recon()[1], .6)

    def test_same_shape_data_change_invalidates_reconstruction(self):
        self.recon()
        self.cache["signal"][:] = .7
        np.testing.assert_allclose(self.recon()[0], .7)

    def test_id_change_invalidates_reconstruction(self):
        self.recon()
        self.cache["ids"][0] = 10
        self.assertEqual(self.recon()[2][0], 10)

    def test_config_change_recomputes(self):
        self.recon()
        self.cfg["channels"]["dedup_current"] = not self.cfg["channels"]["dedup_current"]
        with patch.object(inference, "reconstruct_all", wraps=inference.reconstruct_all) as calc:
            self.recon()
        self.assertEqual(calc.call_count, 1)

    def test_force_cascades_even_when_values_unchanged(self):
        first = self.recon()
        self.feature(*first[:2])
        forced = self.recon(force=True)
        with patch.object(DETECT.fault_metric, "paper_metrics", wraps=DETECT.fault_metric.paper_metrics) as calc:
            self.feature(*forced[:2])
        self.assertEqual(calc.call_count, 1)

    def test_legacy_cache_without_manifest_recomputes(self):
        self.recon()
        for p in self.d.glob("*.json"):
            p.unlink()
        with patch.object(inference, "reconstruct_all", wraps=inference.reconstruct_all) as calc:
            self.recon()
        self.assertEqual(calc.call_count, 1)

    def test_truncated_cached_array_recomputes(self):
        self.recon()
        (self.d / "v_rec.npy").write_bytes(b"interrupted write")
        np.testing.assert_allclose(self.recon()[1], .3)

    def test_feature_force_recomputes(self):
        meas, rec, _ = self.recon()
        self.feature(meas, rec)
        with patch.object(DETECT.fault_metric, "paper_metrics", wraps=DETECT.fault_metric.paper_metrics) as calc:
            self.feature(meas, rec, force=True)
        self.assertEqual(calc.call_count, 1)


    def test_checkpoint_content_change_invalidates(self):
        ckpt = inference.checkpoint_path(self.cfg, "tiny")
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"weights-a")
        self.recon()
        ckpt.write_bytes(b"weights-b")
        with patch.object(inference, "reconstruct_all", wraps=inference.reconstruct_all) as calc:
            self.recon()
        self.assertEqual(calc.call_count, 1)

    def test_broken_json_recomputes(self):
        self.recon()
        (self.d / "reconstruction.json").write_text("{broken", encoding="utf-8")
        with patch.object(inference, "reconstruct_all", wraps=inference.reconstruct_all) as calc:
            self.recon()
        self.assertEqual(calc.call_count, 1)

    def test_feature_array_tampering_recomputes(self):
        meas, rec, _ = self.recon()
        self.feature(meas, rec)
        np.save(self.d / "q3.npy", np.ones((3, 8)))
        np.testing.assert_allclose(self.feature(meas, rec)[3], 0)

    def test_downstream_rejects_stale_reconstruction(self):
        self.recon()
        self.cache["signal"][:] = .7
        with self.assertRaisesRegex(ValueError, "05_detect"):
            inference.load_reconstruction(self.cfg, self.cache, "tiny", "small", mmap=False)

    def test_downstream_rejects_stale_metrics(self):
        from batfd.features import cache as metrics_cache
        meas, rec, _ = self.recon()
        self.feature(meas, rec)
        (self.d / "metrics.json").unlink()
        with self.assertRaisesRegex(ValueError, "05_detect"):
            metrics_cache.load_metrics(self.cfg, self.cache, "tiny", "small")

    def test_downstream_validated_read_matches_fresh(self):
        from batfd.features import cache as metrics_cache
        fresh = self.recon()
        expected = self.feature(*fresh[:2])
        arrays = metrics_cache.load_metrics(self.cfg, self.cache, "tiny", "small")
        for a, b in zip(expected, arrays):
            np.testing.assert_array_equal(a, b)
            b._mmap.close()

if __name__ == "__main__":
    unittest.main()
