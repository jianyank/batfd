"""只读真实验收清单的回归测试；不合成真实标签。"""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np


class RealReadinessTests(unittest.TestCase):
    def test_missing_time_is_not_exposure(self):
        from audit_real_inputs import time_summary
        result = time_summary(None, np.array([6, 6]))
        self.assertFalse(result['available'])
        self.assertFalse(result['window_time_mapping_proven'])
        self.assertNotIn('exposure_days', result)

    def test_pack_boundary_is_not_a_time_reversal(self):
        from audit_real_inputs import time_summary
        result = time_summary(np.array([100, 110, 1, 3], dtype='datetime64[s]'), np.array([6, 6, 8, 8]))
        self.assertEqual(result['packs']['6']['negative_deltas'], 0)
        self.assertEqual(result['packs']['8']['median_positive_delta_seconds'], 2.)
        self.assertFalse(result['window_time_mapping_proven'])

    def test_duplicates_reversals_and_nat_are_counted(self):
        from audit_real_inputs import time_summary
        result = time_summary(np.array([10, 10, 5, 'NaT', 15], dtype='datetime64[s]'), np.ones(5, dtype=int))
        pack = result['packs']['1']
        self.assertEqual(pack['nat_count'], 1)
        self.assertEqual(pack['zero_deltas'], 1)
        self.assertEqual(pack['negative_deltas'], 1)
        self.assertEqual(pack['adjacent_valid_pairs'], 2)

    def test_invalid_time_shape_or_dtype_rejected(self):
        from audit_real_inputs import time_summary
        for times in [np.ones(2), np.ones(1, dtype='datetime64[s]'), np.ones((2, 1), dtype='datetime64[s]')]:
            with self.subTest(shape=times.shape, dtype=times.dtype):
                with self.assertRaises(ValueError):
                    time_summary(times, np.ones(2, dtype=int))

    def test_current_file_hash_drift_is_not_a_pass(self):
        from audit_real_inputs import check_manifest
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p/'source.txt').write_text('changed', encoding='utf-8')
            manifest = {'artifacts': [{'path': str(p/'source.txt'), 'sha256': '0'*64}]}
            (p/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            result = check_manifest(p/'manifest.json')
            self.assertFalse(result['all_recorded_files_match'])
            self.assertEqual(result['files'][0]['status'], 'hash_mismatch')

    def test_missing_recorded_file_remains_unverified(self):
        from audit_real_inputs import check_manifest
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p/'manifest.json').write_text(json.dumps({'protected_sources':[{'path':str(p/'missing'),'sha256':'0'*64}]}),encoding='utf-8')
            result = check_manifest(p/'manifest.json')
            self.assertFalse(result['all_recorded_files_match'])
            self.assertEqual(result['files'][0]['status'], 'missing')

    def test_dictionary_manifest_resolves_files_relative_to_manifest(self):
        import hashlib
        from audit_real_inputs import check_manifest
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p/'audit.json').write_bytes(b'{}')
            expected = hashlib.sha256(b'{}').hexdigest()
            (p/'manifest.json').write_text(json.dumps({'files': {'audit.json': expected}}), encoding='utf-8')
            self.assertTrue(check_manifest(p/'manifest.json')['all_recorded_files_match'])
            (p/'audit.json').write_bytes(b'changed')
            self.assertEqual(check_manifest(p/'manifest.json')['files'][0]['status'], 'hash_mismatch')

    def test_empty_manifest_is_not_verified(self):
        from audit_real_inputs import check_manifest
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'manifest.json'
            p.write_text('{}',encoding='utf-8')
            self.assertFalse(check_manifest(p)['all_recorded_files_match'])


if __name__ == '__main__':
    unittest.main()
