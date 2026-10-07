"""Repository input paths: synthetic artifacts and CLI help only, no training."""
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))

from run_benchmark import checked_frozen, run_frozen, run_raw


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenInputPathTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name)
        self.cache = self.home / 'custom_cache'
        self.frozen = self.home / 'custom_frozen'
        self.cache.mkdir()
        self.frozen.mkdir()
        self.ids = np.repeat([6, 8, 9, 10], 3)
        np.save(self.cache / 'ids.npy', self.ids)
        np.save(self.cache / 'signal.npy', np.zeros((12, 256, 20), dtype=np.float32))
        (self.cache / 'meta.json').write_text(json.dumps({'has_time': False}), encoding='utf-8')
        self.input_hashes = {str(self.cache / name): digest(self.cache / name)
                             for name in ('signal.npy', 'ids.npy', 'meta.json')}
        manifest = {'status': 'complete', 'identity': {'inputs': {'sha256':
                    {Path(name).name: value for name, value in self.input_hashes.items()}}}}
        (self.frozen / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        self.expected = {str(self.frozen / 'manifest.json'): digest(self.frozen / 'manifest.json')}
        for pack in (6, 8, 9, 10):
            fold = self.frozen / 'folds' / str(pack)
            fold.mkdir(parents=True)
            cal = np.flatnonzero(self.ids != pack)
            held = np.flatnonzero(self.ids == pack)
            np.savez(fold / 'split.npz', calibration_idx=cal, held_out_idx=held)
            np.save(fold / 'calibration_features.npy', np.zeros((9, 24)))
            np.save(fold / 'held_out_features.npy', np.zeros((3, 24)))
            files = {name: digest(fold / name) for name in
                     ('split.npz', 'calibration_features.npy', 'held_out_features.npy')}
            (fold / 'manifest.json').write_text(json.dumps({'files': files}), encoding='utf-8')
            self.expected.update({str(fold / name): value for name, value in files.items()})

    def check(self, input_hashes=None):
        return checked_frozen(self.ids, self.input_hashes if input_hashes is None else input_hashes,
                              cache_dir=self.cache, frozen_dir=self.frozen)

    def test_custom_paths_return_all_thirteen_verified_file_hashes(self):
        self.assertEqual(len(self.expected), 13)
        self.assertEqual(self.check(), self.expected)

    def test_changed_cache_fingerprints_are_rejected(self):
        for name in ('signal.npy', 'ids.npy', 'meta.json'):
            with self.subTest(file=name):
                file = self.cache / name
                original = file.read_bytes()
                file.write_bytes(original + b'tampered')
                hashes = dict(self.input_hashes, **{str(file): digest(file)})
                try:
                    with self.assertRaisesRegex(ValueError, 'cache differs'):
                        self.check(hashes)
                finally:
                    file.write_bytes(original)

    def test_changed_artifacts_in_every_fold_are_rejected(self):
        for pack in (6, 8, 9, 10):
            for name in ('split.npz', 'calibration_features.npy', 'held_out_features.npy'):
                with self.subTest(pack=pack, file=name):
                    file = self.frozen / 'folds' / str(pack) / name
                    original = file.read_bytes()
                    file.write_bytes(original + b'tampered')
                    try:
                        with self.assertRaisesRegex(ValueError, 'frozen artifact changed'):
                            self.check()
                    finally:
                        file.write_bytes(original)

    def test_rehashed_wrong_split_is_still_rejected(self):
        fold = self.frozen / 'folds' / '6'
        np.savez(fold / 'split.npz', calibration_idx=np.flatnonzero(self.ids != 6)[::-1],
                 held_out_idx=np.flatnonzero(self.ids == 6))
        manifest_file = fold / 'manifest.json'
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        manifest['files']['split.npz'] = digest(fold / 'split.npz')
        manifest_file.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'frozen split differs'):
            self.check()

    def test_incomplete_frozen_source_is_rejected(self):
        file = self.frozen / 'manifest.json'
        manifest = json.loads(file.read_text(encoding='utf-8'))
        manifest['status'] = 'failed'
        file.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'frozen source incomplete'):
            self.check()

    def test_frozen_path_keywords_are_accepted_without_running_models(self):
        inspect.signature(run_frozen).bind(self.home / 'out', {}, self.ids, frozen_dir=self.frozen)
        inspect.signature(run_raw).bind(None, self.ids, self.home / 'out', frozen_dir=self.frozen)


class RepositoryInputCliTests(unittest.TestCase):
    def check_help(self, script, args, options):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(HOME / script),
                                     *args, '--help'], cwd=temp, capture_output=True,
                                    text=True, encoding='utf-8', timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for option in options:
                self.assertIn(option, result.stdout)
            self.assertEqual(result.stderr, '')

    def test_benchmark_help_exposes_cache_and_frozen_directories(self):
        self.check_help('run_benchmark.py', [], ('--cache-dir', '--frozen-dir'))

    def test_deploy_build_help_exposes_cache_directory(self):
        self.check_help('deploy.py', ['build'], ('--cache-dir',))

    def test_smoke_demo_help_exposes_cache_directory(self):
        self.check_help('smoke_demo.py', [], ('--cache-dir',))


if __name__ == '__main__':
    unittest.main()
