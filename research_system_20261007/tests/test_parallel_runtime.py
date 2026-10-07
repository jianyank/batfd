"""Runtime regression tests: synthetic data only; all contracts must pass.

Run directly with --strict-repros or with unittest discovery.
All artifact fixtures use disposable temporary directories; no real run is loaded.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deploy
import predict
from features import extract
from methods import Detector, RAW_METHODS
from protocol import file_hash, seal, write_json, verify_files
from system import CandidateSystem

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('scores', 'cell_scores', 'ranked_cells', 'exceeded', 'confirmed',
        'alarm_started', 'alarm_cleared', 'ready')
TEMPORAL = ('peer_ewma', 'peer_cusum', 'peer_anchor', 'peer_contrast', 'peer_multiscale')


def sample(n=120, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .002, (n, 32, 20)).astype(np.float32)
    x[:, :, :16:2] += np.sin(np.linspace(0, 3, 32))[None, :, None] * .1
    return x


def fit_system(name='peer_spread'):
    x = sample()
    return CandidateSystem.fit(name, extract(x[:60]), extract(x[60:90]), np.ones(30))


def fold_run(home, system):
    run = home / 'fold_run'
    model = run / 'raw' / '6' / 'selected_system.joblib'
    model.parent.mkdir(parents=True)
    system.save(model)
    seal(run, dict(status='complete'))
    return run


def build(home, x=None, source_inputs=None):
    x = sample(320) if x is None else x
    ids = np.repeat([6, 8, 9, 10], 80)
    source = home / 'development_run'
    source.mkdir()
    rows = [dict(method='peer_spread', pack_id=p, stage='development', complexity=0,
                 detection=.4, worst_detection=.2, localization=.6, ref_alarm=.1,
                 by_kind={'offset': .4}) for p in (6, 8, 9, 10)]
    write_json(source / 'summary.json', dict(stress_summary=rows))
    seal(source, dict(status='complete', source_inputs=source_inputs or {}))
    bundle = home / 'bundle'
    meta = deploy.build_bundle(x, ids, source, bundle, window_samples=32)
    return bundle, meta


class VerifiedRuntimeTests(unittest.TestCase):
    def test_declared_forward_slash_paths_bind_without_normalizing_manifest_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            x = sample(320)
            signal_file, ids_file = home / 'signal.npy', home / 'ids.npy'
            np.save(signal_file, x)
            np.save(ids_file, np.repeat([6, 8, 9, 10], 80))
            declared = {p.as_posix(): file_hash(p) for p in (signal_file, ids_file)}
            _, meta = build(home, x, declared)
            self.assertTrue(meta['refit_inputs_verified'])
            self.assertEqual(meta['source_inputs'], declared)

    def test_verified_model_deserializes_snapshot_not_reopened_path(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.joblib'
            original = fit_system()
            original.save(model)
            digest = file_hash(model)
            import joblib
            deserialize = joblib.load
            def replace_path_then_load(snapshot):
                model.write_bytes(b'changed after byte verification')
                return deserialize(snapshot)
            with patch('system.joblib.load', side_effect=replace_path_then_load):
                result = CandidateSystem.load(model, expected_sha256=digest)
            self.assertEqual(result.threshold, original.threshold)
            self.assertNotEqual(file_hash(model), digest)

    def test_frozen_detector_replay_remains_available_but_raw_system_rejects(self):
        train = extract(sample())
        train['frozen'] = train['peer'][:, :24].copy()
        held = extract(sample(20, 19))
        held['frozen'] = held['peer'][:, :24].copy()
        for name in ('lof_frozen', 'iforest_frozen', 'svm_frozen'):
            with self.subTest(method=name):
                detector = Detector(name).fit(train)
                scores, cells = detector.score(held, np.ones(20))
                self.assertEqual(scores.shape, (20,))
                self.assertEqual(cells.shape, (20, 8))
                with self.assertRaisesRegex(ValueError, 'use Detector feature replay'):
                    CandidateSystem.fit(name, train, held, np.ones(20))

    def test_model_expected_hash_rejects_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.joblib'
            system = fit_system()
            system.save(model)
            expected = file_hash(model)
            loaded = CandidateSystem.load(model, expected_sha256=expected)
            self.assertEqual(loaded.threshold, system.threshold)
            with patch('system.joblib.load') as deserialize:
                with self.assertRaises(ValueError):
                    CandidateSystem.load(model, expected_sha256='0' * 64)
                deserialize.assert_not_called()

    def test_anchor_normal_calibration_threshold_unchanged(self):
        fit = extract(sample())
        cal = extract(sample(48, 19))
        ids = np.repeat([6, 8], 24)
        scores, _ = Detector('peer_anchor').fit(fit).score(cal, ids)
        system = CandidateSystem.fit('peer_anchor', fit, cal, ids)
        self.assertEqual(system.threshold, float(np.quantile(scores, .99)))

    def test_bundle_array_provenance_verified_and_ids_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            x = sample(320)
            signal_file = home / 'signal.npy'
            ids_file = home / 'ids.npy'
            np.save(signal_file, x)
            ids = np.repeat([6, 8, 9, 10], 80)
            np.save(ids_file, ids)
            declared = {str(p): file_hash(p) for p in (signal_file, ids_file)}
            bundle, meta = build(home, x, declared)
            self.assertTrue(meta['refit_inputs_verified'])
            source = home / 'development_run'
            with self.assertRaisesRegex(ValueError, 'ids'):
                deploy.build_bundle(x, np.roll(ids, 1), source, home / 'wrong_ids', window_samples=32)
            self.assertFalse((home / 'wrong_ids').exists())

    def test_bundle_empty_provenance_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            _, meta = build(Path(directory))
            self.assertFalse(meta['refit_inputs_verified'])

    def test_every_raw_method_all_outputs_match_multi_chunk_batch(self):
        train = sample()
        seq = sample(35, 19)
        seq[20:, :, 4] += .1
        for name in RAW_METHODS:
            with self.subTest(method=name):
                system = CandidateSystem.fit(name, extract(train[:60]), extract(train[60:90]), np.ones(30))
                full = system.predict(seq, 6)
                state = None
                pieces = []
                for a, b in ((0, 1), (1, 15), (15, 16), (16, 17), (17, 22), (22, 35)):
                    part = system.predict(seq[a:b], 6, state=state)
                    state = part['state']
                    pieces.append(part)
                for key in KEYS:
                    joined = np.concatenate([part[key] for part in pieces])
                    if np.issubdtype(full[key].dtype, np.floating):
                        np.testing.assert_allclose(joined, full[key], rtol=1e-12, atol=1e-12)
                    else:
                        np.testing.assert_array_equal(joined, full[key])
                self.assertEqual(state['run'], full['state']['run'])
                self.assertEqual(state['confirmed'], full['state']['confirmed'])
                self.assertEqual(set(state['detector']), set(full['state']['detector']))
                for key in state['detector']:
                    np.testing.assert_equal(state['detector'][key], full['state']['detector'][key])

    def test_candidate_cross_pack_requires_explicit_reset(self):
        seq = sample(35, 19)
        for name in TEMPORAL:
            with self.subTest(method=name):
                system = fit_system(name)
                first = system.predict(seq, 6)
                with self.assertRaisesRegex(ValueError, 'another pack'):
                    system.predict(seq, 8, state=first['state'])
                reset = system.predict(seq, 8, state=None)
                for key in KEYS:
                    np.testing.assert_array_equal(reset[key], first[key])

    def test_anchor_16_window_warmup_state_is_reusable(self):
        system = fit_system('peer_anchor')
        seq = sample(35, 27)
        first = system.predict(seq[:15], 6)
        saved = copy.deepcopy(first['state'])
        second = system.predict(seq[15:17], 6, state=first['state'])
        repeated = system.predict(seq[15:17], 6, state=first['state'])
        np.testing.assert_array_equal(first['ready'], np.zeros(15, dtype=bool))
        np.testing.assert_array_equal(second['ready'], [False, True])
        self.assertEqual(first['state']['detector']['warm_count'], saved['detector']['warm_count'])
        np.testing.assert_array_equal(first['state']['detector']['warm_sum'], saved['detector']['warm_sum'])
        for key in KEYS:
            np.testing.assert_array_equal(second[key], repeated[key])
        self.assertFalse(second['reference_health_verified'])

    def test_confirmation_strict_threshold_start_and_clear_across_chunks(self):
        system = fit_system()
        system.threshold = 1.
        scores = np.array([1., 2., 2., 2., 2., 2., 1., 2., 2., 2., 2., 2., 0.])
        cells = np.zeros((len(scores), 8))
        def scored(features, ids, state=None):
            start = (state or {}).get('offset', 0)
            stop = start + len(ids)
            return scores[start:stop], cells[start:stop], dict(offset=stop)
        with patch.object(system.detector, 'score_with_state', side_effect=scored):
            full = system.predict(sample(len(scores)), 6)
            pieces = []
            state = None
            for a, b in ((0, 5), (5, 6), (6, 11), (11, 12), (12, 13)):
                part = system.predict(sample(b-a), 6, state)
                state = part['state']
                pieces.append(part)
        np.testing.assert_array_equal(np.flatnonzero(full['alarm_started']), [5, 11])
        np.testing.assert_array_equal(np.flatnonzero(full['alarm_cleared']), [6, 12])
        np.testing.assert_array_equal(np.flatnonzero(full['confirmed']), [5, 11])
        for key in KEYS:
            np.testing.assert_array_equal(full[key], np.concatenate([p[key] for p in pieces]))

    def test_global_source_mismatch_rejected_before_joblib_load(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            bundle, _ = build(home)
            meta_file = bundle / 'bundle.json'
            meta = json.loads(meta_file.read_text(encoding='utf-8'))
            meta['inference_code_sha256']['system.py'] = '0' * 64
            write_json(meta_file, meta)
            (bundle / 'manifest.json').unlink()
            seal(bundle, dict(status='complete'))
            inp = home / 'input.npy'
            np.save(inp, sample(20))
            with patch.object(CandidateSystem, 'load') as load:
                with self.assertRaisesRegex(ValueError, 'source differs'):
                    deploy.predict_bundle(bundle, inp, home / 'out', 6)
                load.assert_not_called()

    def test_global_rejects_input_changed_after_load(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            bundle, _ = build(home)
            inp = home / 'input.npy'
            seq = sample(20)
            np.save(inp, seq)
            original_predict = CandidateSystem.predict
            def replace_after_load(model, signal, pack_id, state=None):
                result = original_predict(model, signal, pack_id, state)
                np.save(inp, seq + .1)
                return result
            with patch.object(CandidateSystem, 'predict', new=replace_after_load):
                with self.assertRaisesRegex(ValueError, 'input changed'):
                    deploy.predict_bundle(bundle, inp, home / 'out', 6)
            self.assertFalse((home / 'out').exists())

    def test_localization_is_independent_not_lof_causal_attribution(self):
        system = fit_system('lof_peer')
        seq = sample(20, 23)
        result = system.predict(seq, 6)
        peer = (extract(seq)['peer'] - system.detector.peer_center) / system.detector.peer_scale
        expected = np.max(np.abs(peer.reshape(-1, 8, 4)), axis=2)
        np.testing.assert_array_equal(result['cell_scores'], expected)
        self.assertIn('not causal attribution', result['localization_basis'])
        self.assertNotIn('probability', result)


class KnownRuntimeDefects(unittest.TestCase):
    def test_r1_detector_pack_boundary_chunk_equals_batch(self):
        train = extract(sample())
        first = sample(24, 31)
        first[:, :, 4] += .05
        second = sample(24, 32)
        joined = extract(np.concatenate((first, second)))
        deltas = {}
        for name in TEMPORAL:
            detector = Detector(name).fit(train)
            full, full_cells, _ = detector.score_with_state(joined, np.repeat([6, 8], 24))
            _, _, state = detector.score_with_state(extract(first), np.full(24, 6))
            partial, partial_cells, _ = detector.score_with_state(extract(second), np.full(24, 8), state)
            deltas[name] = float(max(np.max(np.abs(partial-full[24:])),
                                     np.max(np.abs(partial_cells-full_cells[24:]))))
        print('R1 score/cell max deltas at cross-pack chunk:', deltas)
        self.assertEqual(deltas, dict.fromkeys(TEMPORAL, 0.))

    def test_r2_fold_prediction_preserves_warmup_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            system = fit_system('peer_anchor')
            run = fold_run(home, system)
            inp = home / 'input.npy'
            seq = sample(20)
            np.save(inp, seq)
            result = predict.predict_file(run, 6, inp, home / 'out', 6)
            with np.load(home / 'out' / 'prediction.npz', allow_pickle=False) as arrays:
                self.assertIn('ready', arrays.files)
                np.testing.assert_array_equal(arrays['ready'], system.predict(seq, 6)['ready'])
            self.assertEqual(result['n_not_ready'], 16)

    def test_r3_fold_rejects_input_changed_after_load(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            system = fit_system()
            run = fold_run(home, system)
            inp = home / 'input.npy'
            old = sample(20)
            new = old.copy()
            new[:, :, 4] += .1
            np.save(inp, old)
            old_hash = file_hash(inp)
            original_predict = CandidateSystem.predict
            def replace_after_load(model, signal, pack_id, state=None):
                result = original_predict(model, signal, pack_id, state)
                np.save(inp, new)
                return result
            rejected = False
            with patch.object(CandidateSystem, 'predict', new=replace_after_load):
                try:
                    result = predict.predict_file(run, 6, inp, home / 'out', 6)
                except ValueError:
                    rejected = True
            if not rejected:
                with np.load(home / 'out' / 'prediction.npz', allow_pickle=False) as arrays:
                    np.testing.assert_allclose(arrays['scores'], system.predict(old, 6)['scores'])
                self.assertNotEqual(old_hash, result['input_sha256'])
                self.assertEqual(result['input_sha256'], file_hash(inp))
                print('R3 accepted old-array scores with new-file SHA256')
            self.assertTrue(rejected, 'changed input must be rejected, not sealed with the replacement hash')

    def test_r4_bundle_loads_verified_model_bytes_only(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            bundle, _ = build(home)
            model_file = bundle / 'candidate.joblib'
            frozen_hash = file_hash(model_file)
            replacement = CandidateSystem.load(model_file)
            replacement.threshold = -1.
            replacement_file = home / 'replacement.joblib'
            replacement.save(replacement_file)
            inp = home / 'input.npy'
            np.save(inp, sample(20))
            original_load = np.load
            def swap_during_input_load(file, *args, **kwargs):
                array = original_load(file, *args, **kwargs)
                if Path(file) == inp:
                    model_file.write_bytes(replacement_file.read_bytes())
                return array
            rejected = False
            with patch.object(deploy.np, 'load', side_effect=swap_during_input_load):
                try:
                    result = deploy.predict_bundle(bundle, inp, home / 'out', 6)
                except ValueError:
                    rejected = True
            if not rejected:
                self.assertEqual(result['threshold'], -1.)
                self.assertNotEqual(result['model_sha256'], frozen_hash)
                with self.assertRaises(ValueError):
                    verify_files(bundle)
                print('R4 accepted model changed between manifest check and joblib.load')
            self.assertTrue(rejected, 'model bytes must still match the verified manifest at load time')

    def test_r5_build_input_matches_declared_source_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            original = sample(320)
            inp = home / 'signal.npy'
            np.save(inp, original)
            id_file = home / 'ids.npy'
            np.save(id_file, np.repeat([6, 8, 9, 10], 80))
            declared = {str(inp): file_hash(inp), str(id_file): file_hash(id_file)}
            changed = original.copy()
            changed[:, :, 4] += .1
            rejected = False
            try:
                bundle, meta = build(home, changed, declared)
            except ValueError:
                rejected = True
            if not rejected:
                self.assertEqual(meta['source_inputs'], declared)
                model = CandidateSystem.load(bundle / 'candidate.joblib')
                self.assertGreater(float(model.detector.spread_center), .09)
                print('R5 fitted substituted array but retained original source_input hash')
            self.assertTrue(rejected, 'build must bind provided arrays to the recorded source inputs')

    def test_r6_anchor_calibration_requires_at_least_one_ready_row(self):
        cal = sample(24, 19)
        ids = np.repeat([6, 8, 9, 10], 6)
        scores, _ = Detector('peer_anchor').fit(extract(sample())).score(extract(cal), ids)
        np.testing.assert_array_equal(scores, np.zeros(24))
        with self.assertRaises(ValueError, msg='all calibration rows are warmup; no ready scores exist'):
            system = CandidateSystem.fit('peer_anchor', extract(sample()), extract(cal), ids)
            print('R6 accepted anchor calibration without ready rows; threshold=', system.threshold)

    def test_r7_frozen_fit_has_predict_contract(self):
        train = extract(sample())
        cal = extract(sample(30, 19))
        train['frozen'] = train['peer'].copy()
        cal['frozen'] = cal['peer'].copy()
        errors = {}
        for name in ('lof_frozen', 'iforest_frozen', 'svm_frozen'):
            try:
                system = CandidateSystem.fit(name, train, cal, np.ones(30))
            except ValueError:
                continue  # Explicit fit-time rejection is also a valid API contract.
            try:
                result = system.predict(sample(20), 6)
                self.assertEqual(result['scores'].shape, (20,))
            except KeyError as error:
                errors[name] = str(error)
        print('R7 fit accepted but raw-signal prediction failed:', errors)
        self.assertEqual(errors, {})

    def test_r8_fold_cli_cannot_write_outside_isolated_research(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            run = fold_run(home, fit_system())
            inp = home / 'input.npy'
            np.save(inp, sample(20))
            args = ['--run-dir', str(run), '--fold', '6', '--input', str(inp),
                    '--output-dir', str(home / 'out'), '--pack-id', '6']
            command = [sys.executable, '-X', 'utf8', '-B', str(ROOT / 'predict.py'), *args]
            completed = subprocess.run(command, cwd=ROOT, text=True, encoding='utf-8', capture_output=True)
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertFalse((home / 'out').exists())


if __name__ == '__main__':
    if '--strict-repros' in sys.argv:
        sys.argv.remove('--strict-repros')
    unittest.main(verbosity=2)
