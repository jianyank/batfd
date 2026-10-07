"""Focused evaluation audit: declared semantics and verifier regression repros.

Only synthetic temporary runs are created. Guard tests reject semantic gaps;
manifest resealing isolates semantic checks from hash checks.
"""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))

from evaluation import aggregate_stress, episode_metrics
from inject import inject, scenarios
from protocol import confirm, file_hash, split_roles, write_json
from run_benchmark import method_table, run_raw, verify_run


def signal(n=320, seed=17):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .01, (n, 32, 20))
    x[:, :, :16:2] += np.arange(8)[None, None, :] * .002
    return x


def spec(onset=5):
    return dict(kind='offset', cells=[2], onset=onset,
                strength=3., seed=8, control=False)


def cells(n):
    result = np.zeros((n, 8))
    result[:, 2] = 5.
    return result


class DeclaredSemanticsTests(unittest.TestCase):
    def test_data_roles_are_disjoint_and_cover_all_rows(self):
        ids = np.repeat([6, 8, 9, 10], 80)
        for pack in (6, 8, 9, 10):
            fit, cal, dev, held = split_roles(ids, pack)
            np.testing.assert_array_equal(
                np.sort(np.r_[fit, cal, dev, held]), np.arange(len(ids)))
            self.assertFalse(np.any(ids[np.r_[fit, cal, dev]] == pack))
            self.assertTrue(np.all(ids[held] == pack))

    def test_stage_seeds_are_disjoint_and_injection_copies_input(self):
        x = signal(20)
        before = x.copy()
        for pack in (6, 8, 9, 10):
            dev = scenarios('development', pack)
            challenge = scenarios('challenge', pack)
            self.assertTrue({s['seed'] for s in dev}.isdisjoint(
                {s['seed'] for s in challenge}))
        altered = inject(x, spec(), .01)
        np.testing.assert_array_equal(x, before)
        np.testing.assert_array_equal(altered[:5], before[:5])
        untouched = np.arange(20) != 4
        np.testing.assert_array_equal(altered[:, :, untouched], before[:, :, untouched])

    def test_existing_identical_confirmation_is_not_new(self):
        row = episode_metrics(np.full(20, 2.), np.full(20, 2.), cells(20), 1., spec())
        self.assertFalse(row['new_detection'])
        self.assertTrue(row['reference_alarm'])
        self.assertIsNone(row['delay_windows'])

    def test_pointwise_credit_when_reference_clears_is_declared(self):
        reference = np.r_[np.full(7, 2.), np.zeros(13)]
        injected = np.full(20, 2.)
        row = episode_metrics(reference, injected, cells(20), 1., spec())
        # No rising edge in injected confirmation after onset, but pointwise credit is intended.
        confirmation = confirm(injected > 1., np.ones(20))
        self.assertTrue(confirmation[4:].all())
        self.assertTrue(row['new_detection'])
        self.assertTrue(row['reference_alarm'])
        self.assertEqual(row['delay_windows'], 2)

    def test_delay_measures_first_new_confirmation_window(self):
        reference = np.zeros(20)
        injected = np.r_[np.zeros(5), np.full(15, 2.)]
        row = episode_metrics(reference, injected, cells(20), 1., spec())
        self.assertEqual(row['delay_windows'], 4)
        # Pre-onset exceedances are retained; four prior exceedances permit delay=0.
        reference = np.r_[np.zeros(1), np.full(4, 2.), np.zeros(15)]
        injected = np.r_[np.zeros(1), np.full(19, 2.)]
        row = episode_metrics(reference, injected, cells(20), 1., spec())
        self.assertEqual(row['delay_windows'], 0)

    def test_detected_localization_is_at_first_new_confirmation(self):
        evidence = cells(20)
        evidence[9] = 0.
        evidence[9, 1] = 10.
        evidence[15, 2] = 100.
        row = episode_metrics(np.zeros(20), np.r_[np.zeros(5), np.full(15, 2.)],
                              evidence, 1., spec())
        self.assertEqual(row['delay_windows'], 4)
        self.assertFalse(row['top1'])

    def test_undetected_localization_is_retrospective_not_alarm_conditioned(self):
        evidence = np.zeros((20, 8))
        evidence[10, 2] = 10.
        row = episode_metrics(np.zeros(20), np.zeros(20), evidence, 1., spec())
        result = aggregate_stress([row])
        self.assertFalse(row['new_detection'])
        self.assertEqual(result['localization'], 1.)
        self.assertIsNone(result['conditional_top1'])


class VerifySemanticGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='evaluation_audit_')
        cls.baseline = Path(cls.temp.name) / 'baseline'
        ids = np.repeat([6, 8, 9, 10], 80)
        with contextlib.redirect_stdout(io.StringIO()):
            run_raw(signal(), ids, cls.baseline, method_names=('peer_spread',),
                    episode_length=12, onset=4)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.temp_run = tempfile.TemporaryDirectory(prefix='evaluation_repro_')
        self.addCleanup(self.temp_run.cleanup)
        self.out = Path(self.temp_run.name) / 'run'
        shutil.copytree(self.baseline, self.out)
        self.folder = self.out / 'raw' / '6' / 'peer_spread'

    def read(self, relative):
        return json.loads((self.out / relative).read_text(encoding='utf-8'))

    def reseal(self):
        manifest = self.read('manifest.json')
        manifest['files'] = {
            p.relative_to(self.out).as_posix(): file_hash(p)
            for p in sorted(self.out.rglob('*'))
            if p.is_file() and p.name != 'manifest.json'
        }
        write_json(self.out / 'manifest.json', manifest)

    def assert_semantically_rejected(self):
        self.reseal()
        with self.assertRaises(ValueError):
            verify_run(self.out, check_sources=False)

    def test_baseline_is_accepted(self):
        self.assertEqual(len(verify_run(self.out, check_sources=False)['raw_folds']), 4)

    def test_existing_event_consistency_check_rejects_changed_top1(self):
        file = self.folder / 'challenge_events.json'
        rows = json.loads(file.read_text(encoding='utf-8'))
        rows[0]['top1'] = not rows[0]['top1']
        write_json(file, rows)
        self.assert_semantically_rejected()

    def test_reject_fit_calibration_role_overlap(self):
        file = self.out / 'raw' / '6' / 'split.npz'
        with np.load(file, allow_pickle=False) as arrays:
            data = {k: arrays[k].copy() for k in arrays.files}
        data['calibration'] = data['fit'].copy()
        np.savez(file, **data)
        self.assert_semantically_rejected()

    def test_reject_development_scenario_using_held_indices(self):
        file = self.out / 'raw' / '6' / 'development_scenarios.json'
        rows = json.loads(file.read_text(encoding='utf-8'))
        with np.load(self.out / 'raw' / '6' / 'split.npz', allow_pickle=False) as split:
            rows[0]['original_indices'] = split['held'][:12].tolist()
        write_json(file, rows)
        self.assert_semantically_rejected()

    def test_reject_held_score_ids_for_another_pack(self):
        file = self.folder / 'held_scores.npz'
        with np.load(file, allow_pickle=False) as arrays:
            data = {k: arrays[k].copy() for k in arrays.files}
        data['ids'][:] = 8
        np.savez_compressed(file, **data)
        self.assert_semantically_rejected()

    def test_reject_wrong_q99_threshold_even_if_stress_is_self_consistent(self):
        file = self.folder / 'held_summary.json'
        held = json.loads(file.read_text(encoding='utf-8'))
        threshold = 1e12
        held['threshold'] = threshold
        write_json(file, held)
        summary = self.read('summary.json')
        for stage in ('development', 'challenge'):
            specs = self.read('raw/6/' + stage + '_scenarios.json')
            with np.load(self.folder / (stage + '_scores.npz'), allow_pickle=False) as arrays:
                length = len(arrays['reference']) // len(specs)
                events = [dict(episode_metrics(
                    arrays['reference'][i*length:(i+1)*length],
                    arrays['injected'][i*length:(i+1)*length],
                    arrays['cell_scores'][i*length:(i+1)*length], threshold, s),
                    episode=i, pack_id=s['pack_id']) for i, s in enumerate(specs)]
            write_json(self.folder / (stage + '_events.json'), events)
            metrics = aggregate_stress(events)
            for row in summary['stress_summary']:
                if row['pack_id'] == 6 and row['stage'] == stage:
                    row.update(metrics)
        chosen = next(r for r in summary['selections'] if r['pack_id'] == 6)
        dev = next(r for r in summary['stress_summary']
                   if r['pack_id'] == 6 and r['stage'] == 'development')
        chosen.update(dev)
        chosen['utility'] = (.55*dev['detection'] + .25*dev['worst_detection']
                             + .20*dev['localization'] - .30*dev['ref_alarm'])
        challenge = next(r for r in summary['stress_summary']
                         if r['pack_id'] == 6 and r['stage'] == 'challenge')
        for row in summary['selected_challenge']:
            if row['pack_id'] == 6:
                row.update(challenge)
        write_json(self.out / 'summary.json', summary)
        self.assert_semantically_rejected()

    def test_reject_selected_challenge_not_matching_selection(self):
        summary = self.read('summary.json')
        summary['selected_challenge'][0]['method'] = 'not_a_fitted_method'
        summary['selected_challenge'][0]['detection'] = 1.
        write_json(self.out / 'summary.json', summary)
        self.assert_semantically_rejected()

    def test_reject_complete_summary_with_all_fold_sections_missing(self):
        summary = self.read('summary.json')
        for key in ('raw_folds', 'frozen_folds', 'stress_summary', 'selections', 'selected_challenge'):
            summary[key] = []
        summary['method_summary'] = method_table([])
        write_json(self.out / 'summary.json', summary)
        self.assert_semantically_rejected()

    def test_reject_unconsumed_trailing_episode_score_rows(self):
        file = self.folder / 'challenge_scores.npz'
        with np.load(file, allow_pickle=False) as arrays:
            data = {k: arrays[k].copy() for k in arrays.files}
        for key in ('reference', 'injected', 'cell_scores'):
            data[key] = np.concatenate([data[key], data[key][-1:]], axis=0)
        np.savez_compressed(file, **data)
        self.assert_semantically_rejected()


    def test_baseline_accepts_verified_source_ids(self):
        file = Path(self.temp_run.name) / 'ids.npy'
        np.save(file, np.repeat([6, 8, 9, 10], 80))
        manifest = self.read('manifest.json')
        manifest['source_inputs'][str(file)] = file_hash(file)
        write_json(self.out / 'manifest.json', manifest)
        self.assertEqual(len(verify_run(self.out)['raw_folds']), 4)

    def test_reject_split_disagreeing_with_verified_source_ids(self):
        file = Path(self.temp_run.name) / 'ids.npy'
        ids = np.repeat([6, 8, 9, 10], 80)
        ids[0], ids[80] = ids[80], ids[0]
        np.save(file, ids)
        manifest = self.read('manifest.json')
        manifest['source_inputs'][str(file)] = file_hash(file)
        write_json(self.out / 'manifest.json', manifest)
        with self.assertRaises(ValueError):
            verify_run(self.out)

    def test_reject_held_score_indices_not_matching_split(self):
        file = self.folder / 'held_scores.npz'
        with np.load(file, allow_pickle=False) as arrays:
            data = {k: arrays[k].copy() for k in arrays.files}
        data['indices'][0] = data['indices'][-1]
        np.savez_compressed(file, **data)
        self.assert_semantically_rejected()

    def test_reject_swapped_calibration_and_development_roles(self):
        file = self.out / 'raw' / '6' / 'split.npz'
        with np.load(file, allow_pickle=False) as arrays:
            data = {k: arrays[k].copy() for k in arrays.files}
        data['calibration'], data['development'] = data['development'], data['calibration']
        np.savez(file, **data)
        self.assert_semantically_rejected()

    def test_reject_duplicate_raw_summary_row(self):
        summary = self.read('summary.json')
        summary['raw_folds'].append(summary['raw_folds'][0].copy())
        summary['method_summary'] = method_table(summary['raw_folds'])
        write_json(self.out / 'summary.json', summary)
        self.assert_semantically_rejected()

    def test_reject_missing_raw_fold_with_matching_aggregate(self):
        summary = self.read('summary.json')
        summary['raw_folds'].pop()
        summary['method_summary'] = method_table(summary['raw_folds'])
        write_json(self.out / 'summary.json', summary)
        self.assert_semantically_rejected()

    def test_reject_injected_score_length_mismatch(self):
        file = self.folder / 'challenge_scores.npz'
        with np.load(file, allow_pickle=False) as arrays:
            data = {k: arrays[k].copy() for k in arrays.files}
        data['injected'] = np.r_[data['injected'], 0.]
        np.savez_compressed(file, **data)
        self.assert_semantically_rejected()

    def test_reject_cell_score_shape_not_eight_cells(self):
        file = self.folder / 'challenge_scores.npz'
        with np.load(file, allow_pickle=False) as arrays:
            data = {k: arrays[k].copy() for k in arrays.files}
        data['cell_scores'] = np.c_[data['cell_scores'], np.zeros(len(data['reference']))]
        np.savez_compressed(file, **data)
        self.assert_semantically_rejected()

    def test_reject_scenario_stage_mismatch(self):
        file = self.out / 'raw' / '6' / 'development_scenarios.json'
        rows = json.loads(file.read_text(encoding='utf-8'))
        rows[0]['stage'] = 'challenge'
        write_json(file, rows)
        self.assert_semantically_rejected()


if __name__ == '__main__':
    unittest.main()
