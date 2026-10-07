"""Acceptance tests for global selection, research bundles and warm-up quality."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from features import extract
from system import CandidateSystem
from test_research import signal


def development_rows():
    return [dict(method=name,pack_id=pack,stage='development',complexity=i,
                 detection=.8 if i else .4,worst_detection=.5 if i else .2,
                 localization=.6,ref_alarm=.1,n_events=36,
                 by_kind={'drift':.5,'noise':1.,'offset':.9})
            for pack in (6,8,9,10) for i,name in enumerate(('peer_spread','lof_peer'))]


class DeliveryTests(unittest.TestCase):
    def test_global_selection_ignores_challenge_results(self):
        from deploy import global_selection
        rows=development_rows()
        result=global_selection(rows+[dict(rows[0],stage='challenge',detection=1000.)])
        self.assertEqual(result[0]['method'],'lof_peer')
        self.assertAlmostEqual(result[0]['utility'],.655)
        self.assertEqual(result[0]['fold_count'],4)

    def test_global_selection_rejects_missing_or_duplicate_folds(self):
        from deploy import global_selection
        rows=development_rows()
        with self.assertRaises(ValueError): global_selection(rows[:-1])
        with self.assertRaises(ValueError): global_selection(rows+[rows[0]])

    def test_anchor_warmup_is_not_reported_as_ready(self):
        x=signal()
        system=CandidateSystem.fit('peer_anchor',extract(x[:60]),extract(x[60:90]),np.ones(30))
        a=system.predict(x[90:],pack_id=6)
        self.assertIn('ready',a)
        np.testing.assert_array_equal(a['ready'],np.arange(30)>=16)
        b=system.predict(x[90:100],pack_id=6)
        c=system.predict(x[100:],pack_id=6,state=b['state'])
        np.testing.assert_array_equal(a['ready'],np.r_[b['ready'],c['ready']])
        self.assertFalse(a['reference_health_verified'])

    def test_bundle_fit_calibration_are_disjoint_and_predict_replays(self):
        from deploy import build_bundle, predict_bundle
        from protocol import seal, write_json, verify_files
        x=signal(320); ids=np.repeat([6,8,9,10],80)
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory); source=home/'run'; source.mkdir()
            write_json(source/'summary.json',dict(stress_summary=development_rows()))
            seal(source,dict(status='complete',source_inputs={}))
            out=home/'bundle'; meta=build_bundle(x,ids,source,out,window_samples=32)
            with np.load(out/'split.npz',allow_pickle=False) as split:
                self.assertFalse(set(split['fit']) & set(split['calibration']))
                self.assertFalse(set(split['fit']) & set(split['excluded']))
                self.assertEqual(len(split['fit']),192)
                self.assertEqual(len(split['calibration']),64)
            self.assertEqual(meta['method'],'lof_peer')
            self.assertFalse(meta['deployment_approved'])
            file=home/'input.npy'; np.save(file,x[:32])
            prediction=predict_bundle(out,file,home/'prediction',pack_id=99)
            model=CandidateSystem.load(out/'candidate.joblib')
            expected=model.predict(x[:32],pack_id=99)
            with np.load(home/'prediction'/'prediction.npz',allow_pickle=False) as arrays:
                np.testing.assert_allclose(arrays['scores'],expected['scores'])
                np.testing.assert_array_equal(arrays['confirmed'],expected['confirmed'])
                self.assertTrue(arrays['ready'].all())
            self.assertEqual(prediction['n_windows'],32)
            verify_files(out); verify_files(home/'prediction')
            with self.assertRaises(FileExistsError): build_bundle(x,ids,source,out)
            np.save(file,x[:32,:4])
            with self.assertRaises(ValueError): predict_bundle(out,file,home/'invalid',pack_id=99)

    def test_bundle_tamper_is_rejected_before_model_load(self):
        from deploy import build_bundle, predict_bundle
        from protocol import seal, write_json
        x=signal(320); ids=np.repeat([6,8,9,10],80)
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory); source=home/'run'; source.mkdir()
            write_json(source/'summary.json',dict(stress_summary=development_rows()))
            seal(source,dict(status='complete',source_inputs={}))
            out=home/'bundle'; build_bundle(x,ids,source,out,window_samples=32)
            with (out/'candidate.joblib').open('ab') as stream: stream.write(b'tamper')
            file=home/'input.npy'; np.save(file,x[:16])
            with self.assertRaises(ValueError): predict_bundle(out,file,home/'prediction',pack_id=99)


if __name__=='__main__': unittest.main()
