import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from features import extract
from methods import Detector, RAW_METHODS
from protocol import split_roles, confirm, summarize, select_candidate, file_hash, seal, verify_files
from inject import inject, scenarios, fit_amplitude
from system import CandidateSystem


def signal(n=120, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .002, (n, 32, 20)).astype(np.float32)
    wave = np.sin(np.linspace(0, 3, 32))[None, :, None] * .1
    x[:, :, :16:2] += wave
    return x


class ProtocolTests(unittest.TestCase):
    def test_roles_disjoint_and_ordered(self):
        ids = np.repeat([6, 8, 9, 10], 20)
        fit, cal, dev, held = split_roles(ids, 6)
        self.assertEqual(set(ids[fit]), {8, 9, 10})
        self.assertEqual([len(x) for x in (fit, cal, dev, held)], [36, 12, 12, 20])
        combined = np.concatenate((fit, cal, dev, held))
        self.assertEqual(len(np.unique(combined)), len(ids))
        for a in (fit, cal, dev, held):
            self.assertTrue(np.all(np.diff(a) > 0))

    def test_bad_roles_rejected(self):
        with self.assertRaises(ValueError):
            split_roles(np.repeat([6, 8, 9], 20), 6)
        with self.assertRaises(ValueError):
            split_roles(np.repeat([6, 8, 9, 10], 3), 6)

    def test_confirmation_causal_and_pack_reset(self):
        out = confirm(np.ones(12, dtype=bool), np.repeat([6, 8], 6), 5)
        np.testing.assert_array_equal(out, [0,0,0,0,1,1,0,0,0,0,1,1])
        before = confirm(np.array([1,1,1,0,0,0], dtype=bool), np.ones(6), 3)
        after = confirm(np.array([1,1,1,1,1,1], dtype=bool), np.ones(6), 3)
        np.testing.assert_array_equal(before[:3], after[:3])

    def test_summary_recomputes(self):
        row = summarize(np.arange(12.), 3., np.repeat([6, 8], 6))
        self.assertEqual(row['n_windows'], 12)
        self.assertEqual(row['n_exceeded'], 8)
        self.assertEqual(row['n_confirmed'], 2)
        self.assertNotIn('false_positive_rate', row)

    def test_select_uses_development_only_and_localization(self):
        rows = [dict(method='a', detection=.8, localization=.9, ref_alarm=.1, worst_detection=.7),
                dict(method='b', detection=.1, localization=1., ref_alarm=.01, worst_detection=0.)]
        self.assertEqual(select_candidate(rows)['method'], 'a')
        with self.assertRaises(ValueError):
            select_candidate([])

    def test_file_tamper_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'a.txt').write_text('first',encoding='utf-8')
            seal(p, {'status':'complete'})
            verify_files(p)
            (p/'a.txt').write_text('changed',encoding='utf-8')
            with self.assertRaises(ValueError):
                verify_files(p)


class FeaturesTests(unittest.TestCase):
    def test_common_mode_invariance_and_shape(self):
        x=signal(20); a=extract(x)
        y=x.copy(); y[:, :, :16:2] += .3
        b=extract(y)
        self.assertEqual(a['absolute'].shape, (20,40))
        self.assertEqual(a['peer'].shape, (20,32))
        np.testing.assert_allclose(a['peer'], b['peer'], atol=1e-7)
        self.assertGreater(np.max(np.abs(a['absolute']-b['absolute'])), .29)

    def test_nonfinite_and_bad_channels_rejected(self):
        x=signal(2); x[0,0,0]=np.nan
        with self.assertRaises(ValueError): extract(x)
        with self.assertRaises(ValueError): extract(np.ones((2,32,10)))


class MethodTests(unittest.TestCase):
    def test_every_detector_runs_without_fit_on_test(self):
        x=signal(); features=extract(x)
        fit={k:v[:80] for k,v in features.items()}
        held={k:v[80:] for k,v in features.items()}
        for name in RAW_METHODS:
            with self.subTest(method=name):
                model=Detector(name).fit(fit)
                center=model.center.copy()
                scores,cells=model.score(held,np.ones(40))
                self.assertEqual(scores.shape,(40,))
                self.assertEqual(cells.shape,(40,8))
                self.assertTrue(np.isfinite(scores).all())
                np.testing.assert_array_equal(center,model.center)

    def test_temporal_score_causality_and_boundaries(self):
        fit=extract(signal()); x=signal(20,11)
        for name in ('peer_ewma','peer_cusum','peer_multiscale'):
            model=Detector(name).fit(fit)
            a=model.score(extract(x),np.ones(20))[0]
            y=x.copy(); y[10:, :, 0]+=2
            b=model.score(extract(y),np.ones(20))[0]
            np.testing.assert_allclose(a[:10],b[:10])
            joined={k:np.concatenate((v,v)) for k,v in extract(x).items()}
            c=model.score(joined,np.repeat([6,8],20))[0]
            np.testing.assert_allclose(c[:20],c[20:])

    def test_single_cell_detection_and_localization(self):
        x=signal(); model=Detector('peer_robust').fit(extract(x))
        y=signal(10,20); y[:,:,6]+= .2
        score,cells=model.score(extract(y),np.ones(10))
        self.assertTrue(np.all(np.argmax(cells,axis=1)==3))
        self.assertGreater(float(np.median(score)),10.)


class InjectionTests(unittest.TestCase):
    def test_injection_reproducible_local_and_source_unchanged(self):
        x=signal(48); before=x.copy()
        spec=dict(kind='offset',cells=[2],strength=3.,onset=16,seed=42)
        a=inject(x,spec,.01); b=inject(x,spec,.01)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(x,before)
        np.testing.assert_array_equal(a[:16],x[:16])
        np.testing.assert_array_equal(a[:,:,np.arange(20)!=4],x[:,:,np.arange(20)!=4])
        self.assertGreater(abs(float(np.mean(a[16:,:,4]-x[16:,:,4]))),.029)

    def test_all_scenarios_and_no_units_assumed(self):
        x=signal(48); amp=fit_amplitude(x)
        self.assertGreater(amp,0.)
        specs=scenarios('challenge',9)
        self.assertTrue({'offset','drift','noise','stuck','intermittent','two_cell','common_shift','common_noise'} <= {s['kind'] for s in specs})
        for spec in specs:
            a=inject(x,spec,amp)
            self.assertEqual(a.shape,x.shape)
            self.assertTrue(np.isfinite(a).all())
        self.assertFalse({s['seed'] for s in specs} & {s['seed'] for s in scenarios('development',9)})


class SystemTests(unittest.TestCase):
    def test_streaming_batch_replay_and_serialization(self):
        x=signal(); train=extract(x[:60]); cal=extract(x[60:90])
        for name in ('peer_spread','peer_ewma','peer_cusum','peer_robust','peer_multiscale','pca_peer','iforest_peer'):
            with self.subTest(method=name):
                system=CandidateSystem.fit(name,train,cal,np.ones(30))
                seq=signal(24,17); a=system.predict(seq,pack_id=6)
                b1=system.predict(seq[:10],pack_id=6)
                b2=system.predict(seq[10:],pack_id=6,state=b1['state'])
                np.testing.assert_allclose(a['scores'],np.r_[b1['scores'],b2['scores']])
                np.testing.assert_array_equal(a['confirmed'],np.r_[b1['confirmed'],b2['confirmed']])
                with tempfile.TemporaryDirectory() as d:
                    file=Path(d)/'system.joblib'; system.save(file)
                    other=CandidateSystem.load(file)
                    np.testing.assert_allclose(a['scores'],other.predict(seq,pack_id=6)['scores'])
                with self.assertRaises(ValueError):
                    system.predict(seq[10:],pack_id=8,state=b1['state'])


class RunnerTests(unittest.TestCase):
    def test_episode_metrics_do_not_credit_existing_alarm(self):
        from evaluation import episode_metrics, aggregate_stress
        spec=dict(kind='offset',cells=[2],onset=5,strength=3.,seed=8,control=False)
        cells=np.zeros((20,8)); cells[:,2]=5
        row=episode_metrics(np.ones(20)*2,np.ones(20)*2,cells,1.,spec)
        self.assertFalse(row['new_detection'])
        self.assertTrue(row['reference_alarm'])
        self.assertIsNone(row['delay_windows'])
        self.assertTrue(row['top1'])
        specs=[dict(row,kind=k) for k in ('offset','noise')]
        result=aggregate_stress(specs)
        self.assertEqual(result['detection'],0.)
        self.assertEqual(result['worst_detection'],0.)

    def test_episode_delay_counts_confirmation_not_first_exceedance(self):
        from evaluation import episode_metrics
        spec=dict(kind='offset',cells=[2],onset=5,strength=3.,seed=8,control=False)
        cells=np.zeros((20,8)); cells[:,2]=5
        a=np.r_[np.zeros(5),np.ones(15)*2]
        row=episode_metrics(np.zeros(20),a,cells,1.,spec)
        self.assertTrue(row['new_detection'])
        self.assertEqual(row['delay_windows'],4)

    def test_mini_experiment_and_verify(self):
        from run_benchmark import run_raw, verify_run
        x=signal(320); ids=np.repeat([6,8,9,10],80)
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'run'
            summary=run_raw(x,ids,target,method_names=('peer_spread','peer_multiscale'),episode_length=12,onset=4)
            self.assertEqual(len(summary['raw_folds']),8)
            self.assertEqual(len(summary['selections']),4)
            verify_run(target,check_sources=False)
            with self.assertRaises(FileExistsError):
                run_raw(x,ids,target,method_names=('peer_spread',),episode_length=12,onset=4)


class IterationTests(unittest.TestCase):
    def test_anchored_reference_is_causal_fixed_and_localizes_change(self):
        x=signal(120); f=extract(x)
        model=Detector('peer_anchor').fit(f)
        y=signal(48,21); y[24:,:,4]+=.1
        scores,cells,state=model.score_with_state(extract(y),np.ones(48))
        self.assertTrue(np.all(scores[:16]==0))
        self.assertTrue(np.all(np.argmax(cells[28:],axis=1)==2))
        anchor=state['anchor'].copy()
        z=y.copy(); z[40:,:,4]+=1
        a=model.score(extract(z),np.ones(48))[0]
        np.testing.assert_allclose(a[:40],scores[:40])
        np.testing.assert_array_equal(anchor,state['anchor'])

    def test_new_temporal_methods_stream_and_reset(self):
        x=signal()
        for name in ('peer_anchor','peer_contrast'):
            s=CandidateSystem.fit(name,extract(x[:60]),extract(x[60:90]),np.ones(30))
            a=s.predict(x[90:],pack_id=6)
            b=s.predict(x[90:100],pack_id=6)
            c=s.predict(x[100:],pack_id=6,state=b['state'])
            np.testing.assert_allclose(a['scores'],np.r_[b['scores'],c['scores']])
            seq=np.r_[x[90:],x[90:]]
            score=s.detector.score(extract(seq),np.repeat([6,8],30))[0]
            np.testing.assert_allclose(score[:30],score[30:])

    def test_verifier_recomputes_events_from_saved_scores(self):
        from run_benchmark import run_raw, verify_run
        x=signal(320); ids=np.repeat([6,8,9,10],80)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run'; run_raw(x,ids,out,method_names=('peer_spread',),episode_length=12,onset=4)
            file=out/'raw'/'6'/'peer_spread'/'challenge_events.json'
            rows=json.loads(file.read_text(encoding='utf-8')); rows[0]['top1']=not rows[0]['top1']
            file.write_text(json.dumps(rows),encoding='utf-8')
            # Reseal to isolate semantic verification from hash tamper detection.
            manifest=out/'manifest.json'; content=json.loads(manifest.read_text(encoding='utf-8'))
            content['files'][file.relative_to(out).as_posix()]=file_hash(file)
            manifest.write_text(json.dumps(content),encoding='utf-8')
            with self.assertRaises(ValueError): verify_run(out,check_sources=False)

    def test_prediction_cli_creates_machine_readable_evidence(self):
        from predict import predict_file
        from run_benchmark import run_raw
        x=signal(320); ids=np.repeat([6,8,9,10],80)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run'; run_raw(x,ids,out,method_names=('peer_spread',),episode_length=12,onset=4)
            inputfile=Path(d)/'input.npy'; np.save(inputfile,x[:16])
            target=Path(d)/'prediction'; result=predict_file(out,6,inputfile,target,pack_id=99)
            self.assertEqual(result['n_windows'],16)
            self.assertTrue((target/'prediction.npz').is_file())
            self.assertFalse(result['real_fault_probability_available'])
            with self.assertRaises(FileExistsError): predict_file(out,6,inputfile,target,pack_id=99)


if __name__=='__main__':
    unittest.main()
