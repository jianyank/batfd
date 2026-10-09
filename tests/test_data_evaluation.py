import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from chronoguard.data import make_windows, split_train, load_smd, download_smd, SMD_SOURCE
from chronoguard.evaluation import alarm_intervals, detection_metrics, normal_metrics


class DataTests(unittest.TestCase):
    def test_windows_align_to_endpoints_without_future_data(self):
        x = np.arange(30).reshape(10,3)
        w, end = make_windows(x, window_size=4, stride=2)
        np.testing.assert_array_equal(end, [3,5,7,9])
        for row, index in zip(w,end): np.testing.assert_array_equal(row,x[index-3:index+1])
        self.assertEqual(make_windows(x,1)[0].shape, (10,1,3))

    def test_bad_window_sizes_and_nonfinite_series_rejected(self):
        for size in (0,11,1.5,True):
            with self.assertRaises(ValueError): make_windows(np.zeros((10,3)),size)
        with self.assertRaises(ValueError): make_windows(np.zeros((10,3)),2,0)
        with self.assertRaises(ValueError): make_windows(np.zeros((10,)),2)
        with self.assertRaises(ValueError): make_windows(np.full((10,3),np.nan),2)

    def test_training_roles_are_contiguous_disjoint_and_exhaustive(self):
        fit,cal,val = split_train(np.arange(100))
        np.testing.assert_array_equal(fit,np.arange(60))
        np.testing.assert_array_equal(cal,np.arange(60,80))
        np.testing.assert_array_equal(val,np.arange(80,100))
        with self.assertRaises(ValueError): split_train(np.arange(4))

    def test_smd_machine_and_shape_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for folder in ('train','test','test_label'): (root/folder).mkdir()
            np.savetxt(root/'train/machine-1-1.txt',np.zeros((20,38)),delimiter=',')
            np.savetxt(root/'test/machine-1-1.txt',np.ones((12,38)),delimiter=',')
            np.savetxt(root/'test_label/machine-1-1.txt',np.zeros(12),fmt='%d')
            train,test,labels=load_smd(root,'machine-1-1')
            self.assertEqual(train.shape,(20,38)); self.assertEqual(test.shape,(12,38))
            self.assertEqual(labels.dtype,np.bool_)
            for name in ('../machine-1-1','machine-4-1','machine-1-9'):
                with self.assertRaises(ValueError): load_smd(root,name)
            np.savetxt(root/'test_label/machine-1-1.txt',np.zeros(11),fmt='%d')
            with self.assertRaises(ValueError): load_smd(root,'machine-1-1')
            np.savetxt(root/'test_label/machine-1-1.txt',np.full(12,2),fmt='%d')
            with self.assertRaises(ValueError): load_smd(root,'machine-1-1')

    def test_load_smd_verifies_existing_manifest_before_reusing_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); machine='machine-1-1'
            arrays={'train':np.zeros((20,38)), 'test':np.ones((12,38)), 'test_label':np.zeros(12)}
            paths={group:root/group/f'{machine}.txt' for group in arrays}
            for group,path in paths.items():
                path.parent.mkdir()
                np.savetxt(path,arrays[group],delimiter=',',fmt='%d')
            train,test,labels=load_smd(root,machine)  # Manual local input without a manifest.
            np.testing.assert_array_equal(train,arrays['train'])
            np.testing.assert_array_equal(test,arrays['test'])
            np.testing.assert_array_equal(labels,arrays['test_label'])
            manifest={'source':SMD_SOURCE, 'files':{
                path.relative_to(root).as_posix():{'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in paths.values()}}
            manifest_path=root/'download_manifest.json'
            manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
            manifest_bytes=manifest_path.read_bytes()
            np.testing.assert_array_equal(load_smd(root,machine)[2],labels)
            for group in ('test_label','train','test'):
                with self.subTest(modified_file=group):
                    original=paths[group].read_bytes()
                    changed=arrays[group].copy(); changed.flat[0]=1-changed.flat[0]
                    np.savetxt(paths[group],changed,delimiter=',',fmt='%d')
                    with self.assertRaisesRegex(ValueError,f'{group}/{machine}'):
                        load_smd(root,machine)
                    self.assertEqual(manifest_path.read_bytes(),manifest_bytes)
                    paths[group].write_bytes(original)
            for source in (None,'unexpected-source'):
                with self.subTest(source=source):
                    invalid=manifest.copy()
                    if source is None: invalid.pop('source')
                    else: invalid['source']=source
                    manifest_path.write_text(json.dumps(invalid),encoding='utf-8')
                    with self.assertRaisesRegex(ValueError,'source'): load_smd(root,machine)
            for name in manifest['files']:
                for missing_hash in (False,True):
                    with self.subTest(file=name,missing_hash=missing_hash):
                        files=manifest['files'].copy()
                        if missing_hash: files[name]={}
                        else: files.pop(name)
                        manifest_path.write_text(json.dumps({'source':SMD_SOURCE,'files':files}),encoding='utf-8')
                        with self.assertRaisesRegex(ValueError,name): load_smd(root,machine)
            manifest_path.write_text(json.dumps({'source':SMD_SOURCE}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'unverified'): load_smd(root,machine)

    def test_download_manifest_and_no_silent_reuse_of_corruption(self):
        arrays = {'train':np.zeros((20,38)), 'test':np.ones((12,38)), 'test_label':np.zeros((12,1))}
        import io
        def response(url,timeout):
            group=url.split('/')[-2]; text=io.StringIO()
            if group in arrays: np.savetxt(text,arrays[group],delimiter=',')
            else: text.write('public source attribution')
            return io.BytesIO(text.getvalue().encode())
        with tempfile.TemporaryDirectory() as temp, patch('chronoguard.data.urlopen',side_effect=response):
            manifest=download_smd(temp,['machine-1-1'])
            self.assertEqual(len(manifest['files']),5)
            manifest2=download_smd(temp,['machine-1-1'])
            self.assertEqual(manifest,manifest2)
            (Path(temp)/'train/machine-1-1.txt').write_text('corrupt',encoding='utf-8')
            with self.assertRaises(ValueError): download_smd(temp,['machine-1-1'])


class EvaluationTests(unittest.TestCase):
    def test_half_open_alarm_intervals(self):
        self.assertEqual(alarm_intervals([0,1,1,0,1]),[(1,3),(4,5)])
        self.assertEqual(alarm_intervals([]),[])
        self.assertEqual(alarm_intervals([1,1]),[(0,2)])

    def test_strict_points_are_not_adjusted_to_cover_whole_events(self):
        out=detection_metrics([0,1,1,1,0,0,1,1],[0,0,1,0,1,0,0,0])
        self.assertEqual((out['tp'],out['fp'],out['fn'],out['tn']),(1,1,4,2))
        self.assertAlmostEqual(out['precision'],.5)
        self.assertAlmostEqual(out['recall'],.2)
        self.assertAlmostEqual(out['f1'],2/7)
        self.assertEqual(out['events_detected'],1)
        self.assertEqual(out['events_total'],2)
        self.assertEqual(out['detected_event_delays'],[1])
        self.assertEqual(out['false_alarm_intervals'],1)

    def test_no_anomalies_and_no_detections_are_explicit(self):
        out=detection_metrics([0,0,0],[0,0,0])
        self.assertIsNone(out['event_recall'])
        self.assertIsNone(out['mean_detected_event_delay'])
        self.assertEqual(out['f1'],0.)
        self.assertEqual(normal_metrics([0,1,1,0])['alarm_intervals'],1)

    def test_bad_labels_or_shapes_rejected(self):
        for labels,pred in (([0,2],[0,1]),([0,1],[0]),([0,np.nan],[0,0])):
            with self.assertRaises(ValueError): detection_metrics(labels,pred)
        with self.assertRaises(ValueError): alarm_intervals([0,2])


if __name__ == '__main__': unittest.main()
