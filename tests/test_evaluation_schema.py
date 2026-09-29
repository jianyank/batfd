import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from batfd.eval import metrics
from helpers import script


class EvaluationSchemaTests(unittest.TestCase):
    def ev(self, alarm, onset=3, persistence=1):
        score = np.zeros(6)
        if alarm is not None:
            score[alarm:] = 1
        return metrics.evaluate_pack(6, score, .5, persistence=persistence, onset_index=onset)

    def test_no_alarm_is_not_detected(self):
        ev = self.ev(None)
        self.assertFalse(ev.alarm_ever)
        self.assertFalse(ev.early_detected)

    def test_early_equal_and_late_alarm_are_distinct(self):
        for alarm, early in [(1, True), (3, False), (5, False)]:
            with self.subTest(alarm=alarm):
                ev = self.ev(alarm)
                self.assertTrue(ev.alarm_ever)
                self.assertEqual(ev.early_detected, early)
                self.assertEqual(ev.lead_days, 3 - alarm)

    def test_missing_or_invalid_onset_is_unknown_not_false(self):
        for onset in [None, -1, 6]:
            with self.subTest(onset=onset):
                ev = self.ev(1, onset=onset)
                self.assertTrue(ev.alarm_ever)
                self.assertIsNone(ev.early_detected)

    def test_persistence_uses_confirmation_not_first_exceedance(self):
        ev = self.ev(2, persistence=2)
        self.assertEqual(ev.alarm_index, 3)
        self.assertFalse(ev.early_detected)

    def test_onset_zero_has_no_preonset_denominator(self):
        ev = self.ev(0, onset=0)
        self.assertFalse(ev.early_detected)
        self.assertTrue(np.isnan(ev.trigger_rate_before_onset))

    def test_summary_keeps_distinct_numerators_and_labelled_denominator(self):
        evs = [self.ev(1), self.ev(3), self.ev(5), self.ev(None), self.ev(0, onset=None)]
        result = metrics.summarize(evs)
        self.assertEqual(result["n_alarm_ever"], 4)
        self.assertEqual(result["n_alarm_ever_labelled"], 3)
        self.assertEqual(result["n_early_detected"], 1)
        self.assertEqual(result["early_detection_rate"], .25)
        self.assertAlmostEqual(result["trigger_rate_before_onset"], 2 / 12)

    def test_legacy_property_warns_and_preserves_original_meaning(self):
        ev = self.ev(5)
        with self.assertWarns(DeprecationWarning):
            self.assertTrue(ev.detected)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(ev.far_per_window, ev.trigger_rate_before_onset)

    def test_tradeoff_uses_same_alarm_semantics(self):
        tradeoff = script("08_tradeoff")
        class Scores:
            def score(self, x):
                return SimpleNamespace(score=x[:, 0])
        train = np.zeros((8, 1))
        test = np.zeros((24, 1))
        for pack, alarm in enumerate([1, 3, 5, None]):
            if alarm is not None:
                test[pack * 6 + alarm: (pack + 1) * 6] = 1
        ids = np.repeat([6, 8, 9, 10], 6)
        labels = {pid: {"onset_point": 3, "n_train": 0} for pid in [6, 8, 9, 10]}
        result = tradeoff.evaluate_point(Scores(), train, test, ids, {"ids": ids}, labels,
                                        mode="train_novelty", q=.99, persistence=1)
        self.assertEqual(result["n_alarm_ever"], 3)
        self.assertEqual(result["n_early_detected"], 1)
        self.assertEqual(result["early_detection_rate"], .25)
        self.assertEqual([r["early_detected"] for r in result["per_pack"]], [True, False, False, False])
        self.assertNotIn("detected", result["per_pack"][0])

    def test_legacy_tradeoff_read_is_explicit_and_early(self):
        figures = script("10_figures")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tradeoff_old.csv"
            path.write_text("tag,lof_mode,q,persistence,trigger_rate_before_onset,detection_rate\n"
                            "ours_full,train_novelty,0.99,1,0.1,0.25\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                figures.load_tradeoff(Path(tmp))
            data = figures.load_tradeoff(Path(tmp), allow_legacy=True)
            self.assertEqual(data["ours_full"]["train_novelty"][(.99, 1)]["early_detection_rate"], .25)


if __name__ == "__main__":
    unittest.main()
