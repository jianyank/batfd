"""Tests for evidence-only episodic versus continuous sequence diagnostics."""
import unittest
import numpy as np


class SequenceDiagnosticTests(unittest.TestCase):
    def test_longest_confirmation_counts_contiguous_windows(self):
        from diagnose_sequences import longest_run
        self.assertEqual(longest_run([False,True,True,False,True]),2)
        self.assertEqual(longest_run([False,False]),0)
        self.assertEqual(longest_run([True,True,True]),3)

    def test_episode_comparison_uses_saved_original_window_indices(self):
        from diagnose_sequences import compare_episode
        scores=np.array([2.,2.,2.,2.,2.,2.,0.])
        indices=np.arange(100,107)
        episode=np.array([2.,2.,0.])
        result=compare_episode(scores,indices,episode,[104,105,106],1.)
        self.assertEqual(result['max_absolute_score_difference'],0.)
        self.assertEqual(result['episode_confirmed_windows'],0)
        self.assertEqual(result['continuous_confirmed_windows'],2)
        self.assertEqual(result['confirmation_disagreement_windows'],2)

    def test_reference_reset_score_difference_is_reported(self):
        from diagnose_sequences import compare_episode
        result=compare_episode(np.ones(8)*2,np.arange(8),np.zeros(8),list(range(8)),1.)
        self.assertEqual(result['max_absolute_score_difference'],2.)
        self.assertEqual(result['continuous_confirmed_windows'],4)
        self.assertEqual(result['episode_confirmed_windows'],0)

    def test_missing_or_repeated_window_indices_rejected(self):
        from diagnose_sequences import compare_episode
        with self.assertRaises(ValueError): compare_episode(np.zeros(5),np.arange(5),np.zeros(2),[3,8],1.)
        with self.assertRaises(ValueError): compare_episode(np.zeros(5),np.array([0,1,1,3,4]),np.zeros(2),[0,1],1.)


if __name__=='__main__': unittest.main()
