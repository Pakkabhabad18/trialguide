"""Checks the TrialGPT ranking metrics (paper Eqs. 29-32) on hand-made rankings."""

import math

from trialguide.baselines.trialgpt.evaluate import dcg, ndcg, precision


def test_perfect_ranking_has_ndcg_one():
	assert ndcg([2, 2, 1, 0, 0]) == 1.0


def test_reversed_ranking_is_worse_than_perfect():
	assert ndcg([0, 0, 1, 2, 2]) < ndcg([2, 2, 1, 0, 0])


def test_dcg_matches_hand_computation():
	# gains at ranks 1..3 are divided by log2(2), log2(3), log2(4)
	assert math.isclose(dcg([2, 1, 0]), 2 / 1 + 1 / math.log2(3) + 0)


def test_only_top_10_counts():
	assert dcg([0] * 10 + [2]) == 0


def test_no_relevant_trial_gives_none():
	# such patients are skipped instead of counted as 0 or 1
	assert ndcg([0, 0, 0]) is None


def test_precision_is_graded():
	# P@10 = sum of gains in the top 10 / (max gain 2 * 10)
	assert precision([2] * 10) == 1.0
	assert precision([1] * 10) == 0.5
	assert precision([2, 2]) == 0.2
