"""T17: ranking metrics, decision rates, bootstrap, recovery, McNemar and verdicts (hand-computed values)."""

import math
from pathlib import Path

import numpy as np
import pytest

from trialguide.eval import decisions as dec
from trialguide.eval import ranking as rk
from trialguide.eval import report, stats
from trialguide.schemas import Split

L3 = 1 / math.log2(3)       # discount of rank 2


# ---- NDCG@10 and P@10 ----

def test_ndcg_perfect_and_hand_computed():
    assert rk.ndcg_at_k([2, 1, 0]) == pytest.approx(1.0)
    # DCG = 0 + 1/log2(3) + 2/log2(4);  ideal = 2 + 1/log2(3)
    assert rk.ndcg_at_k([0, 1, 2]) == pytest.approx((L3 + 1.0) / (2 + L3))


def test_ndcg_undefined_without_positive_grades():
    assert rk.ndcg_at_k([0, 0, 0]) is None


def test_ndcg_only_counts_top_k_but_ideal_uses_all_candidates():
    gains = [0] * 10 + [2]                      # the one eligible trial is ranked 11th
    assert rk.ndcg_at_k(gains, 10) == 0.0
    assert rk.ndcg_at_k([2] + [0] * 10, 10) == pytest.approx(1.0)


def test_precision_at_10_is_graded():
    assert rk.precision_at_k([2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 2]) == pytest.approx(5 / 20)   # 11th ignored


def test_ranked_gains_orders_by_score_and_breaks_ties_by_id():
    grades = {"A": 0, "B": 2, "C": 1}
    assert rk.ranked_gains({"A": 0.1, "B": 0.9, "C": 0.5}, grades) == [2, 1, 0]
    assert rk.ranked_gains({"A": 1, "B": 1, "C": 1}, grades) == [0, 2, 1]      # A, B, C by id


def test_missing_score_is_an_error_not_a_silent_drop():
    with pytest.raises(ValueError):
        rk.ranked_gains({"A": 1.0}, {"A": 0, "B": 2})


def test_per_patient_metrics_skips_undefined_and_ignores_unlabelled_trials():
    gold = {"p1": {"A": 2, "B": 0}, "p2": {"C": 0, "D": 0}}
    rank = {"p1": {"A": 1.0, "B": 0.0, "Z": 5.0}, "p2": {"C": 1.0, "D": 0.0}}   # Z has no gold grade
    m = rk.per_patient_metrics(rank, gold)
    assert set(m) == {"p1"} and m["p1"]["ndcg"] == pytest.approx(1.0)


def test_same_candidates_check():
    rk.same_candidates({"a": {"p": {"t1": 1, "t2": 0}}, "b": {"p": {"t1": 0, "t2": 1}}})
    with pytest.raises(ValueError):
        rk.same_candidates({"a": {"p": {"t1": 1}}, "b": {"p": {"t1": 1, "t2": 0}}})
    with pytest.raises(ValueError):
        rk.same_candidates({"a": {"p": {"t1": 1}}, "b": {"q": {"t1": 1}}})


# ---- decision rates ----

def _decisions():
    gold = {"p1": {"A": 2, "B": 1, "C": 0}, "p2": {"D": 2, "E": 0}}
    pred = {"p1": {"A": "eligible", "B": "eligible", "C": "excluded"},
            "p2": {"D": "excluded", "E": "unknown"}}
    return gold, pred


def test_decision_counts_and_rates_by_hand():
    gold, pred = _decisions()
    c = dec.decision_counts(pred, gold, ["p1", "p2"])
    assert c["n_eligible"].sum() == 2 and c["n_ineligible"].sum() == 3
    assert c["false_eligible"].sum() == 1 and c["false_exclusion"].sum() == 1
    rep = dec.decision_report(c, n_boot=200)
    assert rep["false_eligible_rate"].estimate == pytest.approx(1 / 3)      # B shown eligible, 3 ineligible pairs
    assert rep["false_exclusion_rate"].estimate == pytest.approx(1 / 2)     # D ruled out, 2 eligible pairs
    assert rep["precision_eligible"].estimate == pytest.approx(1 / 2)
    assert rep["recall_eligible"].estimate == pytest.approx(1 / 2)
    assert rep["unknown_rate"].estimate == pytest.approx(1 / 5)


def test_abstaining_is_neither_error():
    gold = {"p": {"A": 2, "B": 0}}
    c = dec.decision_counts({"p": {"A": "unknown", "B": "unknown"}}, gold, ["p"])
    assert c["false_eligible"].sum() == 0 and c["false_exclusion"].sum() == 0 and c["unknown"].sum() == 2


def test_bad_decision_label_is_rejected():
    with pytest.raises(ValueError):
        dec.decision_counts({"p": {"A": "maybe"}}, {"p": {"A": 2}}, ["p"])


# ---- bootstrap ----

def test_bootstrap_is_reproducible_and_constant_data_has_zero_width():
    x = np.full(30, 0.5)
    a = stats.paired_bootstrap({"x": x}, stats.mean_of("x"), n_boot=500, seed=3)
    assert (a.estimate, a.low, a.high) == (0.5, 0.5, 0.5)
    y = np.random.default_rng(0).random(40)
    r1 = stats.paired_bootstrap({"x": y}, stats.mean_of("x"), n_boot=500, seed=7)
    r2 = stats.paired_bootstrap({"x": y}, stats.mean_of("x"), n_boot=500, seed=7)
    assert r1 == r2


def test_bootstrap_interval_has_the_right_width_for_a_mean():
    y = np.random.default_rng(1).normal(size=400)
    r = stats.paired_bootstrap({"x": y}, stats.mean_of("x"), n_boot=4000, seed=0)
    assert r.low < y.mean() < r.high
    assert (r.high - r.low) == pytest.approx(2 * 1.96 * y.std(ddof=1) / 20, rel=0.15)


def test_bootstrap_rejects_misaligned_arrays():
    with pytest.raises(ValueError):
        stats.paired_bootstrap({"a": [1, 2], "b": [1, 2, 3]}, stats.mean_of("a"))


def test_ratio_of_sums_counts_undefined_resamples():
    r = stats.paired_bootstrap({"n": [1, 0, 0], "d": [1, 0, 0]}, stats.ratio_of_sums("n", "d"), n_boot=300, seed=0)
    assert r.estimate == 1.0 and r.n_undefined > 0          # resamples without patient 0 have den = 0


# ---- recovery ----

def test_recovery_endpoints_and_midpoint():
    b1, b6 = np.array([0.9, 0.8, 1.0]), np.array([0.3, 0.2, 0.4])
    assert stats.recovery(b1, b1, b6) == pytest.approx(1.0)
    assert stats.recovery(b6, b1, b6) == pytest.approx(0.0)
    assert stats.recovery((b1 + b6) / 2, b1, b6) == pytest.approx(0.5)


def test_recovery_is_a_ratio_of_averages_not_an_average_of_ratios():
    # Patient 2 has no gap (B1 == B6); a per-patient ratio would be 0/0 there.
    b1, b6, s = np.array([1.0, 0.5]), np.array([0.0, 0.5]), np.array([0.5, 0.5])
    assert stats.recovery(s, b1, b6) == pytest.approx(0.5)      # mean gain 0.25 / mean gap 0.5


def test_recovery_not_reported_when_gap_is_tiny():
    b1, b6 = np.array([0.50, 0.51]), np.array([0.50, 0.50])
    assert stats.recovery(b1, b1, b6) is None


def test_recovery_bootstrap_and_difference():
    rng = np.random.default_rng(0)
    b6 = rng.random(60) * 0.3
    b1 = b6 + 0.4
    arr = {"b1": b1, "b6": b6, "b4": b6 + 0.1, "b7": b6 + 0.3}     # B4 recovers 25 points, B7 75
    r = stats.paired_bootstrap(arr, stats.recovery_stat("b7", "b1", "b6"), n_boot=300)
    assert r.estimate == pytest.approx(0.75) and r.low == pytest.approx(0.75)
    d = stats.paired_bootstrap(arr, stats.recovery_difference_stat("b7", "b4", "b1", "b6"), n_boot=300)
    assert d.estimate == pytest.approx(0.5)


# ---- McNemar ----

def test_mcnemar_exact_values():
    a = np.array([True] * 10 + [True] * 5)
    b = np.array([False] * 10 + [True] * 5)
    r = stats.mcnemar(a, b)
    assert (r.a_only, r.b_only) == (10, 0)
    assert r.p_value == pytest.approx(2 * 0.5 ** 10)
    assert stats.mcnemar([True, False], [True, False]).p_value == 1.0          # no discordant pairs
    assert stats.mcnemar([True, False, True, False], [False, True, True, False]).p_value == 1.0   # 1 vs 1
    with pytest.raises(ValueError):
        stats.mcnemar([True], [True, False])


# ---- verdicts, as worded in the proposal ----

def _res(est, lo, hi):
    return stats.BootResult(est, lo, hi, 1000, 0)


def test_h1_verdicts():
    assert stats.h1_verdict(_res(0.14, 0.05, 0.22), True) == "supported"
    assert stats.h1_verdict(_res(0.14, 0.05, 0.22), False) == "inconclusive"       # false-eligible got worse
    assert stats.h1_verdict(_res(0.12, -0.02, 0.25), True) == "inconclusive"       # CI includes zero
    assert stats.h1_verdict(_res(0.03, -0.02, 0.08), True) == "not supported"      # upper bound below 10 points
    assert stats.h1_verdict(_res(0.08, 0.02, 0.15), True) == "inconclusive"        # estimate under the target


def test_h2_verdicts():
    assert stats.h2_verdict(_res(0.03, 0.01, 0.05)) == "supported"
    assert stats.h2_verdict(_res(0.08, 0.06, 0.11)) == "contradicted"
    assert stats.h2_verdict(_res(0.05, 0.03, 0.08)) == "inconclusive"


# ---- report ----

def test_ranking_report_end_to_end():
    gold = {f"p{i}": {"A": 2, "B": 1, "C": 0, "D": 0} for i in range(20)}
    best = {p: {"A": 4, "B": 3, "C": 2, "D": 1} for p in gold}
    worst = {p: {"A": 1, "B": 2, "C": 3, "D": 4} for p in gold}
    mid = {p: {"A": 4, "B": 1, "C": 3, "D": 2} for p in gold}
    rep = report.ranking_report({"B1": best, "B6": worst, "B7": mid}, gold, n_boot=200)
    assert rep["n_patients"] == 20
    assert rep["systems"]["B1"]["ndcg@10"].estimate == pytest.approx(1.0)
    assert "recovery" not in rep["systems"]["B1"]
    assert 0.0 < rep["systems"]["B7"]["recovery"].estimate < 1.0


# ---- gold grades ----

def test_test_labels_are_sealed():
    with pytest.raises(PermissionError):
        rk.load_gold_grades(Path("data/raw"), Split.TEST)


RAW = Path("data/raw")


@pytest.mark.skipif(not (RAW / "annotations").exists(), reason="run scripts/download_data.sh first")
def test_dev_gold_grades_load():
    gold = rk.load_gold_grades(RAW, Split.DEV)
    assert len(gold) == 37 and all(gold.values())
    assert {g for d in gold.values() for g in d.values()} <= {0, 1, 2}
