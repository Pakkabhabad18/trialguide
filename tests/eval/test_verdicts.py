"""T36: helpers of the H1 / H2 / McNemar computation, on hand-made data."""

import numpy as np
import pytest

from trialguide.eval import verdicts as v
from trialguide.schemas import InfoCategory, RuleCheck, RuleKind, RuleStatus


def test_aligned_requires_the_same_patients():
    nd = {"A": {"p1": 0.5, "p2": 0.7}, "B": {"p1": 0.6, "p2": 0.8}}
    ids, arrays = v.aligned(nd, ["A", "B"])
    assert ids == ["p1", "p2"] and arrays["B"].tolist() == [0.6, 0.8]
    with pytest.raises(ValueError):
        v.aligned({"A": {"p1": 0.5}, "B": {"p1": 0.6, "p2": 0.8}}, ["A", "B"])


def test_rule_outcomes_only_counts_rules_masking_made_b6_leave_open():
    ref = {"p": {"t|r0": ("met", "inclusion"), "t|r1": ("met", "inclusion"),
                 "t|r2": ("not_enough_info", "inclusion"), "t|r3": ("not_met", "exclusion")}}
    b6 = {"p": {"t|r0": "met", "t|r1": "not_enough_info", "t|r2": "not_enough_info", "t|r3": "not_enough_info"}}
    final = {"p": {"t|r1": "met", "t|r3": "met"}}                  # r1 recovered, r3 settled wrongly
    assert v.rule_outcomes(ref, b6, final) == {("p", "t|r1"): True, ("p", "t|r3"): False}


def _check(pid, rid, conf, kind=RuleKind.INCLUSION):
    return RuleCheck(patient_id=pid, trial_id="T", rule_id=rid, kind=kind, status=RuleStatus.MET,
                     category=InfoCategory.INFERRED, evidence_ids=[0], confidence=conf, n_samples=1)


def test_uncorrected_threshold_is_the_smallest_threshold_with_empirical_risk_below_alpha():
    # 10 patients, one rule each, all inferred MET; reference says NOT_MET for the three lowest-confidence ones
    checks = {f"p{i}": [_check(f"p{i}", "r", 0.5 + 0.05 * i)] for i in range(10)}
    ref = {(f"p{i}", "T", "r"): (RuleStatus.NOT_MET if i < 3 else RuleStatus.MET) for i in range(10)}
    # patients 0, 1, 2 (confidence 0.50, 0.55, 0.60) have a wrong inference; a harmful patient has loss 1
    assert v.uncorrected_threshold(checks, ref, alpha=0.5) == pytest.approx(0.5)      # accept all: 3/10 = 0.3
    assert v.uncorrected_threshold(checks, ref, alpha=0.2) == pytest.approx(0.55)     # from 0.55: 2/10 = 0.2 <= 0.2
    assert v.uncorrected_threshold(checks, ref, alpha=0.1) == pytest.approx(0.6)      # from 0.60: 1/10 <= 0.1


def test_h1_h2_verdict_wording_is_wired_through():
    from trialguide.eval.stats import BootResult
    assert v.h2_verdict(BootResult(0.039, 0.0125, 0.0741, 100, 0), v.ALPHA) == "inconclusive"
