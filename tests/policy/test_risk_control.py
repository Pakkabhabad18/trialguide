"""Tests for conformal risk control (T30). Every expected number is worked out in the comments."""

import math

import pytest

from trialguide.policy.policy import PolicyConfig, decide, system_config
from trialguide.policy.risk_control import (
    ACCEPT_NONE, bound, calibrate, counts_at, empirical_risk, is_unsettled, patient_counts,
    risk_report,
)
from trialguide.schemas import ActionType, InfoCategory, RuleCheck, RuleKind, RuleStatus

INC, EXC = RuleKind.INCLUSION, RuleKind.EXCLUSION
MET, NOT_MET, NEI = RuleStatus.MET, RuleStatus.NOT_MET, RuleStatus.NOT_ENOUGH_INFO


def chk(pid, rule, category, status, kind=INC, conf=1.0):
    # One checker output; stated / inferred rules need evidence ids
    ev = [0] if category in (InfoCategory.STATED, InfoCategory.INFERRED) else []
    return RuleCheck(patient_id=pid, trial_id="t1", rule_id=rule, kind=kind, status=status,
                     category=category, evidence_ids=ev, confidence=conf)


def inferred(pid, rule, status, conf, kind=INC):
    return chk(pid, rule, InfoCategory.INFERRED, status, kind, conf)


def missing(pid, rule):
    return chk(pid, rule, InfoCategory.MISSING, NEI)


# ---- which rules count ----

def test_stated_rules_are_settled_everything_else_is_not():
    assert not is_unsettled(chk("p", "r", InfoCategory.STATED, MET))
    assert is_unsettled(inferred("p", "r", MET, 0.9))
    assert is_unsettled(missing("p", "r"))


# ---- one patient ----

def one_patient():
    # 4 unsettled rules with a reference:
    #   r1 inclusion inferred MET at 0.9, reference NOT_MET -> harmful if accepted
    #   r2 inclusion inferred MET at 0.6, reference NEI     -> harmful if accepted
    #   r3 inclusion inferred MET at 0.95, reference MET    -> correct
    #   r4 missing                                          -> never used
    # plus r5 exclusion inferred MET (toward exclusion), wrong but never harmful,
    # plus r6 stated (settled), and r7 inferred with no reference (skipped)
    checks = [inferred("p", "r1", MET, 0.9), inferred("p", "r2", MET, 0.6),
              inferred("p", "r3", MET, 0.95), missing("p", "r4"),
              inferred("p", "r5", MET, 0.3, kind=EXC),
              chk("p", "r6", InfoCategory.STATED, MET), inferred("p", "r7", MET, 0.99)]
    ref = {("p", "t1", r): s for r, s in
           [("r1", NOT_MET), ("r2", NEI), ("r3", MET), ("r4", MET), ("r5", NOT_MET), ("r6", MET)]}
    return checks, ref


def test_loss_counts_harmful_permissive_inferences_over_unsettled_rules():
    checks, ref = one_patient()
    # unsettled with a reference: r1, r2, r3, r4, r5 = 5 rules
    low = patient_counts(checks, ref, 0.0, "p")
    assert low.unsettled == 5 and low.harmful == 2 and low.loss == pytest.approx(2 / 5)
    assert low.accepted == 4 and low.accepted_correct == 1      # r1, r2, r3, r5; only r3 correct
    mid = patient_counts(checks, ref, 0.7, "p")                  # r2 (0.6) now rejected
    assert mid.harmful == 1 and mid.loss == pytest.approx(1 / 5)
    high = patient_counts(checks, ref, 0.92, "p")                # only r3 (0.95) above
    assert high.harmful == 0 and high.accepted == 2              # r3, plus r5 (toward exclusion)


def test_loss_never_increases_with_the_threshold():
    checks, ref = one_patient()
    losses = [patient_counts(checks, ref, t, "p").loss for t in (0, 0.5, 0.6, 0.61, 0.9, 0.95, 1.0, ACCEPT_NONE)]
    assert losses == sorted(losses, reverse=True)
    assert losses[-1] == 0.0


# ---- calibration ----

def population(n_harmful, n_total=20, rules=10):
    # n_total patients, each with `rules` unsettled rules.
    # The first n_harmful patients have one harmful inference at confidence 0.7;
    # every patient has one correct permissive inference at 0.8.
    checks, ref = {}, {}
    for i in range(n_total):
        pid = f"p{i}"
        cs = [inferred(pid, "ok", MET, 0.8)]
        ref[(pid, "t1", "ok")] = MET
        if i < n_harmful:
            cs.append(inferred(pid, "bad", MET, 0.7))
            ref[(pid, "t1", "bad")] = NOT_MET
        while len(cs) < rules:
            r = f"m{len(cs)}"
            cs.append(missing(pid, r))
            ref[(pid, "t1", r)] = NEI
        checks[pid] = cs
    return checks, ref


def test_bound_formula():
    # n=20, R=0.075: 20/21*0.075 + 1/21 = 0.1190...
    assert bound(0.075, 20) == pytest.approx(20 / 21 * 0.075 + 1 / 21)


def test_lambda_hat_is_the_smallest_threshold_meeting_the_bound():
    # alpha=0.10, n=20: need R <= (0.10*21 - 1)/20 = 0.055.
    # 15 harmful patients with loss 0.1 each: R(0.7) = 1.5/20 = 0.075 -> fails.
    # At 0.8 the 0.7 inferences are rejected: R = 0 -> passes, so lambda_hat = 0.8.
    checks, ref = population(15)
    cal = calibrate(checks, ref, alpha=0.10)
    assert cal.certifiable and cal.lambda_hat == pytest.approx(0.8) and cal.risk_at_lambda == 0.0
    assert cal.accepts_any


def test_few_harmful_patients_keep_the_lowest_threshold():
    # 5 harmful patients: R(0.7) = 0.5/20 = 0.025 <= 0.055 -> lambda_hat = 0.7
    checks, ref = population(5)
    assert calibrate(checks, ref, alpha=0.10).lambda_hat == pytest.approx(0.7)


def test_target_below_one_over_n_plus_one_cannot_be_certified():
    # n=20, alpha=0.04 < 1/21 = 0.0476: even accepting nothing fails the bound
    checks, ref = population(0)
    cal = calibrate(checks, ref, alpha=0.04)
    assert not cal.certifiable and cal.lambda_hat == ACCEPT_NONE and not cal.accepts_any
    assert "cannot be certified" in cal.summary()


def test_calibration_ignores_rules_without_a_reference():
    checks, ref = population(15)
    # drop the references of the harmful rules: they can no longer be judged
    ref = {k: v for k, v in ref.items() if k[2] != "bad"}
    assert calibrate(checks, ref, alpha=0.10).lambda_hat == pytest.approx(0.8)  # only 0.8 is a candidate


def test_bad_alpha_is_rejected():
    checks, ref = population(0)
    with pytest.raises(ValueError):
        calibrate(checks, ref, alpha=0.0)


# ---- the policy with the calibrated threshold ----

def test_policy_accepts_inf_threshold_and_then_rejects_every_permissive_inference():
    cfg = system_config("B7", threshold=math.inf)
    c = inferred("p", "r", MET, 1.0)
    c.fact_key = "x"
    assert decide(c, cfg).action == ActionType.ASK
    with pytest.raises(ValueError):
        PolicyConfig("bad", allow_inference=True, allow_asking=True, threshold=1.5)


def test_policy_and_risk_control_agree_on_what_is_accepted():
    # every permissive inference the policy accepts at lambda is one the loss counts at lambda
    checks, ref = one_patient()
    cfg = system_config("B7", threshold=0.7)
    accepted = {c.rule_id for c in checks if decide(c, cfg).action == ActionType.ACCEPT_INFERENCE}
    assert accepted == {"r1", "r3", "r5", "r7"}     # r7 has no reference, so the loss skips it
    assert patient_counts(checks, ref, 0.7, "p").accepted == 3   # r1, r3, r5


# ---- report ----

def test_risk_report_on_a_population():
    checks, ref = population(5)
    rep = risk_report(counts_at(checks, ref, 0.7), n_boot=200)
    # 20 patients x 10 rules; 25 accepted (20 correct + 5 harmful); mean loss 0.5/20
    assert rep["unsettled_rules"] == 200 and rep["accepted_inferences"] == 25 and rep["harmful_inferences"] == 5
    assert rep["harmful_rate"].estimate == pytest.approx(0.025)
    assert rep["useful_coverage"].estimate == pytest.approx(20 / 200)
    assert empirical_risk(counts_at(checks, ref, 0.7)) == pytest.approx(0.025)
