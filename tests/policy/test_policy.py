"""Tests for the infer / ask / abstain policy (T19). Run with: pytest tests/policy"""

import pytest

from trialguide.policy.policy import (
    PolicyConfig, decide, decide_all, finalize, needs_follow_up,
    question_candidates, system_config,
)
from trialguide.schemas import (
    ActionType, Certainty, DoctorAnswer, InfoCategory, Question, RuleCheck,
    RuleKind, RuleStatus,
)

OURS = system_config("B7", threshold=0.8)  # lambda-hat is a placeholder until calibration (T30)


def check(category, status, kind=RuleKind.INCLUSION, confidence=1.0, evidence=(2,),
          fact_key=None, trial="t1", rule="inc_1"):
    # Builds one checker output; evidence is only needed for stated / inferred rules
    needs_evidence = category in (InfoCategory.STATED, InfoCategory.INFERRED)
    return RuleCheck(patient_id="p1", trial_id=trial, rule_id=rule, kind=kind, status=status,
                     category=category, evidence_ids=list(evidence) if needs_evidence else [],
                     confidence=confidence, fact_key=fact_key)


# --- 1. Stated and not applicable: use them ---

def test_stated_rule_is_used():
    assert decide(check(InfoCategory.STATED, RuleStatus.MET), OURS).action == ActionType.USE_STATED


def test_not_applicable_is_settled():
    c = check(InfoCategory.STATED, RuleStatus.NOT_APPLICABLE)
    assert decide(c, OURS).action == ActionType.USE_STATED


# --- 2. Inference toward eligibility: only above the threshold ---

def test_confident_inference_toward_eligibility_is_accepted():
    c = check(InfoCategory.INFERRED, RuleStatus.MET, confidence=0.8)  # inclusion met
    assert decide(c, OURS).action == ActionType.ACCEPT_INFERENCE


def test_unsure_inference_toward_eligibility_becomes_a_question():
    c = check(InfoCategory.INFERRED, RuleStatus.MET, confidence=0.6, fact_key="prior_chemo")
    assert decide(c, OURS).action == ActionType.ASK


def test_exclusion_wrongly_not_met_is_the_harmful_direction():
    # exclusion NOT_MET also favours eligibility, so the threshold applies
    c = check(InfoCategory.INFERRED, RuleStatus.NOT_MET, kind=RuleKind.EXCLUSION,
              confidence=0.6, fact_key="hepatitis_b")
    assert decide(c, OURS).action == ActionType.ASK


# --- 3. Inference away from eligibility: accepted (cannot be harmful, D15) ---

def test_inference_toward_exclusion_is_accepted_even_at_low_confidence():
    c = check(InfoCategory.INFERRED, RuleStatus.NOT_MET, confidence=0.4)  # inclusion not met
    assert decide(c, OURS).action == ActionType.ACCEPT_INFERENCE


# --- 4. Assumption / missing: never used as a fact ---

def test_assumption_with_a_fact_becomes_a_question():
    c = check(InfoCategory.ASSUMPTION, RuleStatus.NOT_ENOUGH_INFO, fact_key="prior_chemo")
    assert decide(c, OURS).action == ActionType.ASK


def test_missing_without_a_fact_key_is_abstained():
    c = check(InfoCategory.MISSING, RuleStatus.NOT_ENOUGH_INFO, fact_key=None)
    assert decide(c, OURS).action == ActionType.ABSTAIN


def test_fact_already_asked_is_not_asked_again():
    c = check(InfoCategory.MISSING, RuleStatus.NOT_ENOUGH_INFO, fact_key="prior_chemo")
    assert decide(c, OURS, already_asked={"prior_chemo"}).action == ActionType.ABSTAIN


# --- 5. Doctor answers by certainty (D11) ---

def test_probably_answer_is_treated_as_an_inference():
    # the checker calls it stated because it cites sentence 100, but that is a "probably" answer
    c = check(InfoCategory.STATED, RuleStatus.MET, confidence=0.6, evidence=(100,), fact_key="prior_chemo")
    action = decide(c, OURS, doctor_certainty={100: Certainty.PROBABLY}, already_asked={"prior_chemo"})
    assert action.action == ActionType.ABSTAIN  # below threshold, and the fact was already asked


def test_sure_answer_is_used_as_stated():
    c = check(InfoCategory.STATED, RuleStatus.MET, evidence=(100,))
    assert decide(c, OURS, doctor_certainty={100: Certainty.SURE}).action == ActionType.USE_STATED


def test_unsure_answer_is_not_used_as_evidence():
    c = check(InfoCategory.STATED, RuleStatus.MET, evidence=(100,), fact_key="prior_chemo")
    action = decide(c, OURS, doctor_certainty={100: Certainty.UNSURE})
    assert action.action == ActionType.ASK


def test_not_applicable_resting_on_an_unsure_answer_is_not_settled():
    # "not applicable" that only rests on an unsure doctor answer must not settle the rule
    c = check(InfoCategory.STATED, RuleStatus.NOT_APPLICABLE, evidence=(100,), fact_key="sex")
    assert decide(c, OURS, doctor_certainty={100: Certainty.UNSURE}).action == ActionType.ASK


def test_one_follow_up_after_unsure_but_not_two():
    q = Question(question_id="q1", patient_id="p1", fact_key="prior_chemo", text="...", turn=1)
    unsure = DoctorAnswer(question_id="q1", value="maybe", certainty=Certainty.UNSURE, quote="Some sentence.")
    assert needs_follow_up(unsure, q)
    follow = Question(question_id="q2", patient_id="p1", fact_key="prior_chemo", text="...", turn=2, is_followup=True)
    assert not needs_follow_up(unsure, follow)


# --- 6. Grouping questions and ending the conversation ---

def test_same_fact_in_two_trials_is_one_question_candidate():
    checks = [check(InfoCategory.MISSING, RuleStatus.NOT_ENOUGH_INFO, fact_key="prior_chemo", trial="t1"),
              check(InfoCategory.MISSING, RuleStatus.NOT_ENOUGH_INFO, fact_key="prior_chemo", trial="t2")]
    groups = question_candidates(decide_all(checks, OURS), checks)
    assert groups == {"prior_chemo": [("t1", "inc_1"), ("t2", "inc_1")]}


def test_unasked_questions_become_abstentions_at_the_end():
    c = check(InfoCategory.MISSING, RuleStatus.NOT_ENOUGH_INFO, fact_key="prior_chemo")
    final = finalize([decide(c, OURS)])
    assert final[0].action == ActionType.ABSTAIN


# --- 7. The same code gives the baselines and ablations ---

def test_b6_never_infers_or_asks():
    b6 = system_config("B6")
    c = check(InfoCategory.INFERRED, RuleStatus.MET, confidence=1.0, fact_key="prior_chemo")
    assert decide(c, b6).action == ActionType.ABSTAIN


def test_b2_accepts_any_inference():
    c = check(InfoCategory.INFERRED, RuleStatus.MET, confidence=0.2)
    assert decide(c, system_config("B2")).action == ActionType.ACCEPT_INFERENCE


def test_b5_uses_a_fixed_0_9_threshold_and_never_asks():
    c = check(InfoCategory.INFERRED, RuleStatus.MET, confidence=0.85, fact_key="prior_chemo")
    assert decide(c, system_config("B5")).action == ActionType.ABSTAIN


def test_a1_asks_instead_of_inferring():
    c = check(InfoCategory.INFERRED, RuleStatus.MET, confidence=1.0, fact_key="prior_chemo")
    assert decide(c, system_config("A1")).action == ActionType.ASK


def test_b7_refuses_to_run_without_a_calibrated_threshold():
    with pytest.raises(ValueError):
        system_config("B7")


def test_threshold_must_be_a_probability():
    with pytest.raises(ValueError):
        PolicyConfig("bad", allow_inference=True, allow_asking=True, threshold=1.5)
