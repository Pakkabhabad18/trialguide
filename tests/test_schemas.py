"""Tests for the shared data formats (T42). Run with: pytest tests/test_schemas.py"""

import json
from dataclasses import asdict

import pytest

from trialguide.schemas import (
    Certainty, ConversationState, DoctorAnswer, InfoCategory, Question, RuleCheck,
    RuleKind, RuleStatus, Split, VisibleNote, is_harmful,
)


def make_question(turn=1, patient_id="p1", fact_key="prior_chemotherapy"):
    # Small helper so each test can build a valid question in one line
    return Question(question_id=f"q{turn}", patient_id=patient_id, fact_key=fact_key,
                    text="Has she received chemotherapy before?", turn=turn)


# --- Harmful inference: wrong AND moves the patient toward eligibility ---

def test_inclusion_wrongly_met_is_harmful():
    assert is_harmful(RuleKind.INCLUSION, RuleStatus.MET, RuleStatus.NOT_MET)


def test_exclusion_wrongly_not_met_is_harmful():
    assert is_harmful(RuleKind.EXCLUSION, RuleStatus.NOT_MET, RuleStatus.MET)


def test_error_toward_exclusion_is_not_harmful():
    # Wrong, but it pushes the patient out of the trial: a possible false exclusion, not a harmful inference
    assert not is_harmful(RuleKind.INCLUSION, RuleStatus.NOT_MET, RuleStatus.MET)
    assert not is_harmful(RuleKind.EXCLUSION, RuleStatus.MET, RuleStatus.NOT_MET)


def test_correct_answer_is_never_harmful():
    assert not is_harmful(RuleKind.INCLUSION, RuleStatus.MET, RuleStatus.MET)


# --- Doctor answers (D10, D11) ---

def test_answer_without_quote_must_be_not_documented():
    with pytest.raises(ValueError):
        DoctorAnswer(question_id="q1", value="yes", certainty=Certainty.SURE, quote=None)


def test_not_documented_must_not_carry_quote():
    with pytest.raises(ValueError):
        DoctorAnswer(question_id="q1", value="", certainty=Certainty.NOT_DOCUMENTED, quote="some text")


def test_valid_answers():
    DoctorAnswer(question_id="q1", value="no", certainty=Certainty.SURE,
                 quote="She has never received chemotherapy.")
    DoctorAnswer(question_id="q2", value="", certainty=Certainty.NOT_DOCUMENTED)


# --- Conversation budget (D8) ---

def test_budget_of_three_is_enforced():
    state = ConversationState(patient_id="p1")
    for turn in (1, 2, 3):
        state.add_turn(make_question(turn, fact_key=f"f{turn}"),
                       DoctorAnswer(question_id=f"q{turn}", value="", certainty=Certainty.NOT_DOCUMENTED))
    assert state.questions_left() == 0
    with pytest.raises(RuntimeError):
        state.add_turn(make_question(4), DoctorAnswer(question_id="q4", value="", certainty=Certainty.NOT_DOCUMENTED))


def test_question_for_other_patient_is_refused():
    state = ConversationState(patient_id="p1")
    with pytest.raises(ValueError):
        state.add_turn(make_question(patient_id="p2"),
                       DoctorAnswer(question_id="q1", value="", certainty=Certainty.NOT_DOCUMENTED))


# --- Rule checks ---

def test_stated_rule_needs_evidence():
    with pytest.raises(ValueError):
        RuleCheck(patient_id="p1", trial_id="t1", rule_id="inc_1", kind=RuleKind.INCLUSION,
                  status=RuleStatus.MET, category=InfoCategory.STATED, evidence_ids=[])


def test_missing_rule_is_unresolved():
    check = RuleCheck(patient_id="p1", trial_id="t1", rule_id="exc_2", kind=RuleKind.EXCLUSION,
                      status=RuleStatus.NOT_ENOUGH_INFO, category=InfoCategory.MISSING, confidence=0.6)
    assert check.unresolved


def test_confidence_must_be_between_0_and_1():
    with pytest.raises(ValueError):
        RuleCheck(patient_id="p1", trial_id="t1", rule_id="inc_1", kind=RuleKind.INCLUSION,
                  status=RuleStatus.MET, category=InfoCategory.STATED, evidence_ids=[2], confidence=1.4)


# --- Logging: every object must turn into JSON for the experiment logs ---

def test_objects_serialise_to_json():
    note = VisibleNote(patient_id="p1", split=Split.DEV, sentences={0: "62-year-old woman.", 2: "Creatinine 1.9 mg/dL."})
    text = json.dumps(asdict(note))
    assert '"dev"' in text
    assert note.text().startswith("0. 62-year-old woman.")
