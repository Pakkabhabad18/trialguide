"""Live demo: the two-model session, with word-matching stand-ins instead of the models."""

from trialguide.demo.live import TwoModelSession, keyword_checker, note_from_facts, note_from_text
from trialguide.pipeline import ACCEPTED
from trialguide.schemas import Certainty, RuleKind, TrialRule

RULES = [TrialRule("T1", "inc_0", RuleKind.INCLUSION, "Histologically confirmed breast carcinoma"),
         TrialRule("T1", "inc_1", RuleKind.INCLUSION, "Signed informed consent"),
         TrialRule("T1", "exc_0", RuleKind.EXCLUSION, "Serum creatinine above normal range"),
         TrialRule("T2", "inc_0", RuleKind.INCLUSION, "Prior chemotherapy for metastatic disease"),
         TrialRule("T2", "exc_0", RuleKind.EXCLUSION, "Pregnant or nursing women")]
NOTE = "A 58-year-old woman has histologically confirmed breast carcinoma. She takes metformin."


def session(budget=3):
    s = TwoModelSession(note_from_text(NOTE), RULES, keyword_checker(), keyword_checker(settle_open=True),
                        threshold=0.766, budget=budget)
    s.start()
    return s


def test_second_model_settles_only_rules_the_first_left_open():
    s = session()
    acts = s.action_of()
    assert ("T1", "inc_1") in s.by_second                     # consent: only the second model settles it
    assert acts[("T1", "inc_1")].action in ACCEPTED
    assert "second model" in acts[("T1", "inc_1")].reason
    assert ("T1", "inc_0") not in s.by_second                 # stated in the note: the first model's
    assert s.counts("T1")["second"] == 1


def test_doctor_is_asked_only_about_rules_neither_model_settles():
    s = session()
    if s.question is not None:                     # the selector may also stop at once (no question can help)
        assert not set(s.question.rule_refs) & s.by_second
    open_refs = {r for r, a in s.action_of().items() if a.action not in ACCEPTED}
    assert not open_refs & s.by_second


def test_answer_and_added_information_update_the_note():
    s = session()
    n = len(s.note.sentences)
    t = s.fork()
    if t.question is not None:
        t.answer("no", Certainty.SURE)
    m = len(t.note.sentences)
    t.add_info("Serum creatinine is within the normal range. She is not pregnant.")
    assert len(t.note.sentences) == m + 2 and len(s.note.sentences) == n    # the fork left s alone
    assert t.turns[-1].question == "You added information"


def test_note_from_facts():
    note = note_from_facts("54", "Female", "type 2 diabetes")
    assert list(note.sentences.values()) == ["The patient is a 54-year-old female.", "Main diagnosis: type 2 diabetes."]
