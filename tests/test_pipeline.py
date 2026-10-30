"""End-to-end runner, with a scripted checker and doctor instead of models."""

from trialguide.pipeline import effective_checks, rules_of_trial, run_patient
from trialguide.policy.policy import PolicyConfig, system_config
from trialguide.schemas import (
    Certainty, DoctorAnswer, InfoCategory, RuleCheck, RuleKind, RuleStatus, TrialRule, VisibleNote,
)

NOTE = VisibleNote("p1", "dev", {0: "A 60-year-old man with lung cancer.", 1: "He takes metformin."})
RULES = [TrialRule("T1", "inc_0", RuleKind.INCLUSION, "Lung cancer"),
         TrialRule("T1", "exc_0", RuleKind.EXCLUSION, "Current smoker"),
         TrialRule("T2", "inc_0", RuleKind.INCLUSION, "Diabetes"),
         TrialRule("T2", "exc_0", RuleKind.EXCLUSION, "Current smoker")]


def fake_check(note, rule):
    """Lung cancer is stated; diabetes is inferred from metformin; smoking is missing until the
    doctor's answer is in the note."""
    def rc(status, category, ev=(), conf=1.0, key=None):
        return RuleCheck(note.patient_id, rule.trial_id, rule.rule_id, rule.kind, status, category,
                         list(ev), confidence=conf, fact_key=key)
    if rule.text == "Lung cancer":
        return rc(RuleStatus.MET, InfoCategory.STATED, [0])
    if rule.text == "Diabetes":
        return rc(RuleStatus.MET, InfoCategory.INFERRED, [1], conf=0.6)
    answered = [i for i, s in note.sentences.items() if "smok" in s.lower()]
    if answered:
        return rc(RuleStatus.NOT_MET, InfoCategory.STATED, answered)
    return rc(RuleStatus.NOT_ENOUGH_INFO, InfoCategory.MISSING, key="smoking_status")


class FakeDoctor:
    def __init__(self):
        self.asked = []

    def answer(self, patient_id, question):
        self.asked.append(question.fact_key)
        return DoctorAnswer(question.question_id, "no", Certainty.SURE, quote="He has never smoked.")


def test_b6_never_asks_and_treats_inference_as_unresolved():
    doc = FakeDoctor()
    r = run_patient(NOTE, RULES, system_config("B6"), fake_check, doctor=doc)
    assert doc.asked == [] and r.turns == []
    # T1: 1/1 inclusion met; T2: diabetes inference not allowed -> 0/1
    assert r.scores == {"T1": 1.0, "T2": 0.0}


def test_system_that_asks_settles_the_shared_fact_with_one_question():
    doc = FakeDoctor()
    cfg = PolicyConfig("test", allow_inference=True, allow_asking=True, threshold=0.5)
    r = run_patient(NOTE, RULES, cfg, fake_check, doctor=doc, strategy="fixed")
    assert doc.asked == ["smoking"]                         # one question settles both trials' rule
    assert len(r.turns) == 1 and r.n_checks == 4 + 2        # the two smoking rules re-checked
    assert r.scores == {"T1": 1.0, "T2": 1.0}               # diabetes inference accepted (0.6 >= 0.5)


def test_inference_below_threshold_counts_as_not_enough_information():
    cfg = PolicyConfig("strict", allow_inference=True, allow_asking=False, threshold=0.9)
    r = run_patient(NOTE, RULES, cfg, fake_check)
    assert r.scores["T2"] == 0.0


def test_effective_checks_keeps_only_accepted_rules():
    from trialguide.policy.policy import decide_all
    checks = [fake_check(NOTE, rule) for rule in RULES]
    eff = effective_checks(checks, decide_all(checks, system_config("B6")))
    assert [c.status for c in eff] == [RuleStatus.MET, RuleStatus.NOT_ENOUGH_INFO,
                                       RuleStatus.NOT_ENOUGH_INFO, RuleStatus.NOT_ENOUGH_INFO]


def test_rules_of_trial_numbers_inclusion_and_exclusion_separately():
    trial = {"NCTID": "NCT1", "inclusion_criteria": "Inclusion Criteria:\n\nAge >= 18\n\nLung cancer",
             "exclusion_criteria": "Exclusion Criteria:\n\nPregnancy"}
    rules = rules_of_trial(trial)
    assert [(r.rule_id, r.kind, r.text) for r in rules] == [
        ("inc_0", RuleKind.INCLUSION, "Age >= 18"), ("inc_1", RuleKind.INCLUSION, "Lung cancer"),
        ("exc_0", RuleKind.EXCLUSION, "Pregnancy")]
