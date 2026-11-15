"""Post-hoc additions: A7 one-time form, the imperfect doctor, A4 (no cited evidence), and the
budget-sweep and trial-level helpers of eval/post_hoc.py."""

import json

from trialguide.checker import rule_checker as checker
from trialguide.eval.post_hoc import at_budget, ece_of, trial_score
from trialguide.pipeline import run_patient, run_patient_form
from trialguide.policy.policy import PolicyConfig
from trialguide.schemas import Certainty, DoctorAnswer, Question, RuleKind
from trialguide.simulator.imperfect import ImperfectDoctor
from tests.test_pipeline import NOTE, RULES, FakeDoctor, fake_check

CFG = PolicyConfig("test", allow_inference=True, allow_asking=True, threshold=0.5)


def test_form_asks_once_and_rechecks_after_all_answers():
    doc = FakeDoctor()
    r = run_patient_form(NOTE, RULES, CFG, fake_check, doc, budget=3)
    assert doc.asked == ["smoking"]                       # the only fact to ask about
    assert len(r.turns) == 1 and r.n_checks == 4 + 2
    assert r.scores == run_patient(NOTE, RULES, CFG, fake_check, doctor=FakeDoctor()).scores


def test_form_without_asking_asks_nothing():
    doc = FakeDoctor()
    r = run_patient_form(NOTE, RULES, PolicyConfig("b2", True, False, 0.0), fake_check, doc)
    assert doc.asked == [] and r.turns == []


def test_imperfect_doctor_is_deterministic_and_downgrades_about_the_rate():
    class Sure:
        model = "m"
        def answer(self, pid, q):
            return DoctorAnswer(q.question_id, "yes", Certainty.SURE, quote="It says yes.")
    doc = ImperfectDoctor(Sure(), rate=0.2)
    qs = [Question(f"q{i}", "p", f"fact{i}", f"What about fact {i}?", 1) for i in range(2000)]
    out = [doc.answer("p", q).certainty for q in qs]
    assert out == [doc.answer("p", q).certainty for q in qs]           # same every time
    changed = sum(c != Certainty.SURE for c in out) / len(out)
    assert 0.17 < changed < 0.23
    assert Certainty.UNSURE in out and Certainty.NOT_DOCUMENTED in out


def test_a4_prompt_drops_only_the_evidence_request_and_keeps_conclusions():
    assert "evidence_ids" not in checker.SYSTEM_PROMPT_NO_EVIDENCE
    raw = json.dumps({"status": "met", "category": "inferred", "explanation": "x", "fact_key": "diabetes"})
    assert checker._parse_sample(raw, {0, 1}, RuleKind.INCLUSION)["status"] == "not_enough_info"
    kept = checker._parse_sample(raw, {0, 1}, RuleKind.INCLUSION, evidence=False)
    assert kept["status"] == "met" and kept["evidence_ids"] == [0, 1]


def test_trial_score_and_budget_helpers():
    assert trial_score([("met", "inclusion"), ("not_enough_info", "inclusion")]) == 0.5
    assert trial_score([("not_met", "inclusion"), ("met", "exclusion")]) == -2.0
    final = {"a": "met", "b": "not_met", "c": "met"}
    start = {"a": "not_enough_info", "b": "not_enough_info", "c": "met"}
    refs = [{"a"}, {"b"}]
    assert at_budget(final, start, refs, 0) == start
    assert at_budget(final, start, refs, 1) == {"a": "met", "b": "not_enough_info", "c": "met"}
    assert at_budget(final, start, refs, 3) == final


def test_ece_of_perfect_and_overconfident():
    assert ece_of([1.0, 1.0], [1, 1])[0] == 0.0
    assert abs(ece_of([0.9, 0.9], [0, 0])[0] - 0.9) < 1e-9
