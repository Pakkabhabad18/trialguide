
import json
from types import SimpleNamespace

import pytest

from trialguide.checker import rule_checker as checker
from trialguide.schemas import InfoCategory, RuleKind, RuleStatus


def make_note():
    return SimpleNamespace(
        patient_id="P1",
        sentences={
            0: "The patient is 45 years old.",
            1: "The patient has no documented history of diabetes.",
        },
        text=lambda: (
            "[0] The patient is 45 years old.\n"
            "[1] The patient has no documented history of diabetes."
        ),
    )


def make_rule(kind=RuleKind.INCLUSION):
    return SimpleNamespace(
        trial_id="T1",
        rule_id="R1",
        kind=kind,
        text="The patient must be at least 18 years old.",
    )


def response(
    status="met",
    category="stated",
    evidence_ids=None,
    explanation="Supported by the note.",
    fact_key=None,
):
    return json.dumps({
        "status": status,
        "category": category,
        "evidence_ids": [0] if evidence_ids is None else evidence_ids,
        "explanation": explanation,
        "fact_key": fact_key,
    })

def test_inferred_result_preserves_fact_key(monkeypatch):
    monkeypatch.setattr(
        checker,
        "chat",
        lambda *args, **kwargs: response(
            status="met",
            category="inferred",
            evidence_ids=[0],
            fact_key="age",
        ),
    )

    result = checker.check_rule(
        make_note(), make_rule(), "mock-model", n_samples=1
    )

    assert result.status == RuleStatus.MET
    assert result.category == InfoCategory.INFERRED
    assert result.fact_key == "age"

def test_inclusion_alias_included_maps_to_met():
    result = checker._parse_sample(
        response(status="included"),
        {0, 1},
        RuleKind.INCLUSION,
    )
    assert result["status"] == RuleStatus.MET.value


def test_exclusion_alias_excluded_maps_to_met():
    result = checker._parse_sample(
        response(status="excluded"),
        {0, 1},
        RuleKind.EXCLUSION,
    )
    assert result["status"] == RuleStatus.MET.value


def test_invalid_evidence_cannot_support_decisive_status():
    result = checker._parse_sample(
        response(status="met", category="stated", evidence_ids=[999]),
        {0, 1},
        RuleKind.INCLUSION,
    )
    assert result["status"] == RuleStatus.NOT_ENOUGH_INFO.value
    assert result["category"] == InfoCategory.MISSING.value
    assert result["evidence_ids"] == []


@pytest.mark.parametrize("category", ["assumption", "missing"])
def test_assumption_or_missing_never_becomes_decisive(category):
    result = checker._parse_sample(
        response(status="met", category=category, evidence_ids=[0]),
        {0, 1},
        RuleKind.INCLUSION,
    )
    assert result["status"] == RuleStatus.NOT_ENOUGH_INFO.value
    assert result["evidence_ids"] == []


def test_unknown_status_is_rejected():
    result = checker._parse_sample(
        response(status="probably"),
        {0, 1},
        RuleKind.INCLUSION,
    )
    assert result is None


def test_malformed_json_is_rejected():
    assert checker._parse_sample(
        "not json",
        {0, 1},
        RuleKind.INCLUSION,
    ) is None


def test_invalid_samples_reduce_confidence(monkeypatch):
    outputs = iter([
        response(status="met", evidence_ids=[0]),
        response(status="met", evidence_ids=[0]),
        "malformed response",
    ])
    monkeypatch.setattr(checker, "chat", lambda *args, **kwargs: next(outputs))

    result = checker.check_rule(make_note(), make_rule(), "mock-model", n_samples=3)

    assert result.status == RuleStatus.MET
    assert result.category == InfoCategory.STATED
    assert result.evidence_ids == [0]
    assert result.confidence == pytest.approx(2 / 3)
    assert result.n_samples == 3


def test_all_invalid_samples_return_unresolved(monkeypatch):
    monkeypatch.setattr(checker, "chat", lambda *args, **kwargs: "not json")

    result = checker.check_rule(make_note(), make_rule(), "mock-model", n_samples=2)

    assert result.status == RuleStatus.NOT_ENOUGH_INFO
    assert result.category == InfoCategory.MISSING
    assert result.evidence_ids == []
    assert result.confidence == 0.0


def test_zero_samples_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        checker.check_rule(make_note(), make_rule(), "mock-model", n_samples=0)
def test_exclusion_met_remains_met_and_is_unresolved_only_if_needed(monkeypatch):

    monkeypatch.setattr(
        checker,
        "chat",
        lambda *args, **kwargs: response(
            fact_key="diabetes_history",
            status="excluded",
            category="stated",
            evidence_ids=[1],
        ),
    )

    result = checker.check_rule(
        make_note(),
        make_rule(RuleKind.EXCLUSION),
        "mock-model",
        n_samples=1,
    )

    assert result.kind == RuleKind.EXCLUSION
    assert result.status == RuleStatus.MET
    assert result.category == InfoCategory.STATED
    assert result.evidence_ids == [1]
    assert result.unresolved is False
    assert result.fact_key == "diabetes_history"

def test_prompt_excludes_patient_and_trial_ids(monkeypatch):
    captured = {}

    def fake_chat(*args, **kwargs):
        captured["prompt"] = str((args, kwargs))
        return response(fact_key="age")

    monkeypatch.setattr(checker, "chat", fake_chat)

    checker.check_rule(
        make_note(), make_rule(), "mock-model", n_samples=1
    )

    assert "P1" not in captured["prompt"]
    assert "T1" not in captured["prompt"]

def test_invalid_evidence_results_in_unresolved_rule(monkeypatch):
    monkeypatch.setattr(
        checker,
        "chat",
        lambda *args, **kwargs: response(
            status="not_met",
            category="inferred",
            evidence_ids=[999],
        ),
    )

    result = checker.check_rule(
        make_note(),
        make_rule(),
        "mock-model",
        n_samples=1,
    )

    assert result.status == RuleStatus.NOT_ENOUGH_INFO
    assert result.category == InfoCategory.MISSING
    assert result.evidence_ids == []
    assert result.unresolved is True


def test_not_applicable_status_is_preserved_when_evidence_backed(monkeypatch):
    monkeypatch.setattr(
        checker,
        "chat",
        lambda *args, **kwargs: response(
            status="not_applicable",
            category="stated",
            evidence_ids=[0],
        ),
    )

    result = checker.check_rule(
        make_note(),
        make_rule(),
        "mock-model",
        n_samples=1,
    )

    assert result.status == RuleStatus.NOT_APPLICABLE
    assert result.unresolved is False
    assert result.fact_key is None

def test_tied_status_votes_return_unresolved(monkeypatch):
    outputs = iter([
        response(status="met", category="stated", evidence_ids=[0]),
        response(status="not_met", category="stated", evidence_ids=[1]),
    ])
    monkeypatch.setattr(
        checker, "chat", lambda *args, **kwargs: next(outputs)
    )

    result = checker.check_rule(
        make_note(), make_rule(), "mock-model", n_samples=2
    )

    assert result.status == RuleStatus.NOT_ENOUGH_INFO
    assert result.category == InfoCategory.MISSING
    assert result.evidence_ids == []
    assert result.unresolved is True


def test_tied_categories_for_winning_status_return_unresolved(monkeypatch):
    outputs = iter([
        response(status="met", category="stated", evidence_ids=[0]),
        response(status="met", category="inferred", evidence_ids=[1]),
    ])
    monkeypatch.setattr(
        checker, "chat", lambda *args, **kwargs: next(outputs)
    )

    result = checker.check_rule(
        make_note(), make_rule(), "mock-model", n_samples=2
    )

    assert result.status == RuleStatus.NOT_ENOUGH_INFO
    assert result.category == InfoCategory.MISSING
    assert result.evidence_ids == []
    assert result.unresolved is True

def logprob_reply(status="met", p_status=0.8):
    """A reply as chat(..., logprobs=True) returns it, with the status value's tokens at p_status."""
    import math
    head, tail = '{"status": "', '", "category": "inferred", "evidence_ids": [0], "explanation": "x", "fact_key": null}'
    return {"text": head + status + tail,
            "tokens": [[head, -0.01], [status, math.log(p_status)], [tail, -0.02]]}


def test_logprob_confidence_is_the_probability_of_the_chosen_status(monkeypatch):
    calls = []

    def fake_chat(messages, model, temperature=0, logprobs=False):
        calls.append(logprobs)
        return logprob_reply("met", 0.8)

    monkeypatch.setattr(checker, "chat", fake_chat)
    result = checker.check_rule(make_note(), make_rule(), "model", n_samples=1, confidence="logprob")
    assert calls == [True]
    assert result.status == RuleStatus.MET and result.category == InfoCategory.INFERRED
    assert result.confidence == pytest.approx(0.8)


def test_logprob_confidence_needs_one_sample_and_a_known_mode():
    with pytest.raises(ValueError):
        checker.check_rule(make_note(), make_rule(), "model", n_samples=5, confidence="logprob")
    with pytest.raises(ValueError):
        checker.check_rule(make_note(), make_rule(), "model", n_samples=1, confidence="other")


def test_status_probability_reads_only_the_status_value():
    reply = logprob_reply("not_met", 0.3)
    assert checker._status_probability(reply["tokens"]) == pytest.approx(0.3)
    assert checker._status_probability([["no json here", -0.1]]) == 0.0
