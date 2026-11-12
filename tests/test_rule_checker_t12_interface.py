
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
):
    return json.dumps({
        "status": status,
        "category": category,
        "evidence_ids": [0] if evidence_ids is None else evidence_ids,
        "explanation": explanation,
    })


def test_fact_key_is_normalised():
    parsed = checker._parse_sample(
        response(
            status="not_enough_info",
            category="missing",
            evidence_ids=[],
        )[:-1] + ', "fact_key": "Prior Chemotherapy"}',
        {0, 1},
        RuleKind.INCLUSION,
    )
    assert parsed["fact_key"] == "prior_chemotherapy"


def test_missing_fact_key_is_none():
    parsed = checker._parse_sample(
        response(
            status="not_enough_info",
            category="missing",
            evidence_ids=[],
        ),
        {0, 1},
        RuleKind.INCLUSION,
    )
    assert parsed["fact_key"] is None


def test_unresolved_rule_gets_consensus_fact_key(monkeypatch):
    outputs = iter([
        json.dumps({
            "status": "not_enough_info",
            "category": "missing",
            "evidence_ids": [],
            "explanation": "History is not documented.",
            "fact_key": "prior_chemotherapy",
        }),
        json.dumps({
            "status": "not_enough_info",
            "category": "missing",
            "evidence_ids": [],
            "explanation": "History is not documented.",
            "fact_key": "Prior Chemotherapy",
        }),
        json.dumps({
            "status": "not_enough_info",
            "category": "missing",
            "evidence_ids": [],
            "explanation": "History is not documented.",
            "fact_key": "prior_chemotherapy",
        }),
    ])
    monkeypatch.setattr(checker, "chat", lambda *args, **kwargs: next(outputs))

    result = checker.check_rule(make_note(), make_rule(), "mock-model", n_samples=3)

    assert result.status == RuleStatus.NOT_ENOUGH_INFO
    assert result.category == InfoCategory.MISSING
    assert result.fact_key == "prior_chemotherapy"


def test_conflicting_fact_keys_do_not_pick_arbitrary_key(monkeypatch):
    outputs = iter([
        json.dumps({
            "status": "not_enough_info",
            "category": "missing",
            "evidence_ids": [],
            "fact_key": "prior_chemotherapy",
        }),
        json.dumps({
            "status": "not_enough_info",
            "category": "missing",
            "evidence_ids": [],
            "fact_key": "radiation_history",
        }),
    ])
    monkeypatch.setattr(checker, "chat", lambda *args, **kwargs: next(outputs))

    result = checker.check_rule(make_note(), make_rule(), "mock-model", n_samples=2)

    assert result.status == RuleStatus.NOT_ENOUGH_INFO
    assert result.fact_key is None
