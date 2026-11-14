"""Tests for the blind doctor check (D10)."""

import csv
import json
from types import SimpleNamespace

import pytest

from trialguide.eval import doctor_check as dc


def test_sample_is_half_answered_half_not_documented():
    qs = [{"doctor_certainty": "sure"}] * 10 + [{"doctor_certainty": "not_documented"}] * 50
    picked = dc.sample(qs, 20, seed=0)
    assert sum(q["doctor_certainty"] == "sure" for q in picked) == 10
    assert len(picked) == 20


def test_same_answer_and_quote_sentence():
    assert dc.same_answer("45", "45-year-old man")
    assert dc.same_answer("Type 2 diabetes", "type 2 diabetes.")
    assert not dc.same_answer("no", "yes")
    sents = {0: "A 45-year-old man.", 1: "He smokes daily."}
    assert dc.quote_sentence("He smokes daily", sents) == 1
    assert dc.quote_sentence(None, sents) is None


def test_kappa():
    assert dc.kappa([True, False, True, False], [True, False, True, False]) == pytest.approx(1.0)
    assert dc.kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)


def test_score_end_to_end(tmp_path):
    key = [{"item": 1, "patient_id": "p", "question": "age?", "doctor_certainty": "sure",
            "doctor_answer": "45", "doctor_quote": "A 45-year-old man.", "doctor_sentence_id": 0},
           {"item": 2, "patient_id": "p", "question": "smoking?", "doctor_certainty": "not_documented",
            "doctor_answer": "", "doctor_quote": None, "doctor_sentence_id": None}]
    (tmp_path / dc.KEY).write_text(json.dumps(key))
    with open(tmp_path / dc.SHEET, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "patient_id", "question", "record", "answer", "sentence_id"])
        w.writerow([1, "p", "age?", "...", "45 years", "0"])
        w.writerow([2, "p", "smoking?", "...", "smokes daily", "1"])     # the doctor missed it
    out = dc.score(SimpleNamespace(out=tmp_path))
    assert out["documented_agreement"] == 0.5
    assert out["doctor_not_documented_but_record_says"] == 1
    assert out["same_answer_when_both_answered"] == 1.0
    assert out["same_sentence_when_both_answered"] == 1.0
