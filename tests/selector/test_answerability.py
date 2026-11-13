import json

from trialguide.selector.answerability import estimate, p_answer


def turn(fact, certainty):
    return {"question": {"fact_key": fact}, "answer": {"certainty": certainty}}


def test_estimate_counts_each_patient_fact_once_and_smooths(tmp_path):
    f = tmp_path / "turns.jsonl"
    rows = [{"patient_id": "p1", "turns": [turn("prior_chemotherapy", "not_documented"), turn("age", "sure")]},
            {"patient_id": "p1", "turns": [turn("chemotherapy", "not_documented")]},   # same fact again
            {"patient_id": "p2", "turns": [turn("chemotherapy", "not_documented"), turn("age", "probably")]}]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    prior = estimate([f])
    assert prior["pairs"] == 4 and prior["overall"] == 0.5
    assert prior["facts"]["chemotherapy"] == {"asked": 2, "answered": 0}
    p = p_answer(prior)
    assert p("History of prior chemo") == (0 + 2 * 0.5) / (2 + 2)       # canonical key, smoothed
    assert p("age") == (2 + 1) / 4
    assert p("never asked") == 0.5                                       # unseen: the overall rate


def test_the_weight_reorders_eq1(monkeypatch):
    from trialguide.selector import selector
    from trialguide.schemas import ConversationState
    a, b = selector.Candidate("chemotherapy", [("T1", "r1")]), selector.Candidate("age", [("T1", "r2")])
    monkeypatch.setattr(selector, "can_change_top_k", lambda c, checks, k: True)
    monkeypatch.setattr(selector, "score_candidate", lambda c, checks, p_met=None: 1.0 if c is a else 0.6)
    state = ConversationState("p1", budget=3)
    assert selector.select_question([a, b], [], state).fact_key == "chemotherapy"
    weights = {"chemotherapy": 0.1, "age": 0.9}
    assert selector.select_question([a, b], [], state, p_answer=weights.get).fact_key == "age"
