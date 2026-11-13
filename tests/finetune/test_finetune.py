"""Tests for E1 (T23): training targets, prompt identity with the checker, label masking, comparison."""

import json

import pytest

from trialguide.checker import rule_checker
from trialguide.finetune import compare, data, train_qlora
from trialguide.schemas import RuleKind, Split, TrialRule, VisibleNote


def teacher(status="met", category="stated", evidence=(1,), kind="inclusion", fact_key="age", tools=()):
    return {"status": status, "category": category, "evidence_ids": list(evidence), "kind": kind,
            "explanation": "teacher reason", "fact_key": fact_key, "tool_calls": list(tools)}


# ---- targets ----

def test_visible_evidence_keeps_teacher_answer():
    kind, target = data.target_for(teacher(evidence=(1, 2)), {0, 1, 2}, "1")
    assert kind == "visible"
    assert json.loads(target) == {"status": "met", "category": "stated", "evidence_ids": [1, 2],
                                  "explanation": "teacher reason", "fact_key": "age"}


def test_hidden_evidence_becomes_not_enough_info_with_fact_key():
    kind, target = data.target_for(teacher(evidence=(3,)), {0, 1}, "1")
    t = json.loads(target)
    assert kind == "hidden"
    assert (t["status"], t["category"], t["evidence_ids"], t["fact_key"]) == ("not_enough_info", "missing", [], "age")
    assert "age" in t["explanation"]


def test_mixed_evidence_tool_answers_and_label_conflicts_are_dropped():
    assert data.target_for(teacher(evidence=(1, 3)), {0, 1}, "1") == ("drop_mixed_evidence", None)
    assert data.target_for(teacher(tools=({"name": "bmi"},)), {0, 1}, "1") == ("drop_tool", None)
    # trial labelled eligible, but the teacher says the patient fails an inclusion rule
    assert data.target_for(teacher(status="not_met"), {0, 1}, "2")[0] == "drop_contradicts_trial_label"
    # ... or has an excluded condition
    assert data.target_for(teacher(kind="exclusion"), {0, 1}, "2")[0] == "drop_contradicts_trial_label"
    # an answer toward eligibility on an eligible trial is fine
    assert data.target_for(teacher(), {0, 1}, "2")[0] == "visible"


def test_unresolved_and_not_applicable_keep_the_teacher():
    kind, target = data.target_for(teacher("not_enough_info", "missing", ()), {0}, "0")
    assert kind == "unresolved" and json.loads(target)["status"] == "not_enough_info"
    kind, target = data.target_for(teacher("not_applicable", "stated", ()), {0}, "0")
    assert kind == "not_applicable" and json.loads(target)["fact_key"] is None


def test_training_prompt_is_the_checkers_prompt():
    """The tuned model must see at training time exactly what the checker sends at test time."""
    note = VisibleNote("p1", Split.DEV, {0: "A 54-year-old man.", 2: "He has type 2 diabetes."})
    rule = TrialRule("NCT1", "inc_0", RuleKind.INCLUSION, "Age 18 or older")
    sent = []

    def fake_chat(messages, model, **kw):
        sent.append(messages)
        return {"text": '{"status":"met","category":"stated","evidence_ids":[0],"explanation":"x",'
                        '"fact_key":"age"}', "tokens": [["x", 0.0]]}

    original = rule_checker.chat
    rule_checker.chat = fake_chat
    try:
        rule_checker.check_rule(note, rule, "m", n_samples=1, confidence="logprob", tools=False)
    finally:
        rule_checker.chat = original
    msgs = data.messages_for(note, rule, "{}")
    assert msgs[:2] == sent[0]
    assert msgs[2] == {"role": "assistant", "content": "{}"}


def test_build_splits_by_patient_and_caps(monkeypatch):
    rules = [TrialRule("T1", f"inc_{i}", RuleKind.INCLUSION, f"rule {i}") for i in range(3)]
    monkeypatch.setattr("trialguide.pipeline.rules_of_trial", lambda trial: rules)

    class P:
        def __init__(self, pid):
            self.patient_id, self.split, self.sentences = pid, Split.DEV, {0: "a.", 1: "b.", 2: "c."}

    pids = [f"p{i}" for i in range(5)]
    manifest = {"patients": {p: {"hidden_ids": [2], "trials": {"0": ["T1"]}} for p in pids}}
    answers = {}
    for p in pids:
        answers[(p, "T1", "inc_0")] = teacher(evidence=(0,))        # visible
        answers[(p, "T1", "inc_1")] = teacher(evidence=(2,))        # hidden
        answers[(p, "T1", "inc_2")] = teacher("not_enough_info", "missing", ())
    caps = {"visible": 2, "hidden": 10, "unresolved": 10, "not_applicable": 10}
    train, val, report = data.build(manifest, answers, {p: P(p) for p in pids}, {"T1": {}}, caps,
                                    n_val_patients=1)
    assert len(report["val_patients"]) == 1
    assert {e["meta"]["patient_id"] for e in val} == set(report["val_patients"])
    assert not {e["meta"]["patient_id"] for e in train} & set(report["val_patients"])
    assert report["train"] == {"visible": 2, "hidden": 4, "unresolved": 4}
    # the hidden sentence never reaches the prompt
    assert all("2. c." not in e["messages"][1]["content"] for e in train + val)


# ---- training: only the answer is trained on ----

class FakeTokenizer:
    """Character-level stand-in for the Qwen tokenizer."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return "".join(m["content"] for m in messages) + "|A|"

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(ch) for ch in text]}


def test_encode_masks_the_prompt_and_ends_the_answer():
    ex = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                       {"role": "assistant", "content": "{}"}]}
    enc = train_qlora.encode(ex, FakeTokenizer(), max_len=100)
    n_prompt = len("SU|A|")
    assert enc["labels"][:n_prompt] == [-100] * n_prompt
    assert "".join(map(chr, enc["labels"][n_prompt:])) == "{}" + train_qlora.END
    assert train_qlora.encode(ex, FakeTokenizer(), max_len=5) is None      # too long: skipped


def test_collate_pads_without_training_on_padding():
    torch = pytest.importorskip("torch")
    batch = train_qlora.collate([{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
                                 {"input_ids": [4], "labels": [4]}], pad_id=0)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [4, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert isinstance(batch["input_ids"], torch.Tensor)


# ---- comparison ----

def test_rule_outcomes_and_summary():
    k1, k2, k3 = ("p", "T", "ann_1"), ("p", "T", "ann_2"), ("p", "T", "ann_3")
    status = {k1: "met", k2: "not_met", k3: "met"}
    evidence = {k1: {0}, k2: {5}, k3: {5}}
    hidden = {"p": {5}}
    checks = {k1: {"status": "met", "kind": "inclusion"},              # right
              k2: {"status": "met", "kind": "inclusion"},              # wrong toward eligibility
              k3: {"status": "not_enough_info", "kind": "inclusion"}}  # abstains on hidden evidence
    rows = compare.rule_outcomes(checks, status, evidence, hidden)
    s = compare.summary(rows)
    assert s["agreement"] == pytest.approx(1 / 3)
    assert s["decided_rate"] == pytest.approx(2 / 3) and s["decided_accuracy"] == pytest.approx(0.5)
    assert s["unsafe_rate"] == pytest.approx(1 / 3)
    assert s["evidence_hidden_rules"] == 2
    assert s["guess_rate_when_evidence_hidden"] == pytest.approx(0.5)


def test_paired_difference_has_the_right_sign():
    keys = [(f"p{i}", "T", "ann_1") for i in range(6)]
    row = lambda agree: {"agree": agree, "unsafe": not agree, "decided": True}
    untuned = {k: row(False) for k in keys}
    tuned = {k: row(True) for k in keys}
    d = compare.paired(untuned, tuned)
    assert d["agree"]["tuned_minus_untuned"] == pytest.approx(1.0)
    assert d["unsafe"]["tuned_minus_untuned"] == pytest.approx(-1.0)
    assert d["mcnemar_agreement"]["tuned_only_right"] == 6
