import json
from types import SimpleNamespace

import pytest

from trialguide.checker import rule_checker as checker
from trialguide.tools import note_tools
from trialguide.tools.base import ToolResult


@pytest.fixture(autouse=True)
def fresh_cache():
    note_tools.tool_block.cache_clear()
    yield
    note_tools.tool_block.cache_clear()


def no_drugs(monkeypatch):
    monkeypatch.setattr(note_tools, "note_drugs", lambda text, model: [])


def test_egfr_from_stated_creatinine_age_and_sex():
    line = note_tools.egfr_line("A 67-year-old woman. Her serum creatinine is 1.9 mg/dL.")
    assert line.startswith("eGFR 28") and "creatinine 1.9 mg/dL" in line and "female" in line


def test_no_egfr_when_an_input_is_missing():
    assert note_tools.egfr_line("Her serum creatinine is 1.9 mg/dL.") is None          # no age
    assert note_tools.egfr_line("A 67-year-old patient. Creatinine 1.9 mg/dL.") is None  # no sex


def test_bmi_from_weight_and_height_but_not_when_stated():
    line = note_tools.bmi_line("She is 165 cm tall and her weight is 113 kg.")
    assert line.startswith("BMI 41.5")
    assert note_tools.bmi_line("Weight 113 kg, height 165 cm, BMI is 41.") is None


def test_listed_drugs_must_appear_in_the_note(monkeypatch):
    import trialguide.llm.client as client
    monkeypatch.setattr(client, "chat", lambda *a, **k: json.dumps({"drugs": ["Metformin", "warfarin", "metformin"]}))
    assert note_tools.note_drugs("He takes metformin 500 mg twice daily.", "m") == ["Metformin"]


def test_block_lists_only_computed_results(monkeypatch):
    monkeypatch.setattr(note_tools, "note_drugs", lambda text, model: ["metformin", "unknownium"])
    monkeypatch.setattr(note_tools, "drug_class", lambda name: ToolResult(
        "drug_class_atc", value=[{"atc": "A10BA", "name": "Biguanides"}]) if name == "metformin"
        else ToolResult("drug_class_atc", missing="no ATC class found"))
    block = note_tools.tool_block("A 50-year-old man on metformin.", "m")
    assert "metformin: A10BA Biguanides" in block and "unknownium" not in block
    assert "eGFR" not in block and "BMI" not in block


def test_empty_block_when_no_tool_applies(monkeypatch):
    no_drugs(monkeypatch)
    assert note_tools.tool_block("A 50-year-old man with a cough.", "m") == ""


def make_note(text):
    return SimpleNamespace(patient_id="P1", sentences={0: text}, text=lambda: f"[0] {text}")


def make_rule():
    return SimpleNamespace(trial_id="T1", rule_id="R1", kind=checker.RuleKind.INCLUSION,
                           text="eGFR at least 30 mL/min/1.73 m2.")


def run_checker(monkeypatch, tools, text="A 67-year-old woman with serum creatinine 1.9 mg/dL."):
    prompts = []

    def fake_chat(messages, model, temperature=0):
        prompts.append(messages[1]["content"])
        return json.dumps({"status": "not_met", "category": "inferred", "evidence_ids": [0],
                           "explanation": "eGFR below 30", "fact_key": "egfr"})

    monkeypatch.setattr(checker, "chat", fake_chat)
    no_drugs(monkeypatch)
    checker.check_rule(make_note(text), make_rule(), "m", n_samples=1, tools=tools)
    return prompts[0]


def test_checker_prompt_gets_the_tool_results_only_when_asked(monkeypatch):
    assert "eGFR 28" in run_checker(monkeypatch, tools=True)
    assert "Calculator results" not in run_checker(monkeypatch, tools=False)


def test_prompt_without_tool_results_is_unchanged(monkeypatch):
    # an empty block must leave the prompt (and so the model-call cache key) as it was
    text = "A 50-year-old man with a cough."
    assert run_checker(monkeypatch, True, text) == run_checker(monkeypatch, False, text)
