"""Linking the expert criterion labels to our rule ids (T30)."""

from pathlib import Path

import pytest

from trialguide.eval.expert_reference import build_reference, expert_status, match_rule, normalise
from trialguide.schemas import RuleKind, RuleStatus, TrialRule

INC, EXC = RuleKind.INCLUSION, RuleKind.EXCLUSION


def rules():
    return {"NCT1": [TrialRule("NCT1", "inc_0", INC, "Age 18 years or older."),
                     TrialRule("NCT1", "inc_1", INC, "Diagnosis of stable angina"),
                     TrialRule("NCT1", "exc_0", EXC, "Pregnant or breast-feeding women")]}


def test_labels_map_by_kind():
    assert expert_status(INC, "included") == RuleStatus.MET
    assert expert_status(EXC, "excluded") == RuleStatus.MET            # has the excluded condition (D15)
    assert expert_status(EXC, "not excluded") == RuleStatus.NOT_MET
    assert expert_status(INC, "Not enough information") == RuleStatus.NOT_ENOUGH_INFO
    with pytest.raises(ValueError):
        expert_status(INC, "excluded")                                  # wrong label for the kind


def test_normalise_ignores_case_spaces_and_punctuation():
    assert normalise("  Age 18 years,  or OLDER. ") == "age 18 years or older"


def test_exact_fuzzy_and_unmatched():
    rs = rules()["NCT1"]
    inc = [r for r in rs if r.kind == INC]
    assert match_rule("age 18 years or older", inc) == (rs[0], "exact")
    assert match_rule("Diagnosis of stable angina.", inc)[1] == "exact"
    rule, how = match_rule("Diagnosis of a stable angina", inc)          # one word added
    assert rule is rs[1] and how == "fuzzy"
    assert match_rule("Prior myocardial infarction", inc) == (None, "unmatched")


def test_build_reference_and_report():
    ann = [
        {"patient_id": "p", "trial_id": "NCT1", "criterion_type": "inclusion",
         "criterion_text": "Age 18 years or older", "expert_eligibility": "included"},
        {"patient_id": "p", "trial_id": "NCT1", "criterion_type": "exclusion",
         "criterion_text": "Pregnant or breast-feeding women", "expert_eligibility": "not excluded"},
        {"patient_id": "p", "trial_id": "NCT1", "criterion_type": "inclusion",
         "criterion_text": "Something no rule says", "expert_eligibility": "included"},
        {"patient_id": "p", "trial_id": "NCT9", "criterion_type": "inclusion",
         "criterion_text": "Trial we do not have", "expert_eligibility": "included"},
    ]
    ref, report = build_reference(ann, rules())
    assert ref == {("p", "NCT1", "inc_0"): RuleStatus.MET, ("p", "NCT1", "exc_0"): RuleStatus.NOT_MET}
    assert report["exact"] == 2 and len(report["unmatched"]) == 2 and report["matched_rules"] == 2


def test_two_rows_that_disagree_on_one_rule_are_dropped():
    ann = [{"patient_id": "p", "trial_id": "NCT1", "criterion_type": "inclusion",
            "criterion_text": "Age 18 years or older", "expert_eligibility": lab}
           for lab in ("included", "not included")]
    ref, report = build_reference(ann, rules())
    assert ref == {} and report["conflicts"] == 1


RAW = Path("data/raw")


@pytest.mark.skipif(not (RAW / "annotations").exists(), reason="run scripts/download_data.sh first")
def test_most_expert_rows_match_a_rule_on_real_data():
    from trialguide.eval.expert_reference import load_reference
    ref, report = load_reference(RAW)
    assert report["rows"] == 1015
    assert (report["exact"] + report["fuzzy"]) / report["rows"] >= 0.9
