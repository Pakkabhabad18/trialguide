"""Tests for the test-run error analysis: rule types, outcomes and the lost-rule table."""

from trialguide.eval import error_analysis as ea


def test_rule_types():
    assert ea.rule_type("Age 18 years or older") == "age"
    assert ea.rule_type("Pregnant or breastfeeding women") == "pregnancy"
    assert ea.rule_type("Serum creatinine > 1.5 mg/dL") == "lab or vital value"
    assert ea.rule_type("Willing to give written informed consent") == "consent or compliance"
    assert ea.rule_type("Prior chemotherapy within 4 weeks") == "prior treatment"
    assert ea.rule_type("History of myocardial infarction") == "condition or other"


def test_outcome_directions():
    assert ea.outcome("met", "met", "inclusion") == "recovered"
    assert ea.outcome("met", "not_met", "inclusion") == "wrong toward eligibility"
    assert ea.outcome("met", "not_met", "exclusion") == "wrong toward exclusion"
    assert ea.outcome("not_enough_info", "met", "inclusion") == "still open"


def test_lost_rules_only_counts_rules_b6_leaves_open():
    full = {"p": {"T|a": {"status": "met"}, "T|b": {"status": "not_met"}, "T|c": {"status": "not_enough_info"}}}
    b6 = {"p": {"T|a": "not_enough_info", "T|b": "not_met", "T|c": "not_enough_info"}}
    b7 = {"p": {"T|a": ["met", "inferred", "inclusion"], "T|b": ["not_met", "stated", "exclusion"],
                "T|c": ["not_enough_info", "missing", "inclusion"]}}
    rows = ea.lost_rules(full, b6, b7, about={}, rule_text={"T|a": "Age over 18"})
    assert len(rows) == 1                      # b is settled by B6, c is not settled on the full note
    assert rows[0]["outcome"] == "recovered" and rows[0]["settled_by"] == "inference"
    assert rows[0]["type"] == "age"
