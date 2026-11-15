"""T44/T45: automatic comparison of the full-note checker with the experts, on hand-made rules."""

import pytest

from trialguide.eval import reference_audit as ra


def _expert(status, evidence=(), kind="inclusion", n=8):
    return {"status": status, "kind": kind, "evidence": set(evidence), "n_sentences": n}


def _check(status, cited=(), kind="inclusion"):
    return {"status": status, "kind": kind, "evidence_ids": list(cited)}


def _rows():
    experts = {("p1", "T", "inc_0"): _expert("met", {1, 2}),                    # settled, evidence {1,2}
               ("p1", "T", "inc_1"): _expert("not_met", {4}),                   # settled, checker wrong, toward eligibility? no
               ("p1", "T", "exc_0"): _expert("met", {3}, "exclusion"),          # exclusion MET; checker says not_met = wrong toward eligibility
               ("p2", "T", "inc_0"): _expert("not_enough_info"),                # experts say nothing settles it
               ("p2", "T", "inc_1"): _expert("met", {0}, n=3),                  # checker cites an id outside the note
               ("p2", "T", "inc_2"): _expert("met", {5})}                       # not in the checks file at all
    checks = {("p1", "T", "inc_0"): _check("met", {2, 6}),
              ("p1", "T", "inc_1"): _check("met", {4}),
              ("p1", "T", "exc_0"): _check("not_met", {3}, "exclusion"),
              ("p2", "T", "inc_0"): _check("met", {0}),                         # settles a rule experts leave open
              ("p2", "T", "inc_1"): _check("met", {0, 9})}
    return ra.audit_rows(experts, checks)


def test_only_rules_in_both_files_are_audited():
    assert len(_rows()) == 5


def test_summary_rates_by_hand():
    s = ra.summarise(_rows(), n_boot=200)
    assert s["rules"] == 5 and s["patients"] == 2
    assert s["agreement_all"].estimate == pytest.approx(2 / 5)                  # inc_0 and the p2 inc_1 agree
    assert s["expert_settled_rules"] == 4
    assert s["checker_settles_them"].estimate == pytest.approx(1.0)
    assert s["accuracy_when_settled_by_both"].estimate == pytest.approx(2 / 4)
    assert s["wrong_toward_eligibility"].estimate == pytest.approx(2 / 4)       # inc_1 (met vs not_met) and exc_0
    assert s["expert_nei_rules"] == 1 and s["checker_settles_expert_nei"].estimate == pytest.approx(1.0)


def test_evidence_precision_recall_and_invalid_citations():
    s = ra.summarise(_rows(), n_boot=200)
    # evidence rules: p1 inc_0 {1,2} vs cited {2,6}; p1 inc_1 {4} vs {4}; p1 exc_0 {3} vs {3}; p2 inc_1 {0} vs {0,9}
    assert s["evidence_rules"] == 4
    assert s["evidence_precision"].estimate == pytest.approx(4 / 6)             # overlap 1+1+1+1 over cited 2+1+1+2 = 6
    assert s["evidence_recall"].estimate == pytest.approx(4 / 5)                # overlap over expert 2+1+1+1
    assert s["evidence_exact_match"].estimate == pytest.approx(2 / 4)
    assert s["evidence_any_overlap"].estimate == pytest.approx(1.0)
    assert s["invalid_citations"] == 1                                          # id 9 in a 3-sentence note


def test_checker_not_deciding_is_not_an_error_toward_eligibility():
    rows = ra.audit_rows({("p", "T", "inc_0"): _expert("met", {1})},
                         {("p", "T", "inc_0"): _check("not_enough_info", {1})})
    assert not rows[0]["toward_eligibility_wrong"] and not rows[0]["decided"] and not rows[0]["ev"]
