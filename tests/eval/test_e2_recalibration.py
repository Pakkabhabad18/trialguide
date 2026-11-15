"""T38: E2 recalibration from a rule table, on a hand-made table."""

import csv
import json

import pytest

from trialguide.eval import e2_recalibration as rc

COLS = ["condition", "patient_uid", "trial", "rule", "kind", "status", "category", "confidence", "evidence_ids",
        "final_B6", "final_B2", "final_B5"]


def _write(tmp_path, n=10, bad_every=4):
    """n patients, each with 4 rules the full note settles as NOT_MET (inclusion). On the masked note the checker infers MET
    (toward eligibility) with confidence 0.5 + 0.05 * rule index; every `bad_every`-th patient's rules are the low-confidence ones."""
    path = tmp_path / "rule_table.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(COLS)
        for p in range(n):
            for r in range(4):
                conf = round(0.5 + 0.05 * r + (0.0 if p % bad_every == 0 else 0.35), 3)
                w.writerow(["full", f"p{p}", "T", f"inc_{r}", "inclusion", "not_met", "stated", 0.99, "[0]", "", "", ""])
                w.writerow(["rule40", f"p{p}", "T", f"inc_{r}", "inclusion", "met", "inferred", conf, "[0]", "not_enough_info", "met", "met"])
    return path


def test_load_table_builds_checks_and_reference(tmp_path):
    masked, ref = rc.load_table(_write(tmp_path, n=3))
    assert sorted(masked) == ["p0", "p1", "p2"] and len(masked["p0"]) == 4 and len(ref) == 12
    assert all(v.value == "not_met" for v in ref.values())                     # reference only from the full-note rows


def test_ten_patients_cannot_certify_five_percent(tmp_path):
    masked, ref = rc.load_table(_write(tmp_path))
    out = rc.recalibrate(masked, ref)
    assert out["smallest_certifiable_alpha"] == pytest.approx(1 / 11)
    assert out["by_alpha"]["0.05"]["certifiable"] is False and out["by_alpha"]["0.05"]["accepts_any"] is False
    assert out["by_alpha"]["0.1"]["certifiable"] is True


def test_evaluate_counts_harmful_inferences_by_threshold(tmp_path):
    masked, ref = rc.load_table(_write(tmp_path))
    ev = rc.evaluate(masked, ref, {"all": 0.0, "none": float("inf")}, n_boot=100)
    assert ev["all"]["harmful_inferences"] == ev["all"]["accepted_inferences"] == 40      # every inference is wrong here
    assert ev["none"]["accepted_inferences"] == 0 and ev["none"]["threshold"] is None


def test_leave_one_out_has_a_fold_per_patient(tmp_path):
    masked, ref = rc.load_table(_write(tmp_path))
    loo = rc.leave_one_out(masked, ref, 0.10)
    assert len(loo["lambda_hat_per_fold"]) == 10
    assert loo["accept_all_B2"]["accepted"] == 40 and loo["accept_all_B2"]["unsettled"] == 40
    assert loo["recalibrated_on_the_other_9"]["harmful"] <= loo["accept_all_B2"]["harmful"]
