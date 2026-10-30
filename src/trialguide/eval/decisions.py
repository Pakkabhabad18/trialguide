"""Trial-level decision metrics (T17): false-eligible and false-exclusion rates.

A system turns the rule results of each (patient, trial) pair into one decision:
"eligible", "excluded" or "unknown" (it could not decide: abstention). Against the gold grade
(2 = eligible; 0 and 1 = not eligible, i.e. not relevant or excluded):

    false-eligible rate  = pairs decided "eligible" that are not eligible / pairs that are not eligible
                           (an ineligible patient shown as eligible: the safety-relevant error)
    false-exclusion rate = pairs decided "excluded" that are eligible / pairs that are eligible
                           (an eligible patient ruled out: a missed trial)
    precision, recall    = of the "eligible" decisions, and of the eligible pairs
    unknown rate         = pairs left undecided / all pairs

Errors toward eligibility and toward exclusion are always reported separately. Rates are pooled
over pairs, but intervals come from resampling PATIENTS, because the pairs of one patient are
correlated.
"""

import numpy as np

from trialguide.eval.stats import BootResult, paired_bootstrap, ratio_of_sums

ELIGIBLE, EXCLUDED, UNKNOWN = "eligible", "excluded", "unknown"
DECISIONS = (ELIGIBLE, EXCLUDED, UNKNOWN)


def decision_counts(predicted: dict, gold: dict, patients: list) -> dict:
    """Per-patient count arrays (one entry per patient in `patients`, in that order).

    predicted: {patient_id: {trial_id: decision}}; only trials with a gold grade are counted.
    """
    names = ("n_pairs", "n_eligible", "n_ineligible", "false_eligible", "false_exclusion",
             "pred_eligible", "true_eligible", "unknown")
    out = {k: np.zeros(len(patients)) for k in names}
    for i, pid in enumerate(patients):
        for trial, decision in predicted.get(pid, {}).items():
            if trial not in gold[pid]:
                continue
            if decision not in DECISIONS:
                raise ValueError(f"unknown decision {decision!r} for {pid}/{trial}")
            eligible = gold[pid][trial] == 2
            out["n_pairs"][i] += 1
            out["n_eligible"][i] += eligible
            out["n_ineligible"][i] += not eligible
            out["unknown"][i] += decision == UNKNOWN
            if decision == ELIGIBLE:
                out["pred_eligible"][i] += 1
                out["true_eligible"][i] += eligible
                out["false_eligible"][i] += not eligible
            elif decision == EXCLUDED and eligible:
                out["false_exclusion"][i] += 1
    return out


RATES = {
    "false_eligible_rate": ("false_eligible", "n_ineligible"),
    "false_exclusion_rate": ("false_exclusion", "n_eligible"),
    "precision_eligible": ("true_eligible", "pred_eligible"),
    "recall_eligible": ("true_eligible", "n_eligible"),
    "unknown_rate": ("unknown", "n_pairs"),
}


def decision_report(counts: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    """{rate name: BootResult} with patient-level bootstrap 95% CIs."""
    return {name: paired_bootstrap(counts, ratio_of_sums(num, den), n_boot, seed)
            for name, (num, den) in RATES.items()}


__all__ = ["ELIGIBLE", "EXCLUDED", "UNKNOWN", "decision_counts", "decision_report", "BootResult"]
