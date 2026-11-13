"""Conformal risk control for harmful inferences (task T30).

The policy (T19) accepts an inference that pushes a patient TOWARD eligibility only if the
checker's confidence is at least a threshold lambda. This module picks that threshold on the
calibration split so that the expected per-patient harmful-inference rate stays at or below a
target alpha (conformal risk control, Angelopoulos et al., ICLR 2024; proposal Section 5).

Definitions (proposal Section 5):
- A rule is "unsettled" for a patient if the checker did not settle it from a stated sentence:
  its category is inferred, assumption or missing, or its status is "not enough information".
- An accepted inference is harmful if it is wrong and favours eligibility (schemas.is_harmful).
- Patient loss L_i(lambda) = harmful inferences accepted at confidence >= lambda
                            / unsettled rules of that patient          (0 if it has none).
  L_i lies in [0, 1] and never increases as lambda rises.
- lambda_hat = the smallest lambda with  n/(n+1) * mean_i L_i(lambda) + 1/(n+1) <= alpha.

If even "accept no inference toward eligibility" (L_i = 0 for everyone) fails that bound,
i.e. 1/(n+1) > alpha, the target cannot be certified with n patients. We then return
lambda_hat = inf (accept none) and say so; the target is never relaxed.

Only rules that have a reference label are scored. With expert labels as the reference, that
means the expert-labelled rules of the calibration patients (decision D16).

The doctor is not involved here: calibration uses the checker's first pass on each masked note.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from trialguide.schemas import InfoCategory, RuleCheck, RuleStatus, favours_eligibility, is_harmful

ACCEPT_NONE = math.inf      # threshold that rejects every inference toward eligibility

# (patient_id, trial_id, rule_id) -> reference status (expert label or model-derived reference)
Reference = dict


# ---------------------------------------------------------------------------
# Which rules count, and which inferences are candidates for acceptance
# ---------------------------------------------------------------------------

def is_unsettled(check: RuleCheck) -> bool:
    """True if the note alone did not settle this rule (the denominator of the loss)."""
    stated_and_decided = (check.category == InfoCategory.STATED
                          and check.status != RuleStatus.NOT_ENOUGH_INFO)
    return not stated_and_decided


def is_permissive_inference(check: RuleCheck) -> bool:
    """An inference the threshold acts on: inferred, decided, and pushing toward eligibility.

    Inferences toward exclusion are always accepted by the policy and cannot be harmful,
    so they never enter the loss.
    """
    return (check.category == InfoCategory.INFERRED
            and check.status in (RuleStatus.MET, RuleStatus.NOT_MET)
            and favours_eligibility(check.kind, check.status))


def _key(check: RuleCheck) -> tuple:
    return (check.patient_id, check.trial_id, check.rule_id)


# ---------------------------------------------------------------------------
# Per-patient counts at a threshold (also used for the test-split report)
# ---------------------------------------------------------------------------

@dataclass
class PatientCounts:
    """What happens to one patient's unsettled rules at a given threshold."""
    patient_id: str
    unsettled: int = 0                # rules the note did not settle (and that have a reference)
    accepted: int = 0                 # inferences accepted at this threshold (both directions)
    accepted_correct: int = 0         # ... that match the reference
    accepted_permissive: int = 0      # accepted inferences toward eligibility
    harmful: int = 0                  # accepted inferences that are wrong and favour eligibility

    @property
    def loss(self) -> float:
        """L_i(lambda): the share of unsettled rules that got a harmful inference."""
        return self.harmful / self.unsettled if self.unsettled else 0.0


def patient_counts(checks: Iterable[RuleCheck], reference: Reference, threshold: float,
                   patient_id: str) -> PatientCounts:
    """Count one patient's outcomes at `threshold`. Rules without a reference are skipped."""
    out = PatientCounts(patient_id)
    for c in checks:
        ref = reference.get(_key(c))
        if ref is None or not is_unsettled(c):
            continue
        out.unsettled += 1
        decided_inference = (c.category == InfoCategory.INFERRED
                             and c.status in (RuleStatus.MET, RuleStatus.NOT_MET))
        if not decided_inference:
            continue                              # assumption / missing: never used as a fact
        if is_permissive_inference(c) and c.confidence < threshold:
            continue                              # below lambda: the policy asks or abstains
        out.accepted += 1
        out.accepted_correct += c.status == ref
        if is_permissive_inference(c):
            out.accepted_permissive += 1
            out.harmful += is_harmful(c.kind, c.status, ref)
    return out


def counts_at(checks_by_patient: dict, reference: Reference, threshold: float) -> list[PatientCounts]:
    """patient_counts for every patient, in sorted patient order."""
    return [patient_counts(checks_by_patient[p], reference, threshold, p)
            for p in sorted(checks_by_patient)]


def empirical_risk(counts: list[PatientCounts]) -> float:
    """R_hat(lambda): the mean patient loss."""
    return float(np.mean([c.loss for c in counts])) if counts else 0.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    alpha: float
    n_patients: int
    lambda_hat: float                 # ACCEPT_NONE (inf) if no threshold meets the bound
    certifiable: bool                 # False if 1/(n+1) > alpha: the target cannot be met at all
    risk_at_lambda: float             # R_hat(lambda_hat) on the calibration patients
    curve: list = field(default_factory=list)   # (lambda, R_hat, bound) for every candidate

    @property
    def accepts_any(self) -> bool:
        return math.isfinite(self.lambda_hat)

    def summary(self) -> str:
        if not self.certifiable:
            return (f"alpha={self.alpha:.0%} cannot be certified with n={self.n_patients} patients "
                    f"(needs alpha >= 1/(n+1) = {1 / (self.n_patients + 1):.1%}); no inference "
                    f"toward eligibility is accepted")
        if not self.accepts_any:
            return (f"alpha={self.alpha:.0%}: no threshold meets the bound; no inference toward "
                    f"eligibility is accepted")
        return (f"alpha={self.alpha:.0%}, n={self.n_patients}: lambda_hat={self.lambda_hat:.3f}, "
                f"calibration risk {self.risk_at_lambda:.3%}")


def bound(risk: float, n: int) -> float:
    """The conformal risk-control bound n/(n+1) * R_hat + 1/(n+1)."""
    return n / (n + 1) * risk + 1 / (n + 1)


def calibrate(checks_by_patient: dict, reference: Reference, alpha: float) -> Calibration:
    """Pick lambda_hat on the calibration patients.

    checks_by_patient: patient_id -> that patient's RuleChecks (checker run on the masked note).
    reference:         (patient_id, trial_id, rule_id) -> reference RuleStatus.

    Candidate thresholds are the distinct confidences of permissive inferences, plus
    ACCEPT_NONE. A threshold equal to a confidence accepts that inference, so every candidate
    is attained; the smallest candidate that meets the bound is returned.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(checks_by_patient)
    if n == 0:
        raise ValueError("no calibration patients")

    confidences = sorted({c.confidence
                          for checks in checks_by_patient.values() for c in checks
                          if is_permissive_inference(c) and reference.get(_key(c)) is not None})
    candidates = confidences + [ACCEPT_NONE]

    curve = []
    lambda_hat = ACCEPT_NONE
    for lam in candidates:                            # ascending; the bound only gets easier
        risk = empirical_risk(counts_at(checks_by_patient, reference, lam))
        curve.append((lam, risk, bound(risk, n)))
        if bound(risk, n) <= alpha and not math.isfinite(lambda_hat):
            lambda_hat = lam
    risk_at = empirical_risk(counts_at(checks_by_patient, reference, lambda_hat))
    return Calibration(alpha=alpha, n_patients=n, lambda_hat=lambda_hat,
                       certifiable=1 / (n + 1) <= alpha, risk_at_lambda=risk_at, curve=curve)


# ---------------------------------------------------------------------------
# Reporting on the test split (patient-level bootstrap, as for every headline number)
# ---------------------------------------------------------------------------

def risk_arrays(counts: list[PatientCounts]) -> dict:
    """Per-patient arrays in the shape eval.stats.paired_bootstrap expects."""
    return {
        "loss": [c.loss for c in counts],
        "harmful": [c.harmful for c in counts],
        "unsettled": [c.unsettled for c in counts],
        "accepted": [c.accepted for c in counts],
        "accepted_correct": [c.accepted_correct for c in counts],
    }


def risk_report(counts: list[PatientCounts], n_boot: int = 10000, seed: int = 0) -> dict:
    """Harmful-inference rate (the controlled quantity), inference coverage and useful coverage,
    each with a patient-level bootstrap 95% CI, plus the raw counts the proposal asks to report."""
    from trialguide.eval.stats import mean_of, paired_bootstrap, ratio_of_sums
    arrays = risk_arrays(counts)
    return {
        "n_patients": len(counts),
        "unsettled_rules": int(sum(arrays["unsettled"])),
        "accepted_inferences": int(sum(arrays["accepted"])),
        "harmful_inferences": int(sum(arrays["harmful"])),
        "harmful_rate": paired_bootstrap(arrays, mean_of("loss"), n_boot, seed),
        "inference_coverage": paired_bootstrap(arrays, ratio_of_sums("accepted", "unsettled"), n_boot, seed),
        "useful_coverage": paired_bootstrap(arrays, ratio_of_sums("accepted_correct", "unsettled"), n_boot, seed),
    }


def threshold_for_policy(cal: Optional[Calibration]) -> float:
    """The value to pass to policy.system_config('B7', threshold=...)."""
    if cal is None:
        raise ValueError("B7 needs a calibration result")
    return cal.lambda_hat
