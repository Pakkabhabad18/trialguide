"""Exploratory resplits (T36): how much does the realised harmful-inference rate depend on which
patients happened to be in calibration and which in test?

    python -m trialguide.eval.resplits --cal outputs/cal/checks_cal.jsonl --test outputs/test/checks_test.jsonl \
        --out results/resplits

The proposal (Section 5): "we repeat the calibration/test split 200 times within the non-development
patients". Here the pool is the 52 expert-labelled patients (26 calibration + 26 test), because only
they have expert reference labels. Each resplit shuffles the pool, calibrates lambda-hat on the first
half exactly as the primary run does (risk_control.calibrate, alpha = 5%) and measures the harmful-
inference rate on the second half at that lambda-hat. The fixed 0.9 threshold (B5) and "accept every
inference" (B2) are measured on the same test half for comparison.

These resplits reuse the test patients, so they are exploratory: reported separately and never used to
choose models, prompts or thresholds. The numbers describe the spread of one realised rate; they are
not a second test. Owner: Murthy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from trialguide.policy.risk_control import calibrate, counts_at, empirical_risk

SEED = 20261115            # fixed before any resplit was run
N_RESPLITS = 200
ALPHA = 0.05
FIXED = {"B2": 0.0, "B5": 0.9}


def _rate(counts) -> float:
    return empirical_risk(counts)


def _summary(counts) -> dict:
    return {"harmful_rate": _rate(counts),
            "accepted": int(sum(c.accepted for c in counts)),
            "harmful": int(sum(c.harmful for c in counts)),
            "unsettled": int(sum(c.unsettled for c in counts))}


def resplit(checks_by_patient: dict, reference: dict, n_resplits: int = N_RESPLITS,
            alpha: float = ALPHA, seed: int = SEED) -> list:
    """One row per resplit: the calibration half's lambda-hat and, on the test half, the harmful-
    inference rate and counts of B7 (lambda-hat), B5 (0.9) and B2 (accept all)."""
    patients = sorted(checks_by_patient)
    half = len(patients) // 2
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_resplits):
        order = rng.permutation(len(patients))
        cal = {patients[j]: checks_by_patient[patients[j]] for j in order[:half]}
        test = {patients[j]: checks_by_patient[patients[j]] for j in order[half:]}
        c = calibrate(cal, reference, alpha)
        row = {"resplit": i, "n_cal": len(cal), "n_test": len(test),
               "lambda_hat": c.lambda_hat if math.isfinite(c.lambda_hat) else None,
               "accepts_any": c.accepts_any, "cal_risk": c.risk_at_lambda}
        row["B7"] = _summary(counts_at(test, reference, c.lambda_hat))
        for name, lam in FIXED.items():
            row[name] = _summary(counts_at(test, reference, lam))
        rows.append(row)
    return rows


def summarise(rows: list, alpha: float = ALPHA) -> dict:
    """Spread of the test-half harmful-inference rate over the resplits, per system."""
    out = {"n_resplits": len(rows), "alpha": alpha,
           "share_accepting_nothing": float(np.mean([not r["accepts_any"] for r in rows]))}
    lam = [r["lambda_hat"] for r in rows if r["lambda_hat"] is not None]
    out["lambda_hat"] = ({"mean": float(np.mean(lam)), "sd": float(np.std(lam, ddof=1)) if len(lam) > 1 else 0.0,
                          "min": float(min(lam)), "max": float(max(lam))} if lam else None)
    for name in ("B7", "B5", "B2"):
        rate = np.array([r[name]["harmful_rate"] for r in rows])
        acc = np.array([r[name]["accepted"] for r in rows])
        out[name] = {"mean_rate": float(rate.mean()), "sd": float(rate.std(ddof=1)) if len(rate) > 1 else 0.0,
                     "p2.5": float(np.percentile(rate, 2.5)), "median": float(np.median(rate)),
                     "p97.5": float(np.percentile(rate, 97.5)),
                     "share_at_or_below_alpha": float(np.mean(rate <= alpha)),
                     "mean_accepted": float(acc.mean())}
    return out


def main(argv=None) -> None:
    from trialguide.policy.calibrate import read_checks
    from trialguide.baselines.run_risk import EXPERT
    import pandas as pd
    from trialguide.schemas import RuleStatus

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cal", type=Path, required=True, help="checks_cal.jsonl from run_risk --split cal")
    ap.add_argument("--test", type=Path, required=True, help="checks_test.jsonl from run_risk --split test")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=N_RESPLITS)
    args = ap.parse_args()

    checks = {**read_checks(args.cal), **read_checks(args.test)}
    ann = pd.read_parquet(args.raw / "annotations" / "criterion_annotations.parquet")
    ann = ann[ann.patient_id.isin(set(checks))]
    reference = {(r.patient_id, r.trial_id, f"ann_{r.annotation_id}"): RuleStatus(EXPERT[r.expert_eligibility])
                 for r in ann.itertuples()}
    rows = resplit(checks, reference, args.n)
    summary = summarise(rows)
    summary["pool"] = {"patients": len(checks), "rules": len(reference)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "resplits.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
