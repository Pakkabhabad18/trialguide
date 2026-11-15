"""E2 recalibration (T38, exploratory): refit the risk threshold on the E2 smoke run, from its committed rule table.

    python -m trialguide.eval.e2_recalibration --table results/e2_smoke/rule_table.csv --out results/e2_recalibration

The proposal recalibrates on part of E2 because calibration and test patients must be exchangeable and E2 notes are not like the
main benchmark. Here the only E2 data with checker outputs is the smoke run (10 calibration-half patients, `results/e2_smoke/`),
so this is the recalibration MECHANICS on that small set, not a validated threshold:

- the reference is the checker on the FULL note (model-derived; E2 has no expert labels), and "harmful" means a masked-note
  inference toward eligibility that disagrees with it, exactly as in `policy/risk_control.py`;
- with n = 10 patients the smallest certifiable target is 1/(n+1) = 9.1%, so alpha = 5% and 2% cannot be met and the policy
  would accept nothing; alpha = 10% can;
- there is no held-out E2 set (the E2 test half stays sealed), so the in-sample numbers are optimistic. A leave-one-out version is
  reported as the out-of-sample view: lambda-hat is fitted on 9 patients and applied to the 10th.

Owner: Murthy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from trialguide.policy.risk_control import calibrate, counts_at, empirical_risk, risk_report
from trialguide.schemas import InfoCategory, RuleCheck, RuleKind, RuleStatus

FROZEN_LAMBDA = 0.7660011405599875          # the main run's threshold (results/dev_run/calibration.json)
ALPHAS = (0.02, 0.05, 0.10)
DECIDED = ("met", "not_met")


def load_table(path: Path) -> tuple:
    """(masked checks by patient, reference) from the smoke run's rule table.
    Reference: the full-note status of every rule the full-note checker settled (met / not met)."""
    masked, reference = {}, {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["patient_uid"], r["trial"], r["rule"])
            if r["condition"] == "full":
                if r["status"] in DECIDED:
                    reference[key] = RuleStatus(r["status"])
            else:
                masked.setdefault(r["patient_uid"], []).append(RuleCheck(
                    patient_id=r["patient_uid"], trial_id=r["trial"], rule_id=r["rule"], kind=RuleKind(r["kind"]),
                    status=RuleStatus(r["status"]), category=InfoCategory(r["category"]),
                    evidence_ids=json.loads(r["evidence_ids"]), confidence=float(r["confidence"]), n_samples=1))
    return masked, reference


def recalibrate(masked: dict, reference: dict, alphas=ALPHAS) -> dict:
    n = len(masked)
    out = {"patients": n, "smallest_certifiable_alpha": 1 / (n + 1), "by_alpha": {}}
    for a in alphas:
        c = calibrate(masked, reference, a)
        out["by_alpha"][str(a)] = {"certifiable": c.certifiable,
                                   "lambda_hat": c.lambda_hat if math.isfinite(c.lambda_hat) else None,
                                   "accepts_any": c.accepts_any, "risk_at_lambda": c.risk_at_lambda}
    return out


def evaluate(masked: dict, reference: dict, thresholds: dict, n_boot: int = 10000) -> dict:
    """Harmful-inference rate, coverage and counts (in-sample) at each named threshold."""
    return {name: {"threshold": lam if math.isfinite(lam) else None, **risk_report(counts_at(masked, reference, lam), n_boot)}
            for name, lam in thresholds.items()}


def leave_one_out(masked: dict, reference: dict, alpha: float) -> dict:
    """lambda-hat fitted on all other patients and applied to the held-out one, for every patient."""
    held, frozen, accept_all = [], [], []
    lambdas = {}
    for pid in sorted(masked):
        rest = {p: c for p, c in masked.items() if p != pid}
        cal = calibrate(rest, reference, alpha)
        lambdas[pid] = cal.lambda_hat if math.isfinite(cal.lambda_hat) else None
        one = {pid: masked[pid]}
        held += counts_at(one, reference, cal.lambda_hat)
        frozen += counts_at(one, reference, FROZEN_LAMBDA)
        accept_all += counts_at(one, reference, 0.0)
    pack = lambda cs: {"harmful_rate": empirical_risk(cs), "accepted": sum(c.accepted for c in cs),
                       "harmful": sum(c.harmful for c in cs), "unsettled": sum(c.unsettled for c in cs)}
    return {"alpha": alpha, "lambda_hat_per_fold": lambdas, "recalibrated_on_the_other_9": pack(held),
            "frozen_threshold_0.766": pack(frozen), "accept_all_B2": pack(accept_all)}


def _plain(o):
    if hasattr(o, "estimate"):
        return {"estimate": o.estimate, "low": o.low, "high": o.high}
    raise TypeError(type(o))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.10, help="the target used for the thresholds and leave-one-out")
    args = ap.parse_args(argv)
    masked, reference = load_table(args.table)
    cal = recalibrate(masked, reference)
    lam = cal["by_alpha"][str(args.alpha)]["lambda_hat"]
    thresholds = {"B2_accept_all": 0.0, "B5_fixed_0.9": 0.9, "frozen_0.766": FROZEN_LAMBDA}
    thresholds[f"E2_recalibrated_alpha_{args.alpha}"] = lam if lam is not None else math.inf
    res = {"reference": "checker on the full note (model-derived)", "recalibration": cal,
           "in_sample": evaluate(masked, reference, thresholds), "leave_one_out": leave_one_out(masked, reference, args.alpha)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "e2_recalibration.json").write_text(json.dumps(res, indent=1, default=_plain) + "\n")
    print(json.dumps(res, indent=1, default=_plain))


if __name__ == "__main__":
    main()
