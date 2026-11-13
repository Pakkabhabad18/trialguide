"""Harmful-inference risk on the expert-labelled rules: calibrate lambda (T30) or report it.

The TrialGPT criterion annotations label single rules of some SIGIR patient-trial pairs (26
patients in calibration, 26 in test). For each labelled rule this script runs the checker on the
patient's rule-masked note (the same masking as every other run) and compares the inferences
with the expert label, so the reference is always expert judgement (proposal Section 5, source a).

    --split cal:   lambda_hat = smallest threshold whose conformal bound on the per-patient
                   harmful-inference rate is <= alpha. Writes {out}/calibration.json.
    --split test:  the harmful-inference rate, coverage and useful coverage at the given
                   thresholds (B2: 0, B5: 0.9, B7: lambda_hat), with patient-level bootstrap CIs.
                   Needs --allow-test: the test split is opened once, after everything is frozen.

The checker sees only the rule text and the masked note, never the label.
Owner of this file: Prateek (with Prakash's risk_control, T30).
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHECKER_MODEL = "Qwen/Qwen3-8B"
# expert label -> status of the condition in the rule text (schemas.RuleStatus)
EXPERT = {"included": "met", "not included": "not_met", "excluded": "met", "not excluded": "not_met",
          "not enough information": "not_enough_info", "not applicable": "not_applicable"}


def to_json(obj):
    from trialguide.baselines.run_asking import to_json as convert
    return convert(obj)


def plain(obj):
    """json.dumps default: bootstrap results (named tuples or dataclasses) as objects."""
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    return to_json(obj) if not isinstance(obj, (str, int, float)) else str(obj)


def main():
    import pandas as pd

    from trialguide.data.masking import candidate_rules, mask_rule, rule_demand
    from trialguide.data.patients import load_patients
    from trialguide.pipeline import checker
    from trialguide.policy.risk_control import calibrate, counts_at, risk_report
    from trialguide.schemas import RuleKind, RuleStatus, Split, TrialRule

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=["cal", "test"], required=True)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--fraction", type=float, default=0.4)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--thresholds", nargs="*", default=[], metavar="NAME=VALUE",
                    help="test only, e.g. B2=0 B5=0.9 B7=0.93")
    ap.add_argument("--no-tools", action="store_true", help="checker without tools (A5)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("the test split is opened once, after the system is frozen: pass --allow-test")
    args.out.mkdir(parents=True, exist_ok=True)

    split = Split(args.split)
    patients = load_patients(args.raw, split, allow_test=args.allow_test)
    ann = pd.read_parquet(args.raw / "annotations" / "criterion_annotations.parquet")
    ann = ann[ann.patient_id.isin(patients)]

    notes, rules, reference = {}, {}, {}
    for pid, rows in ann.groupby("patient_id"):
        pt = patients[pid]
        notes[pid] = mask_rule(pt, rule_demand(pt.sentences, candidate_rules(args.raw, pt.cohort, pid)),
                               args.fraction).visible
        rules[pid] = []
        for row in rows.itertuples():
            kind = RuleKind.INCLUSION if row.criterion_type == "inclusion" else RuleKind.EXCLUSION
            rule = TrialRule(row.trial_id, f"ann_{row.annotation_id}", kind, row.criterion_text)
            rules[pid].append(rule)
            reference[(pid, rule.trial_id, rule.rule_id)] = RuleStatus(EXPERT[row.expert_eligibility])

    check = checker(CHECKER_MODEL, n_samples=1, tools=not args.no_tools)
    jobs = [(pid, r) for pid in rules for r in rules[pid]]
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda job: check(notes[job[0]], job[1]), jobs))
    checks = {pid: [] for pid in rules}
    for (pid, _), c in zip(jobs, results):
        checks[pid].append(c)
    with open(args.out / f"checks_{args.split}.jsonl", "w") as f:
        for pid, cs in checks.items():
            f.write(json.dumps({"patient_id": pid, "checks": to_json(cs)}) + "\n")

    info = {"split": args.split, "checker": CHECKER_MODEL, "tools": not args.no_tools,
            "mask": "rule", "fraction": args.fraction, "reference": "expert labels",
            "patients": len(checks), "rules": len(jobs)}
    if args.split == "cal":
        cal = calibrate(checks, reference, args.alpha)
        info.update({"alpha": cal.alpha, "lambda_hat": cal.lambda_hat if math.isfinite(cal.lambda_hat) else "inf",
                     "certifiable": cal.certifiable, "risk_at_lambda": cal.risk_at_lambda,
                     "curve": [[lam if math.isfinite(lam) else "inf", risk, b] for lam, risk, b in cal.curve]})
        for name, lam in (("B2", 0.0), ("B5", 0.9), ("B7", cal.lambda_hat)):
            info.setdefault("at_threshold", {})[name] = risk_report(counts_at(checks, reference, lam))
        (args.out / "calibration.json").write_text(json.dumps(info, indent=1, default=plain) + "\n")
        print(json.dumps({k: v for k, v in info.items() if k != "curve"}, indent=1, default=plain))
    else:
        for item in args.thresholds:
            name, value = item.split("=")
            lam = math.inf if value == "inf" else float(value)
            info.setdefault("at_threshold", {})[name] = {"threshold": value,
                                                          **risk_report(counts_at(checks, reference, lam))}
        (args.out / "risk_test.json").write_text(json.dumps(info, indent=1, default=plain) + "\n")
        print(json.dumps(info, indent=1, default=plain))


if __name__ == "__main__":
    main()
