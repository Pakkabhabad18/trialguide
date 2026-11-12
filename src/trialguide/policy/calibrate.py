"""Fit the risk threshold lambda-hat on the calibration split (task T30).

Two steps, so the GPU part and the statistics part can run on different machines:

    # 1. GPU: run the frozen checker on the masked notes of the expert-labelled calibration
    #    patients, only on the rules that have an expert label (a few hundred calls)
    python -m trialguide.policy.calibrate check --mask rule --fraction 0.4 --out outputs/calibration

    # 2. CPU: fit lambda-hat for each alpha and write results/calibration_<mask>.json
    python -m trialguide.policy.calibrate fit --checks outputs/calibration/checks_rule40.jsonl

The masking must be the same procedure as on the test split (exchangeability), and the checker,
its prompts and the model must be frozen before step 1: lambda-hat is only valid for the exact
checker it was fitted on. Calibration patients get their own mask seeds (masking.derive_seed).
Owner: Prakash.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trialguide.schemas import InfoCategory, RuleCheck, RuleKind, RuleStatus, ToolCall

ALPHAS = (0.02, 0.05, 0.10)


def check_from_json(d: dict) -> RuleCheck:
    """A RuleCheck back from the JSON written by run_systems.to_json."""
    d = dict(d)
    d["kind"], d["status"] = RuleKind(d["kind"]), RuleStatus(d["status"])
    d["category"] = InfoCategory(d["category"])
    d["tool_calls"] = [ToolCall(**t) for t in d.get("tool_calls", [])]
    return RuleCheck(**d)


def read_checks(path: Path) -> dict:
    """checks JSONL (one row per patient: {"patient_id", "checks": [...]}) -> patient_id -> checks."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["patient_id"]] = [check_from_json(c) for c in row["checks"]]
    return out


def reference_to_json(reference: dict) -> list:
    return [{"patient_id": p, "trial_id": t, "rule_id": r, "status": s.value}
            for (p, t, r), s in sorted(reference.items())]


def reference_from_json(rows: list) -> dict:
    return {(r["patient_id"], r["trial_id"], r["rule_id"]): RuleStatus(r["status"]) for r in rows}


# ---------------------------------------------------------------------------
# Step 1 (GPU): checker on the calibration patients' expert-labelled rules
# ---------------------------------------------------------------------------

def run_checks(args) -> None:
    from trialguide.baselines.run_systems import to_json
    from trialguide.data.masking import candidate_rules, evidence_sets, mask_evidence, mask_rule, rule_demand
    from trialguide.data.patients import load_patients
    from trialguide.eval.expert_reference import load_reference
    from trialguide.pipeline import checker, rules_of_trial
    from trialguide.schemas import Split

    reference, report = load_reference(args.raw)
    print(f"expert reference: {report['matched_rules']} rules matched "
          f"({report['exact']} exact, {report['fuzzy']} fuzzy), {len(report['unmatched'])} unmatched")
    patients = load_patients(args.raw, Split.CAL)                 # calibration only, never test
    labelled = sorted({p for p, _, _ in reference} & set(patients))
    if args.limit:
        labelled = labelled[: args.limit]
    print(f"{len(labelled)} expert-labelled calibration patients")

    trials = {}
    with open(args.raw / "trialgpt" / "sigir" / "retrieved_trials.json", encoding="utf-8") as f:
        for entry in json.load(f):
            for label in ("0", "1", "2"):
                for t in entry.get(label, []):
                    trials.setdefault(t["NCTID"], t)
    evidence = evidence_sets(args.raw, "patient") if args.mask == "evidence" else {}
    check = checker(args.model, 1)                                 # one greedy call, logprob confidence

    name = f"{args.mask}{round(args.fraction * 100)}" if args.mask == "rule" else args.mask
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "expert_reference.json").write_text(json.dumps(
        {"report": report, "reference": reference_to_json(reference)}, indent=1))
    with open(args.out / f"checks_{name}.jsonl", "w", encoding="utf-8") as log:
        for pid in labelled:
            pt = patients[pid]
            if args.mask == "rule":
                demand = rule_demand(pt.sentences, candidate_rules(args.raw, pt.cohort, pid))
                note = mask_rule(pt, demand, args.fraction).visible
            else:
                note = mask_evidence(pt, evidence.get(pid, frozenset())).visible
            wanted = {(t, r) for p, t, r in reference if p == pid}
            rules = [r for tid in sorted({t for t, _ in wanted}) if tid in trials
                     for r in rules_of_trial(trials[tid]) if (r.trial_id, r.rule_id) in wanted]
            with ThreadPoolExecutor(args.workers) as pool:
                checks = list(pool.map(lambda r: check(note, r), rules))
            log.write(json.dumps({"patient_id": pid, "mask": name, "checks": to_json(checks)}) + "\n")
            log.flush()
            print(f"{pid}: {len(checks)} rules", flush=True)


# ---------------------------------------------------------------------------
# Step 2 (CPU): fit lambda-hat
# ---------------------------------------------------------------------------

def fit(args) -> dict:
    from trialguide.policy.risk_control import calibrate, counts_at, risk_report

    checks = read_checks(args.checks)
    ref_file = args.reference or args.checks.parent / "expert_reference.json"
    reference = reference_from_json(json.loads(Path(ref_file).read_text())["reference"])
    out = {"checks": str(args.checks), "n_patients": len(checks),
           "rules": sum(len(c) for c in checks.values()), "alphas": {}}
    for alpha in args.alphas:
        cal = calibrate(checks, reference, alpha)
        rep = risk_report(counts_at(checks, reference, cal.lambda_hat), n_boot=args.n_boot)
        out["alphas"][f"{alpha:.2f}"] = {
            "lambda_hat": None if math.isinf(cal.lambda_hat) else cal.lambda_hat,
            "accept_none": not cal.accepts_any, "certifiable": cal.certifiable,
            "risk_at_lambda": cal.risk_at_lambda, "summary": cal.summary(),
            "calibration_report": {k: (v if isinstance(v, (int, float)) else
                                       {"estimate": v.estimate, "low": v.low, "high": v.high})
                                   for k, v in rep.items()},
        }
        print(cal.summary())
    name = args.checks.stem.replace("checks_", "")
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / f"calibration_{name}.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="GPU: run the checker on the calibration patients")
    c.add_argument("--raw", type=Path, default=Path("data/raw"))
    c.add_argument("--model", default="Qwen/Qwen3-8B")
    c.add_argument("--mask", choices=("rule", "evidence"), default="rule")
    c.add_argument("--fraction", type=float, default=0.4, help="for --mask rule")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--limit", type=int, default=0, help="first N patients only (for a quick test)")
    c.add_argument("--out", type=Path, default=Path("outputs/calibration"))
    f = sub.add_parser("fit", help="CPU: fit lambda-hat from the checks")
    f.add_argument("--checks", type=Path, required=True)
    f.add_argument("--reference", type=Path, default=None)
    f.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    f.add_argument("--n-boot", type=int, default=10000)
    f.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    run_checks(args) if args.cmd == "check" else fit(args)


if __name__ == "__main__":
    main()
