"""Run the systems that never ask (B6, B2, B5) with our checker, on the patients and trials of a B6 manifest.

    python -m trialguide.baselines.run_systems --manifest outputs/b6/data/manifest.json

These systems differ only in the policy (T19), so the checker (T12) runs ONCE per patient and
condition and every policy is applied to the same rule checks. Same patients, trials and masks
as the TrialGPT B6 run (trialguide.baselines.b6), so the two B6 versions differ only in the
checker: TrialGPT's prompts (a) vs. our checker and policy (b).

Writes {out}/summary.json (NDCG@10, P@10, rates per system and condition), rankings per system
and condition, and every rule check to checks_{condition}.jsonl.
Owner of this file: Prateek (T16).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEMS = ("B6", "B2", "B5")
LABELS = ("0", "1", "2")


def to_json(obj):
    """RuleChecks / Actions / enums -> plain JSON."""
    if dataclasses.is_dataclass(obj):
        return {k: to_json(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    return getattr(obj, "value", obj)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    from trialguide.data.masking import candidate_rules, mask_none, mask_rule, rule_demand
    from trialguide.data.patients import load_patients
    from trialguide.eval.ranking import per_patient_metrics
    from trialguide.pipeline import checker, rules_of_trial, run_patient
    from trialguide.policy.policy import system_config
    from trialguide.schemas import ActionType, Split

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n-samples", type=int, default=1, help="checker samples per rule (greedy: 1)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("outputs/systems"))
    args = ap.parse_args()

    configs = {s: system_config(s) for s in args.systems}
    asking = [s for s, c in configs.items() if c.allow_asking]
    assert not asking, f"{asking} ask questions and need the doctor; not run here"
    m = json.loads(args.manifest.read_text())
    patients = load_patients(args.raw, Split.DEV)            # never the test split
    check = checker(args.model, args.n_samples)
    args.out.mkdir(parents=True, exist_ok=True)

    # Same masks as the TrialGPT B6 run: same function, seed and fraction
    conditions = {
        "full": lambda pt: mask_none(pt).visible,
        f"rule{round(m['fraction'] * 100)}": lambda pt: mask_rule(
            pt, rule_demand(pt.sentences, candidate_rules(args.raw, pt.cohort, pt.patient_id)),
            m["fraction"]).visible,
    }
    summary = {"model": args.model, "n_samples": args.n_samples, "manifest": str(args.manifest),
               "confidence": "logprob (probability of the status)" if args.n_samples == 1 else "votes",
               "policies": {s: dataclasses.asdict(c) for s, c in configs.items()}, "results": {}}
    gold = {}
    for cond, make_note in conditions.items():
        rankings = {s: {} for s in configs}
        rates = {s: {"accepted_inference": 0, "abstained": 0, "rules": 0} for s in configs}
        checker_counts, log = {}, open(args.out / f"checks_{cond}.jsonl", "w")
        for pid, info in m["patients"].items():
            pt = patients[pid]
            entry = {e["patient_id"]: e for e in json.load(
                open(args.raw / "trialgpt" / info["cohort"] / "retrieved_trials.json"))}[pid]
            trials = [t for lab in LABELS for t in entry.get(lab, []) if t["NCTID"] in info["trials"][lab]]
            gold[pid] = {t["NCTID"]: int(lab) for lab in LABELS for t in entry.get(lab, [])
                         if t["NCTID"] in info["trials"][lab]}
            rules = [r for t in trials for r in rules_of_trial(t)]
            note = make_note(pt)

            # Checker once, concurrently (the client batches the calls on the GPU)
            with ThreadPoolExecutor(args.workers) as pool:
                done = dict(zip([(r.trial_id, r.rule_id) for r in rules],
                                pool.map(lambda r: check(note, r), rules)))
            for c in done.values():
                key = f"{c.category.value}/{c.status.value}"
                checker_counts[key] = checker_counts.get(key, 0) + 1
            row = {"patient_id": pid, "checks": to_json(list(done.values())), "actions": {}}

            for s, cfg in configs.items():                 # every policy on the same checks
                res = run_patient(note, rules, cfg, lambda n, r: done[(r.trial_id, r.rule_id)])
                rankings[s][pid] = res.scores
                kinds = [a.action for a in res.actions]
                rates[s]["accepted_inference"] += kinds.count(ActionType.ACCEPT_INFERENCE)
                rates[s]["abstained"] += sum(k not in (ActionType.USE_STATED, ActionType.ACCEPT_INFERENCE)
                                             for k in kinds)
                rates[s]["rules"] += len(kinds)
                row["actions"][s] = to_json(res.actions)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"{cond} {pid}: {len(rules)} rules in {len(trials)} trials", flush=True)
        log.close()

        for s in configs:
            metrics = per_patient_metrics(rankings[s], gold)
            (args.out / f"rankings_{s}_{cond}.json").write_text(json.dumps(rankings[s], indent=1))
            n = rates[s]["rules"]
            summary["results"].setdefault(s, {})[cond] = {
                "patients": len(metrics),
                "NDCG@10": mean(v["ndcg"] for v in metrics.values()),
                "P@10": mean(v["p10"] for v in metrics.values()),
                "accepted_inference_rate": rates[s]["accepted_inference"] / n if n else None,
                "abstain_rate": rates[s]["abstained"] / n if n else None,
                "per_patient": metrics,
            }
        summary.setdefault("checker_outputs", {})[cond] = checker_counts

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({s: {c: {k: v[k] for k in ("NDCG@10", "P@10", "accepted_inference_rate", "abstain_rate")}
                          for c, v in r.items()} for s, r in summary["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
