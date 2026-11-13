"""Recovery of the full-information result (RQ1), from the saved runs; no model calls.

The reference is the checker on the FULL note (model-derived, proposal Section 5 source b):
a rule is "lost by masking" if the checker settled it (met / not met) on the full note but B6
leaves it unresolved on the masked note. For each system on the masked note:

- recovered:      lost rules the system ends with the same status as on the full note;
- wrong:          lost rules it settles with a different status;
- false-eligible: wrong ones that push the patient toward eligibility (the harmful direction);
- ranking agreement: Kendall's tau between the system's trial scores and B6's full-note scores,
  per patient, averaged.

Shares are ratios of sums over patients, with patient-level bootstrap 95% CIs.

    python -m trialguide.eval.recovery --run outputs/dev --asking outputs/dev/final_lam

Owner: Prateek.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DECIDED = ("met", "not_met")
ACCEPTED = ("use_stated", "accept_inference")


def favours_eligibility(kind: str, status: str) -> bool:
    return (kind == "inclusion" and status == "met") or (kind == "exclusion" and status == "not_met")


def reference(checks_full: Path) -> dict:
    """patient -> 'trial|rule' -> (status, kind), the checker's raw output on the full note."""
    ref = {}
    for line in open(checks_full):
        row = json.loads(line)
        ref[row["patient_id"]] = {f"{c['trial_id']}|{c['rule_id']}": (c["status"], c["kind"])
                                  for c in row["checks"]}
    return ref


def policy_finals(checks_file: Path, policy: str) -> dict:
    """patient -> rule -> final status for a non-asking policy: accepted checks keep their status."""
    finals = {}
    for line in open(checks_file):
        row = json.loads(line)
        accepted = {f"{a['trial_id']}|{a['rule_id']}" for a in row["actions"][policy] if a["action"] in ACCEPTED}
        finals[row["patient_id"]] = {
            key: (c["status"] if key in accepted else "not_enough_info")
            for c in row["checks"] for key in [f"{c['trial_id']}|{c['rule_id']}"]}
    return finals


def asking_finals(path: Path) -> dict:
    return {pid: {k: v[0] for k, v in rules.items()} for pid, rules in json.load(open(path)).items()}


def counts(ref: dict, lost_from: dict, final: dict) -> dict:
    """Per patient: lost, recovered, wrong, false_eligible."""
    out = {}
    for pid, rules in ref.items():
        c = {"lost": 0, "recovered": 0, "wrong": 0, "false_eligible": 0}
        for key, (status, kind) in rules.items():
            if status not in DECIDED or lost_from[pid].get(key) in DECIDED:
                continue
            c["lost"] += 1
            got = final.get(pid, {}).get(key, "not_enough_info")
            if got == status:
                c["recovered"] += 1
            elif got in DECIDED:
                c["wrong"] += 1
                c["false_eligible"] += favours_eligibility(kind, got)
        out[pid] = c
    return out


def kendall_tau(a: dict, b: dict) -> float:
    keys = sorted(set(a) & set(b))
    concordant = discordant = ties_a = ties_b = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            da, db = a[keys[i]] - a[keys[j]], b[keys[i]] - b[keys[j]]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = concordant + discordant
    denom = ((n0 + ties_a) * (n0 + ties_b)) ** 0.5
    return (concordant - discordant) / denom if denom else 1.0


def boot(per_patient: dict, num: str, den: str, n: int = 5000, seed: int = 0) -> list:
    ids = sorted(per_patient)
    rng = random.Random(seed)

    def ratio(sample):
        d = sum(per_patient[i][den] for i in sample)
        return sum(per_patient[i][num] for i in sample) / d if d else 0.0
    stats = sorted(ratio([rng.choice(ids) for _ in ids]) for _ in range(n))
    return [ratio(ids), stats[int(0.025 * n)], stats[int(0.975 * n) - 1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="folder with systems/ (run_systems output)")
    ap.add_argument("--asking", type=Path, required=True, help="run_asking output with final_*.json")
    ap.add_argument("--masked", default="rule40")
    args = ap.parse_args()

    ref = reference(args.run / "systems" / "checks_full.jsonl")
    masked = args.run / "systems" / f"checks_{args.masked}.jsonl"
    b6 = policy_finals(masked, "B6")
    full_scores = json.load(open(args.run / "systems" / "rankings_B6_full.json"))

    systems = {p: (policy_finals(masked, p), json.load(open(args.run / "systems" / f"rankings_{p}_{args.masked}.json")))
               for p in ("B6", "B2", "B5")}
    for f in sorted(args.asking.glob("final_*.json")):
        name = f.stem[len("final_"):]
        systems[name] = (asking_finals(f), json.load(open(args.asking / f"rankings_{name}.json")))

    table = {}
    for name, (final, scores) in systems.items():
        c = counts(ref, b6, final)
        taus = [kendall_tau(scores[p], full_scores[p]) for p in scores if p in full_scores]
        table[name] = {
            "lost_rules": sum(v["lost"] for v in c.values()),
            "recovered": boot(c, "recovered", "lost"),
            "wrong": boot(c, "wrong", "lost"),
            "false_eligible": boot(c, "false_eligible", "lost"),
            "kendall_tau_vs_full": sum(taus) / len(taus),
        }
    (args.asking / "recovery.json").write_text(json.dumps(table, indent=1) + "\n")
    print(f"{'system':8} {'recovered [95% CI]':24} {'wrong':7} {'false-elig':10} tau-vs-full")
    for name, t in table.items():
        r = t["recovered"]
        print(f"{name:8} {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]   {t['wrong'][0]:.3f}   {t['false_eligible'][0]:.3f}"
              f"      {t['kendall_tau_vs_full']:.3f}")


if __name__ == "__main__":
    main()
