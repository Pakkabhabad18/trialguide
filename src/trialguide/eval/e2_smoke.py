"""E2 smoke run (T38, exploratory): the frozen system on a few Indian-affiliated case reports.

    python -m trialguide.eval.e2_smoke --out outputs/e2_smoke [--n 10] [--trials 2] [--fraction 0.4]

Patients come from the E2 CALIBRATION half only (the E2 test half stays sealed), a seeded sample. For each patient the
candidate trials are the top-k of the 157 E2 trials by BM25 on the full note (a fixed candidate set, so the full and the
masked note are checked against the SAME rules; retrieval itself is not what is evaluated here). The frozen checker runs once
per rule on the full note and on the rule-targeted masked note, and the policies that never ask (B6, B2, B5) are applied to the
same checks, exactly as in baselines/run_systems.py. E2 has no expert labels, so the reference is the checker on the full note
(model-derived, like the T45 reference), and the numbers are rule recovery in the sense of eval/recovery.py:
of the rules the full-note checker settles and masking made B6 leave open, how many each policy settles the same way, wrongly,
and wrongly toward eligibility.

This is a check that the system runs on this kind of note and a first look at its behaviour. It is NOT a calibrated E2
evaluation: no recalibration, no doctor (B7), no gold labels, a handful of patients. Limits: docs/e2_limits.md. Owner: Murthy.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SEED = 20261115
SNAPSHOT = Path(__file__).resolve().parents[3] / "results" / "e2_smoke" / "e2_trials_corpus.json"
POLICIES = ("B6", "B2", "B5")


def load_trial_snapshot(path: Path = SNAPSHOT) -> list:
    """The 157 E2 trials (corpus format), a snapshot of ClinicalTrials.gov taken on 2026-10-30."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def trialgpt_form(entry: dict) -> dict:
    """An E2 corpus entry in the shape pipeline.rules_of_trial expects."""
    return {"NCTID": entry["_id"], "inclusion_criteria": entry["metadata"]["inclusion_criteria"],
            "exclusion_criteria": entry["metadata"]["exclusion_criteria"]}


def sample_patients(ids: list, n: int, seed: int = SEED) -> list:
    return sorted(random.Random(seed).sample(sorted(ids), min(n, len(ids))))


def _tokens(text: str) -> list:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def top_trials(note_text: str, trials: list, k: int) -> list:
    """The k trials whose text best matches the note (BM25); ties broken by trial id."""
    from rank_bm25 import BM25Okapi
    docs = [_tokens(" ".join([t.get("title", ""), t.get("text", ""), t["metadata"]["inclusion_criteria"],
                              t["metadata"]["exclusion_criteria"], " ".join(t["metadata"].get("diseases_list", []))]))
            for t in trials]
    scores = BM25Okapi(docs).get_scores(_tokens(note_text))
    order = sorted(range(len(trials)), key=lambda i: (-scores[i], trials[i]["_id"]))
    return [trials[i] for i in order[:k]]


def run(check, patients: dict, trials: list, k: int, fraction: float, out: Path, workers: int = 8) -> dict:
    """Run the full and the masked condition for every patient and write checks_{condition}.jsonl in the format of
    run_systems (rows with the checks and the actions of every policy)."""
    from trialguide.baselines.run_systems import to_json
    from trialguide.data.masking import mask_none, mask_rule, rule_demand, split_criteria
    from trialguide.pipeline import rules_of_trial, run_patient
    from trialguide.policy.policy import system_config
    configs = {s: system_config(s) for s in POLICIES}
    out.mkdir(parents=True, exist_ok=True)
    chosen, cond_name = {}, f"rule{round(fraction * 100)}"
    logs = {c: open(out / f"checks_{c}.jsonl", "w", encoding="utf-8") for c in ("full", cond_name)}
    for pid, pt in sorted(patients.items()):
        full_text = " ".join(pt.sentences[i] for i in sorted(pt.sentences))
        picked = top_trials(full_text, trials, k)
        chosen[pid] = [t["_id"] for t in picked]
        rules = [r for t in picked for r in rules_of_trial(trialgpt_form(t))]
        trial_rules = [split_criteria(t["metadata"]["inclusion_criteria"]) + split_criteria(t["metadata"]["exclusion_criteria"])
                       for t in picked]
        notes = {"full": mask_none(pt).visible,
                 cond_name: mask_rule(pt, rule_demand(pt.sentences, trial_rules), fraction).visible}
        for cond, note in notes.items():
            with ThreadPoolExecutor(workers) as pool:
                done = dict(zip([(r.trial_id, r.rule_id) for r in rules], pool.map(lambda r: check(note, r), rules)))
            row = {"patient_id": pid, "checks": to_json(list(done.values())), "actions": {}}
            for s, cfg in configs.items():
                res = run_patient(note, rules, cfg, lambda n, r: done[(r.trial_id, r.rule_id)])
                row["actions"][s] = to_json(res.actions)
            logs[cond].write(json.dumps(row) + "\n")
            logs[cond].flush()
        print(f"{pid}: {len(rules)} rules in {len(picked)} trials, {len(pt.sentences)} sentences", flush=True)
    for f in logs.values():
        f.close()
    return chosen


def summarise(out: Path, condition: str, n_boot: int = 5000) -> dict:
    """Behaviour of the policies against the full-note checker (the model-derived reference)."""
    from trialguide.eval.recovery import DECIDED, boot, counts, policy_finals, reference
    ref = reference(out / "checks_full.jsonl")
    masked = out / f"checks_{condition}.jsonl"
    b6 = policy_finals(masked, "B6")
    full_status = [s for rules in ref.values() for s, _ in rules.values()]
    res = {"patients": len(ref), "rules": len(full_status),
           "full_note_settled_share": sum(s in DECIDED for s in full_status) / max(len(full_status), 1),
           "full_note_status_counts": {s: full_status.count(s) for s in sorted(set(full_status))}}
    open_masked = sum(v not in DECIDED for rules in b6.values() for v in rules.values())
    res["masked_unresolved_share_b6"] = open_masked / max(sum(len(r) for r in b6.values()), 1)
    for s in ("B2", "B5"):
        c = counts(ref, b6, policy_finals(masked, s))
        res[s] = {"lost_rules": sum(v["lost"] for v in c.values()),
                  "recovered": boot(c, "recovered", "lost", n_boot), "wrong": boot(c, "wrong", "lost", n_boot),
                  "false_eligible": boot(c, "false_eligible", "lost", n_boot)}
    return res


def main(argv=None) -> None:
    from trialguide.data import e2
    from trialguide.pipeline import checker
    from trialguide.schemas import Split
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10, help="E2 calibration patients")
    ap.add_argument("--trials", type=int, default=2, help="candidate trials per patient")
    ap.add_argument("--fraction", type=float, default=0.4)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--part", default="", help="a/b: process every other sampled patient (two GPUs)")
    args = ap.parse_args(argv)

    pool = e2.load_e2_patients(args.raw, Split.CAL)             # the calibration half only
    ids = sample_patients(list(pool), args.n)
    if args.part:
        ids = ids[0::2] if args.part == "a" else ids[1::2]
    chosen = run(checker(args.model, 1), {p: pool[p] for p in ids}, load_trial_snapshot(), args.trials, args.fraction,
                 args.out, args.workers)
    (args.out / "chosen.json").write_text(json.dumps({"seed": SEED, "patients": chosen, "fraction": args.fraction,
                                                      "trials_per_patient": args.trials}, indent=1))
    res = summarise(args.out, f"rule{round(args.fraction * 100)}")
    (args.out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
