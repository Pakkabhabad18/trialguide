"""B6, TrialGPT-style abstention (task T16): the TrialGPT pipeline on masked notes.

B6 is plain TrialGPT (T4) given only the visible part of each note. It cannot infer or
ask, so a rule whose fact was hidden ends as "not enough information". The same pipeline
on the full note is the reference; the gap between the two is what masking costs
TrialGPT, and what B7 tries to recover.

    python -m trialguide.baselines.b6 --raw data/raw --out outputs/b6/data [--n 5] [--per-label 2]

Writes one TrialGPT-layout data folder per condition ("full" and e.g. "rule40"), holding
only the chosen patients and trials, with the note replaced by the visible note:
    {out}/{condition}/{cohort}/retrieved_trials.json and queries.jsonl
plus {out}/manifest.json (patients, trials, how many sentences each mask hid). The T4
scripts then run unchanged on each folder (scripts/run_b6.sh). Both conditions use the
same patients and trials, so their scores differ only by the masking.
Owner: Prateek (T16).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED = 20261030
LABELS = ("0", "1", "2")


def choose_patients(dev: dict, n: int, seed: int = SEED) -> list:
    """n development patients, spread over cohorts in proportion to their size (largest remainder).
    dev: patient_id -> cohort."""
    by_cohort = {}
    for p, c in sorted(dev.items()):
        by_cohort.setdefault(c, []).append(p)
    exact = {c: n * len(ps) / len(dev) for c, ps in by_cohort.items()}
    take = {c: int(x) for c, x in exact.items()}
    for c in sorted(exact, key=lambda c: (take[c] - exact[c], c))[: n - sum(take.values())]:
        take[c] += 1
    rng = random.Random(seed)
    return sorted(p for c in sorted(by_cohort) for p in rng.sample(by_cohort[c], take[c]))


def cap_trials(entry: dict, per_label: int) -> dict:
    """The first per_label trials of each expert label (TrialGPT's own order), as in the T4 smoke test."""
    out = dict(entry)
    for label in LABELS:
        if label in entry:
            out[label] = entry[label][:per_label]
    return out


def visible_text(sentences: dict, drop: str = "") -> str:
    """The visible sentences in their original order, as one note. `drop` removes a sentence the
    TrialGPT scripts add back themselves (the SIGIR consent sentence), so it is not doubled."""
    return " ".join(s for _, s in sorted(sentences.items()) if s != drop)


def write_condition(folder: Path, entries: dict, notes: dict) -> None:
    """entries: cohort -> list of retrieved_trials entries; notes: patient_id -> note text to use."""
    for cohort, rows in entries.items():
        d = folder / cohort
        d.mkdir(parents=True, exist_ok=True)
        rows = [dict(r, patient=notes[r["patient_id"]]) for r in rows]
        (d / "retrieved_trials.json").write_text(json.dumps(rows, indent=1))
        with open(d / "queries.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({"_id": r["patient_id"], "text": r["patient"]}) + "\n")


def main():
    # Imported here: masking and patients come from T10, which may not be on main yet
    from trialguide.data.masking import candidate_rules, mask_none, mask_rule, rule_demand
    from trialguide.data.patients import SIGIR_CONSENT, load_patients
    from trialguide.schemas import Split

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", type=Path, default=Path("data/raw"), help="folder holding trialgpt/{cohort}/")
    ap.add_argument("--out", type=Path, default=Path("outputs/b6/data"))
    ap.add_argument("--n", type=int, default=5, help="development patients")
    ap.add_argument("--per-label", type=int, default=2, help="trials per expert label (0/1/2)")
    ap.add_argument("--fraction", type=float, default=0.4, help="share of sentences rule-targeted masking hides")
    args = ap.parse_args()

    patients = load_patients(args.raw, Split.DEV)              # never the test split
    chosen = choose_patients({p: pt.cohort for p, pt in patients.items()}, args.n)
    masked_name = f"rule{round(args.fraction * 100)}"

    entries, notes = {}, {"full": {}, masked_name: {}}
    manifest = {"seed": SEED, "split": "dev", "mask": "rule", "fraction": args.fraction,
                "per_label": args.per_label, "patients": {}}
    for p in chosen:
        pt = patients[p]
        retrieved = {e["patient_id"]: e for e in json.load(open(args.raw / "trialgpt" / pt.cohort / "retrieved_trials.json"))}
        entry = cap_trials(retrieved[p], args.per_label)
        entries.setdefault(pt.cohort, []).append(entry)

        demand = rule_demand(pt.sentences, candidate_rules(args.raw, pt.cohort, p))
        masked = mask_rule(pt, demand, args.fraction)
        notes["full"][p] = visible_text(mask_none(pt).visible.sentences, SIGIR_CONSENT)
        notes[masked_name][p] = visible_text(masked.visible.sentences, SIGIR_CONSENT)
        manifest["patients"][p] = {
            "cohort": pt.cohort, "sentences": len(pt.sentences), "hidden": len(masked.hidden_ids),
            "hidden_ids": list(masked.hidden_ids),       # scoring and logs only
            "trials": {lab: [t["NCTID"] for t in entry.get(lab, [])] for lab in LABELS},
        }

    for name, texts in notes.items():
        write_condition(args.out / name, entries, texts)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    pairs = sum(len(v) for m in manifest["patients"].values() for v in m["trials"].values())
    print(f"{len(chosen)} patients {chosen}, {pairs} pairs per condition; conditions: {list(notes)}")


if __name__ == "__main__":
    main()
