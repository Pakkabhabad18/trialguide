"""E1 training data (T23): teach Qwen3-8B to check rules on a masked note the way the teacher
does on the full note, and to say "not enough information" when the needed sentence is hidden.

    python -m trialguide.finetune.data --manifest results/dev_run/manifest.json \
        --teacher outputs/dev_run/checks_full.jsonl --out outputs/e1

Inputs
- the development manifest (patients, their candidate trials, and the hidden sentence ids of the
  rule-targeted 40% mask used in every dev run);
- the teacher: the frozen checker (Qwen3-8B, one greedy call) run on the FULL note of the same
  patients and trials (run_systems writes it as checks_full.jsonl).

One training example = one (patient, trial, rule). The model input is exactly what the checker
sends at test time (same system prompt, same user message, tools off), with the MASKED note. The
target is a JSON answer built from the teacher's full-note answer:

    visible     teacher decided the rule (stated / inferred) and every evidence sentence it cited
                is still visible  -> the teacher's answer, unchanged
    hidden      teacher decided the rule, but every evidence sentence it cited is hidden
                -> not_enough_info / missing, with the teacher's fact_key (what to ask about)
    unresolved  the teacher itself could not decide on the full note -> its answer, unchanged
    n/a         teacher said not_applicable -> its answer, unchanged

Dropped (counted in the report): evidence partly visible and partly hidden (unclear target),
answers that used a tool result (the tuned model is trained and tested with tools off), and
answers that contradict the expert trial label (the trial is labelled eligible, but the teacher
says the patient fails an inclusion rule or meets an exclusion rule). These are model-made
labels, not expert labels, and are reported as such (proposal, E1).

Only development patients are used. Calibration and test patients are never read here, so the
tuned model is evaluated on patients it has never seen.

Outputs: {out}/train.jsonl and {out}/val.jsonl (a few held-out dev patients, for the validation
loss only), one {"messages": [system, user, assistant], "meta": {...}} per line, and
{out}/data_report.json with the counts.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from trialguide.checker.rule_checker import SYSTEM_PROMPT, user_prompt
from trialguide.schemas import RuleKind, RuleStatus, VisibleNote

DECIDED = ("met", "not_met")
# Trial-label bucket that means "eligible" in the TrialGPT files (TREC and SIGIR alike)
ELIGIBLE = "2"

# How many examples of each type to keep at most (the "unresolved" type is the most common one
# and would otherwise teach the model to abstain on everything)
DEFAULT_CAPS = {"visible": 700, "hidden": 700, "unresolved": 300, "not_applicable": 100}


def visible_note(patient, hidden_ids) -> VisibleNote:
    """The masked note exactly as the dev runs saw it: the full note minus the manifest's hidden
    sentence ids (same ids as masking.mask_rule; checked for all 37 dev patients)."""
    hidden = set(hidden_ids)
    return VisibleNote(patient_id=patient.patient_id, split=patient.split,
                       sentences={i: s for i, s in patient.sentences.items() if i not in hidden})


def answer_json(status, category, evidence_ids, explanation, fact_key) -> str:
    """The target text: the JSON object the checker prompt asks for, keys in the prompt's order."""
    return json.dumps({"status": status, "category": category, "evidence_ids": list(evidence_ids),
                       "explanation": explanation, "fact_key": fact_key}, ensure_ascii=False)


def missing_explanation(fact_key) -> str:
    """Short reason for a 'hidden' target. Names the missing fact so the reasoning stays useful."""
    if fact_key:
        return (f"The visible note does not mention {fact_key.replace('_', ' ')}, "
                "so this rule cannot be decided without asking.")
    return "The visible note has no sentence that settles this rule, so it cannot be decided."


def excludes(kind: str, status: str) -> bool:
    """True if this rule status rules the patient out of the trial."""
    return (kind == RuleKind.INCLUSION.value and status == RuleStatus.NOT_MET.value) or \
           (kind == RuleKind.EXCLUSION.value and status == RuleStatus.MET.value)


def target_for(teacher: dict, visible_ids: set, trial_label: str):
    """(example type, target JSON text) for one teacher answer, or (drop reason, None).

    `teacher` is one check from checks_full.jsonl (a RuleCheck as JSON)."""
    status, category = teacher["status"], teacher["category"]
    evidence = list(teacher.get("evidence_ids") or [])
    fact_key = teacher.get("fact_key")

    if teacher.get("tool_calls"):
        return "drop_tool", None
    if trial_label == ELIGIBLE and status in DECIDED and excludes(teacher["kind"], status):
        return "drop_contradicts_trial_label", None

    if status == RuleStatus.NOT_APPLICABLE.value:
        return "not_applicable", answer_json(status, category, [], teacher.get("explanation", ""), None)
    if status not in DECIDED:
        # the teacher could not decide even with the full note: keep its "not enough information"
        return "unresolved", answer_json(RuleStatus.NOT_ENOUGH_INFO.value, category, [],
                                         teacher.get("explanation", ""), fact_key)

    seen = [i for i in evidence if i in visible_ids]
    if evidence and len(seen) == len(evidence):
        return "visible", answer_json(status, category, evidence, teacher.get("explanation", ""), fact_key)
    if evidence and not seen:
        return "hidden", answer_json(RuleStatus.NOT_ENOUGH_INFO.value, "missing", [],
                                     missing_explanation(fact_key), fact_key)
    return "drop_mixed_evidence", None


def messages_for(note: VisibleNote, rule, target: str) -> list:
    """System, user and assistant messages; system and user are the checker's own (tools off)."""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(rule, note.text(), "", 0, 1)},
            {"role": "assistant", "content": target}]


def read_teacher(path: Path) -> dict:
    """checks_full.jsonl -> {(patient, trial, rule): check dict}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for c in row["checks"]:
                out[(row["patient_id"], c["trial_id"], c["rule_id"])] = c
    return out


def build(manifest: dict, teacher: dict, patients: dict, trials: dict, caps: dict,
          n_val_patients: int = 4, seed: int = 0, max_val: int = 150):
    """All examples, split by patient into train and val, each type capped. Returns
    (train, val, report)."""
    from trialguide.pipeline import rules_of_trial

    rng = random.Random(seed)
    pids = sorted(manifest["patients"])
    val_pids = set(rng.sample(pids, n_val_patients)) if n_val_patients else set()
    counts, examples = Counter(), {"train": [], "val": []}
    for pid in pids:
        info = manifest["patients"][pid]
        note = visible_note(patients[pid], info["hidden_ids"])
        for label, tids in info["trials"].items():
            for tid in tids:
                for rule in rules_of_trial(trials[tid]):
                    t = teacher.get((pid, tid, rule.rule_id))
                    if t is None:
                        counts["drop_no_teacher"] += 1
                        continue
                    kind, target = target_for(t, set(note.sentences), label)
                    counts[kind] += 1
                    if target is None:
                        continue
                    examples["val" if pid in val_pids else "train"].append({
                        "messages": messages_for(note, rule, target),
                        "meta": {"patient_id": pid, "trial_id": tid, "rule_id": rule.rule_id,
                                 "type": kind, "trial_label": label}})

    # cap each type (training split only), keeping a seeded random subset
    train = []
    for kind, cap in caps.items():
        group = [e for e in examples["train"] if e["meta"]["type"] == kind]
        rng.shuffle(group)
        train.extend(group[:cap])
    rng.shuffle(train)
    # a small validation set: on a T4 every evaluation pass costs minutes
    val = examples["val"]
    rng.shuffle(val)
    val = val[:max_val]
    report = {"teacher_answers_by_type": dict(counts),
              "train": dict(Counter(e["meta"]["type"] for e in train)),
              "val": dict(Counter(e["meta"]["type"] for e in val)),
              "val_patients": sorted(val_pids), "caps": caps, "seed": seed}
    return train, val, report


def load_trials(raw: Path) -> dict:
    """NCT id -> trial dict, from the candidate-trial files of all three cohorts."""
    trials = {}
    for cohort in ("sigir", "trec_2021", "trec_2022"):
        with open(raw / "trialgpt" / cohort / "retrieved_trials.json", encoding="utf-8") as f:
            for entry in json.load(f):
                for bucket in ("0", "1", "2"):
                    for t in entry.get(bucket, []):
                        trials.setdefault(t["NCTID"], t)
    return trials


def write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None) -> None:
    from trialguide.data.patients import load_patients
    from trialguide.schemas import Split

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=Path("results/dev_run/manifest.json"))
    ap.add_argument("--teacher", type=Path, required=True, help="checks_full.jsonl of the dev run")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--val-patients", type=int, default=4)
    ap.add_argument("--max-val", type=int, default=150, help="validation examples kept at most")
    ap.add_argument("--seed", type=int, default=0)
    for kind, cap in DEFAULT_CAPS.items():
        ap.add_argument(f"--max-{kind.replace('_', '-')}", type=int, default=cap)
    ap.add_argument("--out", type=Path, default=Path("outputs/e1"))
    args = ap.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("split") != "dev":
        raise SystemExit("E1 trains on development patients only")
    caps = {k: getattr(args, f"max_{k}") for k in DEFAULT_CAPS}
    train, val, report = build(manifest, read_teacher(args.teacher), load_patients(args.raw, Split.DEV),
                               load_trials(args.raw), caps, args.val_patients, args.seed,
                               args.max_val)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "val.jsonl", val)
    report.update({"manifest": str(args.manifest), "teacher": str(args.teacher)})
    (args.out / "data_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
