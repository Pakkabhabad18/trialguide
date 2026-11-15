"""Automatic checks of the full-note checker against the experts (T45 reference audit, T44 evidence).

    python -m trialguide.eval.reference_audit --checks results/test_run/checks_full.jsonl --out results/reference_audit

Two questions, answered on the expert-labelled rules of the patients in the checks file:

1. Reference (T45): the "model-derived reference" is the checker's judgement on the FULL note. How often
   does it agree with the expert label? Reported overall and on rules the experts settled
   (met / not met), including how often it is wrong toward eligibility and how often it settles a rule
   the experts say the note does not settle.
2. Evidence (T44): when the checker cites sentences for a rule the experts settled from evidence,
   how many cited sentences are the experts' (precision), how many expert sentences are cited
   (recall), and are the cited ids valid sentence numbers of the note?

This is a descriptive, post-hoc look at saved outputs of the frozen system: nothing here changes a
prompt, threshold or model. It is automatic, not the 200-rule / 50-inference hand audit of the plan.
Expert evidence exists only for the SIGIR patients with rule labels. Owner: Murthy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trialguide.eval.expert_reference import expert_status, match_rule
from trialguide.eval.stats import paired_bootstrap, ratio_of_sums
from trialguide.schemas import RuleKind, RuleStatus, favours_eligibility

DECIDED = ("met", "not_met")


def expert_rows(raw: Path) -> dict:
    """{(patient, trial, rule_id): {"status", "evidence", "n_sentences", "kind"}} for expert rows that
    match one rule of the trial; rows whose labels disagree for the same rule are dropped."""
    import pandas as pd
    from trialguide.pipeline import rules_of_trial
    ann = pd.read_parquet(Path(raw) / "annotations" / "criterion_annotations.parquet")
    rules_by_trial = {}
    with open(Path(raw) / "trialgpt" / "sigir" / "retrieved_trials.json", encoding="utf-8") as f:
        for entry in json.load(f):
            for label in ("0", "1", "2"):
                for trial in entry.get(label, []):
                    rules_by_trial.setdefault(trial["NCTID"], rules_of_trial(trial))
    out, dropped = {}, set()
    for r in ann.itertuples():
        kind = RuleKind(r.criterion_type.strip().lower())
        rule, _ = match_rule(r.criterion_text or "", [x for x in rules_by_trial.get(r.trial_id, []) if x.kind == kind])
        if rule is None:
            continue
        key = (r.patient_id, r.trial_id, rule.rule_id)
        row = {"status": expert_status(kind, r.expert_eligibility).value, "kind": kind.value,
               "evidence": set(json.loads(r.expert_sentences)) if isinstance(r.expert_sentences, str) else set(),
               "n_sentences": len(r.note.split("\n"))}
        if key in out and out[key]["status"] != row["status"]:
            dropped.add(key)
        out.setdefault(key, row)
    return {k: v for k, v in out.items() if k not in dropped}


def read_checks(path: Path) -> dict:
    """checks JSONL -> {(patient, trial, rule_id): check dict}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for c in row["checks"]:
                out[(row["patient_id"], c["trial_id"], c["rule_id"])] = c
    return out


def audit_rows(experts: dict, checks: dict) -> list:
    """One record per expert-labelled rule that the checks file also covers."""
    rows = []
    for key, e in sorted(experts.items()):
        c = checks.get(key)
        if c is None:
            continue
        cited, exp = set(c.get("evidence_ids") or []), e["evidence"]
        decided = c["status"] in DECIDED
        e_definite = e["status"] in DECIDED
        rows.append({
            "patient": key[0], "agree": c["status"] == e["status"],
            "e_definite": e_definite, "e_nei": e["status"] == "not_enough_info",
            "decided": decided, "right": decided and c["status"] == e["status"],
            "toward_eligibility_wrong": decided and c["status"] != e["status"] and e_definite
            and favours_eligibility(RuleKind(c["kind"]), RuleStatus(c["status"])),
            "settles_nei": decided and e["status"] == "not_enough_info",
            # evidence is compared only where both sides have it: experts settled it from sentences
            # and the checker decided and cited something
            "ev": e_definite and bool(exp) and decided and bool(cited),
            "cited": len(cited), "expert": len(exp), "overlap": len(cited & exp),
            "invalid_cites": sum(not 0 <= i < e["n_sentences"] for i in cited),
            "exact": cited == exp, "any_overlap": bool(cited & exp)})
    return rows


def _by_patient(rows: list, fields: dict) -> tuple:
    patients = sorted({r["patient"] for r in rows})
    arrays = {k: np.zeros(len(patients)) for k in fields}
    index = {p: i for i, p in enumerate(patients)}
    for r in rows:
        for k, f in fields.items():
            arrays[k][index[r["patient"]]] += f(r)
    return patients, arrays


def summarise(rows: list, n_boot: int = 5000, seed: int = 0) -> dict:
    """Pooled rates over rules with patient-level bootstrap 95% CIs."""
    fields = {"n": lambda r: 1, "agree": lambda r: r["agree"],
              "definite": lambda r: r["e_definite"], "decided_def": lambda r: r["e_definite"] and r["decided"],
              "right_def": lambda r: r["e_definite"] and r["right"],
              "wrong_elig": lambda r: r["toward_eligibility_wrong"],
              "nei": lambda r: r["e_nei"], "settles_nei": lambda r: r["settles_nei"],
              "ev_n": lambda r: r["ev"], "ev_cited": lambda r: r["cited"] if r["ev"] else 0,
              "ev_expert": lambda r: r["expert"] if r["ev"] else 0, "ev_overlap": lambda r: r["overlap"] if r["ev"] else 0,
              "ev_exact": lambda r: r["ev"] and r["exact"], "ev_any": lambda r: r["ev"] and r["any_overlap"],
              "ev_invalid": lambda r: r["invalid_cites"] if r["ev"] else 0}
    patients, a = _by_patient(rows, fields)
    ci = lambda num, den: paired_bootstrap(a, ratio_of_sums(num, den), n_boot, seed)
    return {
        "patients": len(patients), "rules": int(a["n"].sum()),
        "agreement_all": ci("agree", "n"),
        "expert_settled_rules": int(a["definite"].sum()),
        "checker_settles_them": ci("decided_def", "definite"),
        "accuracy_when_settled_by_both": ci("right_def", "decided_def"),
        "wrong_toward_eligibility": ci("wrong_elig", "definite"),
        "expert_nei_rules": int(a["nei"].sum()),
        "checker_settles_expert_nei": ci("settles_nei", "nei"),
        "evidence_rules": int(a["ev_n"].sum()),
        "evidence_precision": ci("ev_overlap", "ev_cited"),
        "evidence_recall": ci("ev_overlap", "ev_expert"),
        "evidence_exact_match": ci("ev_exact", "ev_n"),
        "evidence_any_overlap": ci("ev_any", "ev_n"),
        "invalid_citations": int(a["ev_invalid"].sum())}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checks", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = audit_rows(expert_rows(args.raw), read_checks(args.checks))
    summary = summarise(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reference_audit.json").write_text(
        json.dumps(summary, indent=1, default=lambda o: o._asdict() if hasattr(o, "_asdict") else o.__dict__) + "\n")
    print(json.dumps(summary, indent=1, default=lambda o: o.__dict__))


if __name__ == "__main__":
    main()
