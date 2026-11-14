"""Error analysis of the frozen test run (proposal: failure analysis by rule type, and case studies).

    python -m trialguide.eval.error_analysis --run results/test_run --allow-test

Reads only saved outputs (no model calls): the checker on the full and on the masked note, B7's final
rule statuses and its conversations. The reference is the checker on the FULL note, as in
eval/recovery.py: a rule is "lost" if the checker settled it (met / not met) on the full note and
B6 leaves it unresolved on the masked note. Expert-label results are in risk_test.json; this file is
about where and why B7 gains or fails, so it uses the larger model-derived reference.

1. Lost rules: what B7 ends with (recovered / wrong toward eligibility / wrong toward exclusion /
   still open), split by how the rule was settled: after a doctor's answer about it, or by
   inference alone.
2. The same by rule type. Types come from keywords in the rule text (age, pregnancy, lab or vital
   value, consent or compliance, prior treatment, otherwise condition); a rough grouping, stated as such.
3. Questions: how many settled at least one rule, by answer certainty, and the share taken by the
   most frequent fact_key (the "prior chemotherapy" key the checker copies from its prompt example).
4. Case studies: one example each of a correct inference, an inference wrong toward eligibility, a
   question that settled rules, a wasted question, and a rule left open because its evidence was
   hidden. Picked as the first example in patient order, so the choice is reproducible, not selected.

Writes {run}/error_analysis.json.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from trialguide.eval.recovery import favours_eligibility, policy_finals

DECIDED = ("met", "not_met")

# rule type = first pattern that matches the rule text (order matters: age before lab values etc.)
RULE_TYPES = [
    ("age", re.compile(r"\bage[ds]?\b|years? (of age|old)|\b\d+\s*(years|yrs)\b", re.I)),
    ("pregnancy", re.compile(r"pregnan|lactat|breast-?feed|contracept|childbearing", re.I)),
    ("lab or vital value", re.compile(
        r"creatinine|bilirubin|h(a)?emoglobin|platelet|neutrophil|leukocyte|wbc|\balt\b|\bast\b|egfr|"
        r"hba1c|glucose|potassium|sodium|blood pressure|bmi|body mass|ejection fraction|"
        r"\d\s*(mg/dl|g/dl|mmol|[uµμ]mol|/mm3|/[uµμ]l|mmhg|ml/min|kg/m|iu/l|u/l|%)", re.I)),
    ("consent or compliance", re.compile(r"consent|willing|able to (comply|understand|attend)|comply", re.I)),
    ("prior treatment", re.compile(
        r"therap|treat|chemo|surgery|surgical|drug|medication|radiation|radiotherap|transplant|"
        r"vaccin|received|receiving|taking|use of|investigational", re.I)),
]


def rule_type(text: str) -> str:
    for name, pattern in RULE_TYPES:
        if pattern.search(text or ""):
            return name
    return "condition or other"


def by_patient(path: Path) -> dict:
    """checks JSONL -> patient -> 'trial|rule' -> check dict."""
    out = {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        out[row["patient_id"]] = {f"{c['trial_id']}|{c['rule_id']}": c for c in row["checks"]}
    return out


def answered_rules(turns_path: Path, system: str = "B7") -> tuple[dict, list]:
    """Rules each patient's documented answers were about, and every question of `system`."""
    about, questions = {}, []
    for line in open(turns_path, encoding="utf-8"):
        row = json.loads(line)
        if row["system"] != system:
            continue
        for t in row["turns"]:
            refs = [f"{tid}|{rid}" for tid, rid in t["question"]["rule_refs"]]
            questions.append({"patient_id": row["patient_id"], "fact_key": t["question"]["fact_key"],
                              "text": t["question"]["text"], "certainty": t["answer"]["certainty"],
                              "answer": t["answer"].get("value"), "quote": t["answer"].get("quote"),
                              "rules": refs})
            if t["answer"]["certainty"] != "not_documented":
                about.setdefault(row["patient_id"], set()).update(refs)
    return about, questions


def outcome(final_status: str, full_status: str, kind: str) -> str:
    if final_status not in DECIDED:
        return "still open"
    if final_status == full_status:
        return "recovered"
    return "wrong toward eligibility" if favours_eligibility(kind, final_status) else "wrong toward exclusion"


def lost_rules(full: dict, b6: dict, b7: dict, about: dict, rule_text: dict) -> list:
    """One row per lost rule with B7's outcome, how it was settled and the rule type."""
    rows = []
    for pid, rules in full.items():
        for key, f in rules.items():
            if f["status"] not in DECIDED or b6[pid].get(key) in DECIDED:
                continue
            status, category, kind = b7[pid][key]
            rows.append({"patient_id": pid, "rule": key, "kind": kind, "full_status": f["status"],
                         "b7_status": status, "b7_category": category,
                         "outcome": outcome(status, f["status"], kind),
                         "settled_by": ("doctor answer" if key in about.get(pid, set()) else "inference")
                         if status in DECIDED else "-",
                         "type": rule_type(rule_text.get(key, ""))})
    return rows


def table(rows: list, group: str) -> dict:
    """Outcome shares per group value."""
    out = {}
    for value in sorted({r[group] for r in rows}):
        g = [r for r in rows if r[group] == value]
        c = Counter(r["outcome"] for r in g)
        out[value] = {"rules": len(g), **{k: c[k] / len(g) for k in
                      ("recovered", "wrong toward eligibility", "wrong toward exclusion", "still open")}}
    return out


def question_stats(questions: list, b6: dict, b7: dict) -> dict:
    """Did each question settle at least one of its rules (unresolved under B6, decided at the end)?"""
    for q in questions:
        q["settled"] = sum(b6[q["patient_id"]].get(k) not in DECIDED and b7[q["patient_id"]][k][0] in DECIDED
                           for k in q["rules"] if k in b7[q["patient_id"]]) if q["certainty"] != "not_documented" else 0
    keys = Counter(q["fact_key"] for q in questions)
    top, n_top = keys.most_common(1)[0]
    top_q = [q for q in questions if q["fact_key"] == top]
    rest = [q for q in questions if q["fact_key"] != top]
    nd = lambda qs: sum(q["certainty"] == "not_documented" for q in qs)
    return {"questions": len(questions),
            "by_certainty": dict(Counter(q["certainty"] for q in questions)),
            "settled_at_least_one_rule": sum(q["settled"] > 0 for q in questions),
            "most_frequent_fact_key": top, "most_frequent_count": n_top,
            "most_frequent_rules_per_question": sum(len(q["rules"]) for q in top_q) / n_top,
            "most_frequent_not_documented": nd(top_q),
            "other_questions": len(rest), "other_not_documented": nd(rest),
            "share_of_all_not_documented_from_most_frequent": nd(top_q) / max(nd(questions), 1)}


def case_studies(lost: list, questions: list, masked: dict, full: dict, notes: dict, hidden: dict,
                 rule_text: dict) -> dict:
    """The first example of each kind, in patient order, with the text needed to read it."""
    def rule_case(r):
        m, f = masked[r["patient_id"]][r["rule"]], full[r["patient_id"]][r["rule"]]
        sents = notes[r["patient_id"]]
        return {"patient_id": r["patient_id"], "rule": r["rule"], "kind": r["kind"],
                "rule_text": rule_text.get(r["rule"], ""), "full_note_status": r["full_status"],
                "b7_status": r["b7_status"], "b7_category": r["b7_category"],
                "masked_explanation": m["explanation"], "masked_evidence": [sents[i] for i in m["evidence_ids"] if i in sents],
                "full_note_evidence": [sents[i] for i in f["evidence_ids"] if i in sents],
                "full_note_evidence_hidden": bool(f["evidence_ids"]) and set(f["evidence_ids"]) <= hidden[r["patient_id"]]}

    lost = sorted(lost, key=lambda r: (r["patient_id"], r["rule"]))
    first = lambda cond: next((r for r in lost if cond(r)), None)
    cases = {
        "correct_inference": first(lambda r: r["settled_by"] == "inference" and r["outcome"] == "recovered"),
        "inference_wrong_toward_eligibility": first(lambda r: r["settled_by"] == "inference"
                                                    and r["outcome"] == "wrong toward eligibility"),
        "left_open_evidence_hidden": first(lambda r: r["outcome"] == "still open" and
                                           set(full[r["patient_id"]][r["rule"]]["evidence_ids"]) <= hidden[r["patient_id"]]
                                           and full[r["patient_id"]][r["rule"]]["evidence_ids"]),
    }
    out = {k: rule_case(v) if v else None for k, v in cases.items()}
    qs = sorted(questions, key=lambda q: (q["patient_id"], q["text"]))
    useful = next((q for q in qs if q["settled"] > 0), None)
    wasted = next((q for q in qs if q["certainty"] == "not_documented" and len(q["rules"]) > 5), None)
    for name, q in (("question_that_settled_rules", useful), ("wasted_question", wasted)):
        out[name] = None if q is None else {
            **{k: q[k] for k in ("patient_id", "text", "certainty", "answer", "quote", "settled")},
            "rules_merged": len(q["rules"]),
            "rule_texts": [rule_text.get(k, "") for k in q["rules"][:5]]}
    return out


def main(argv=None) -> None:
    from trialguide.data.patients import load_patients
    from trialguide.finetune.data import load_trials
    from trialguide.pipeline import rules_of_trial
    from trialguide.schemas import Split

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, default=Path("results/test_run"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest["split"] == "test" and not args.allow_test:
        raise SystemExit("test split: pass --allow-test")

    full = by_patient(args.run / "checks_full.jsonl")
    masked = by_patient(args.run / f"checks_{manifest['mask']}{round(manifest['fraction'] * 100)}.jsonl")
    b6 = policy_finals(args.run / f"checks_{manifest['mask']}{round(manifest['fraction'] * 100)}.jsonl", "B6")
    b7 = json.loads((args.run / "final_B7.json").read_text())
    about, questions = answered_rules(args.run / "turns_asking.jsonl")
    trials = load_trials(args.raw)
    rule_text = {f"{r.trial_id}|{r.rule_id}": r.text
                 for info in manifest["patients"].values() for ts in info["trials"].values()
                 for t in ts for r in rules_of_trial(trials[t])}
    patients = load_patients(args.raw, Split(manifest["split"]), allow_test=args.allow_test)
    notes = {pid: patients[pid].sentences for pid in manifest["patients"]}
    hidden = {pid: set(info["hidden_ids"]) for pid, info in manifest["patients"].items()}

    lost = lost_rules(full, b6, b7, about, rule_text)
    result = {"run": str(args.run), "reference": "checker on the full note (model-derived)",
              "lost_rules": len(lost),
              "outcomes": dict(Counter(r["outcome"] for r in lost)),
              "by_how_settled": table([r for r in lost if r["settled_by"] != "-"], "settled_by"),
              "by_rule_type": table(lost, "type"),
              "by_rule_kind": table(lost, "kind"),
              "questions_B7": question_stats(questions, b6, b7),
              "case_studies": case_studies(lost, questions, masked, full, notes, hidden, rule_text)}
    (args.run / "error_analysis.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "case_studies"}, indent=1))


if __name__ == "__main__":
    main()
