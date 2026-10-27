"""Expert-label audit (T7): what do the 1,015 TrialGPT criterion annotations cover?

    python -m trialguide.data.audit [--root data] [--out results]

Answers the week-1 questions of the proposal: which patients have expert rule labels,
how many rows are "not enough information" (NEI), and how many of those a better
system could have resolved. Writes results/label_audit.json and results/label_audit.md.
Reads only the downloaded files (T3); uses no model and no split.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# Labels that settle a rule. "not applicable" and NEI do not.
DEFINITE = {"included", "not included", "excluded", "not excluded"}
NEI = "not enough information"


def cohort_of(patient_id: str) -> str:
    """sigir-20141 -> sigir, trec-20211 -> trec_2021, trec-20221 -> trec_2022."""
    if patient_id.startswith("sigir"):
        return "sigir"
    if patient_id.startswith("trec-2021"):
        return "trec_2021"
    if patient_id.startswith("trec-2022"):
        return "trec_2022"
    return "unknown"


def load_queries(raw: Path) -> dict:
    """cohort -> {patient_id: note text}."""
    out = {}
    for c in ("sigir", "trec_2021", "trec_2022"):
        with open(raw / "trialgpt" / c / "queries.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        out[c] = {r["_id"]: r["text"] for r in rows}
    return out


def patients_with_trial_labels(raw: Path, cohort: str) -> set:
    """Patients that have at least one expert patient-trial label (qrels)."""
    qrels = pd.read_csv(raw / "trialgpt" / cohort / "qrels" / "test.tsv", sep="\t")
    return set(qrels["query-id"])


def audit(raw: Path) -> dict:
    ann = pd.read_parquet(raw / "annotations" / "criterion_annotations.parquet")
    queries = load_queries(raw)
    res = {"rows": len(ann), "annotated_patients": int(ann.patient_id.nunique())}

    # --- cohort sizes and how many patients are usable (note + trial labels) ---
    cohorts = {}
    for c, q in queries.items():
        labelled = patients_with_trial_labels(raw, c)
        annotated = set(ann.patient_id) & set(q)
        cohorts[c] = {
            "patients_in_queries": len(q),
            "patients_with_trial_labels": len(labelled & set(q)),
            "patients_without_trial_labels": sorted(set(q) - labelled),
            "patients_with_rule_labels": len(annotated),
            "rule_label_rows": int(ann.patient_id.isin(annotated).sum()),
        }
    res["cohorts"] = cohorts
    # Annotated patients that exist in no cohort file (no note, no trial labels to build on)
    known = set().union(*[set(q) for q in queries.values()])
    orphans = sorted(set(ann.patient_id) - known)
    res["annotated_patients_missing_from_cohorts"] = {
        p: int((ann.patient_id == p).sum()) for p in orphans}

    # --- label distributions ---
    res["expert_label_counts"] = {t: g.expert_eligibility.value_counts().to_dict()
                                  for t, g in ann.groupby("criterion_type")}
    res["rows_per_patient"] = ann.groupby("patient_id").size().describe().round(1).to_dict()
    res["patient_trial_pairs"] = int(ann[["patient_id", "trial_id"]].drop_duplicates().shape[0])

    # --- NEI breakdown: expert label vs GPT-4 label ---
    e, g = ann.expert_eligibility, ann.gpt4_eligibility
    n_nei_expert = int((e == NEI).sum())
    recoverable = (g == NEI) & e.isin(DEFINITE)         # GPT-4 gave up, expert could decide
    overcommit = (e == NEI) & g.isin(DEFINITE)          # GPT-4 decided, expert says note is silent
    disagree = g != e
    res["nei"] = {
        "expert_nei_rows": n_nei_expert,
        "expert_nei_share": round(n_nei_expert / len(ann), 3),
        "gpt4_nei_rows": int((g == NEI).sum()),
        "both_nei": int(((g == NEI) & (e == NEI)).sum()),
        "recoverable_nei": int(recoverable.sum()),
        "gpt4_decided_but_expert_nei": int(overcommit.sum()),
        "gpt4_nei_other_expert_label": int(((g == NEI) & ~e.isin(DEFINITE) & (e != NEI)).sum()),
        "gpt4_disagreements": int(disagree.sum()),
        "recoverable_share_of_disagreements": round(recoverable.sum() / max(disagree.sum(), 1), 3),
        "recoverable_share_of_gpt4_nei": round(recoverable.sum() / max((g == NEI).sum(), 1), 3),
        "recoverable_by_type": ann[recoverable].criterion_type.value_counts().to_dict(),
    }
    # Does the expert point at note sentences when recoverable? (needed for cited inference)
    ev = ann[recoverable].expert_sentences.map(lambda s: len(json.loads(s)) if s else 0)
    res["nei"]["recoverable_with_expert_evidence"] = int((ev > 0).sum())
    res["gpt4_accuracy_vs_expert"] = round(float((g == e).mean()), 3)
    return res


def to_markdown(r: dict) -> str:
    n = r["nei"]
    L = ["# Expert-label audit (T7)", "",
         f"Source: `ncbi/TrialGPT-Criterion-Annotations`, {r['rows']} rule-level rows, "
         f"{r['annotated_patients']} patients, {r['patient_trial_pairs']} patient-trial pairs.", "",
         "## Coverage by cohort", "",
         "| Cohort | Patients (queries) | With trial labels | With rule labels | Rule-label rows |",
         "|---|---|---|---|---|"]
    for c, v in r["cohorts"].items():
        L.append(f"| {c} | {v['patients_in_queries']} | {v['patients_with_trial_labels']} | "
                 f"{v['patients_with_rule_labels']} | {v['rule_label_rows']} |")
    L += ["", f"Annotated patients missing from every cohort file: {r['annotated_patients_missing_from_cohorts']}", "",
          "Patients without trial labels: " + "; ".join(
              f"{c}: {v['patients_without_trial_labels']}" for c, v in r["cohorts"].items()
              if v["patients_without_trial_labels"]), "",
          "## Expert label counts", ""]
    for t, cnt in r["expert_label_counts"].items():
        L.append(f"- {t}: {cnt}")
    L += ["", "## Not enough information (NEI)", "",
          "| | Rows |", "|---|---|",
          f"| Expert says NEI | {n['expert_nei_rows']} ({n['expert_nei_share']:.1%}) |",
          f"| GPT-4 says NEI | {n['gpt4_nei_rows']} |",
          f"| Both say NEI | {n['both_nei']} |",
          f"| **Recoverable** (GPT-4 NEI, expert gives a definite label) | {n['recoverable_nei']} |",
          f"| ... of which the expert cites note sentences | {n['recoverable_with_expert_evidence']} |",
          f"| GPT-4 decided, expert says NEI | {n['gpt4_decided_but_expert_nei']} |",
          f"| GPT-4 NEI, expert says not applicable | {n['gpt4_nei_other_expert_label']} |", "",
          f"Recoverable rows are {n['recoverable_share_of_disagreements']:.1%} of the "
          f"{n['gpt4_disagreements']} GPT-4/expert disagreements and "
          f"{n['recoverable_share_of_gpt4_nei']:.1%} of GPT-4's NEI answers "
          f"(TrialGPT paper: 30.7% of 26 errors). By type: {n['recoverable_by_type']}.", ""]
    return "\n".join(L)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)
    r = audit(Path(a.root) / "raw")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_audit.json").write_text(json.dumps(r, indent=2, default=str) + "\n")
    md = to_markdown(r)
    (out / "label_audit.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
