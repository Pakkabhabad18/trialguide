"""T7: the expert-label audit on a tiny hand-made dataset."""

import json

import pandas as pd

from trialguide.data import audit


def _write_raw(tmp_path):
    raw = tmp_path / "raw"
    for c, ids in {"sigir": ["sigir-1", "sigir-2"], "trec_2021": ["trec-20211"], "trec_2022": ["trec-20221"]}.items():
        d = raw / "trialgpt" / c
        (d / "qrels").mkdir(parents=True)
        (d / "queries.jsonl").write_text("\n".join(json.dumps({"_id": i, "text": "note"}) for i in ids))
        # sigir-2 has no trial labels
        labelled = [i for i in ids if i != "sigir-2"]
        pd.DataFrame({"query-id": labelled, "corpus-id": ["NCT1"] * len(labelled), "score": [1] * len(labelled)}
                     ).to_csv(d / "qrels" / "test.tsv", sep="\t", index=False)
    rows = [  # patient, type, gpt4, expert, expert evidence
        ("sigir-1", "inclusion", "not enough information", "included", "[1]"),                  # recoverable, cited
        ("sigir-1", "inclusion", "not enough information", "included", "[]"),                   # recoverable, no cite
        ("sigir-1", "exclusion", "not enough information", "not enough information", "[]"),     # both NEI
        ("sigir-1", "exclusion", "not excluded", "not enough information", "[]"),               # gpt4 over-commits
        ("sigir-1", "inclusion", "included", "included", "[0]"),                                # agree
        ("sigir-9", "inclusion", "included", "included", "[0]"),                                # patient in no cohort
    ]
    ann = pd.DataFrame(rows, columns=["patient_id", "criterion_type", "gpt4_eligibility",
                                      "expert_eligibility", "expert_sentences"])
    ann["annotation_id"] = range(len(ann))
    ann["trial_id"] = "NCT1"
    (raw / "annotations").mkdir()
    ann.to_parquet(raw / "annotations" / "criterion_annotations.parquet")
    return raw


def test_cohort_of():
    assert audit.cohort_of("sigir-20141") == "sigir"
    assert audit.cohort_of("trec-20211") == "trec_2021"
    assert audit.cohort_of("trec-202250") == "trec_2022"


def test_audit_counts(tmp_path):
    r = audit.audit(_write_raw(tmp_path))
    assert r["cohorts"]["sigir"]["patients_with_trial_labels"] == 1
    assert r["cohorts"]["sigir"]["patients_without_trial_labels"] == ["sigir-2"]
    assert r["cohorts"]["sigir"]["rule_label_rows"] == 5
    assert r["cohorts"]["trec_2021"]["rule_label_rows"] == 0
    assert r["annotated_patients_missing_from_cohorts"] == {"sigir-9": 1}
    n = r["nei"]
    assert n["expert_nei_rows"] == 2
    assert n["recoverable_nei"] == 2 and n["recoverable_with_expert_evidence"] == 1
    assert n["gpt4_decided_but_expert_nei"] == 1
    assert n["both_nei"] == 1


def test_markdown_renders(tmp_path):
    md = audit.to_markdown(audit.audit(_write_raw(tmp_path)))
    assert "Recoverable" in md and "sigir" in md
