"""Synthetic tests that prove DEV-only readers skip non-DEV record payloads."""

import json

import pytest

from trialguide.data import dev_only
from trialguide.schemas import Split

COHORTS = ("sigir", "trec_2021", "trec_2022")


def _source_tree(root, split_dir):
    """Create cohort source files and a DEV ID list without real records or labels."""
    split_dir.mkdir(parents=True)
    (split_dir / "dev.txt").write_text("dev-patient\n", encoding="utf-8")
    raw = root / "raw"
    for cohort in COHORTS:
        queries = raw / "trialgpt" / cohort / "queries.jsonl"
        queries.parent.mkdir(parents=True)
        queries.write_text("", encoding="utf-8")
        qrels = raw / "trialgpt" / cohort / "qrels" / "test.tsv"
        qrels.parent.mkdir()
        qrels.write_text("query-id\ttrial-id\tgrade\n", encoding="utf-8")
    return raw


def test_dev_loaders_skip_malformed_non_dev_payloads(tmp_path, monkeypatch):
    """Parse DEV note/qrel fields while skipping malformed non-DEV payloads."""
    split_dir = tmp_path / "splits"
    raw = _source_tree(tmp_path, split_dir)
    query_file = raw / "trialgpt" / "sigir" / "queries.jsonl"
    query_file.write_text(
        json.dumps({"text": "DEV visible note", "_id": "dev-patient"})
        + '\n{"_id":"cal-patient","text": NOT-VALID-JSON}\n'
        + '{"text": NOT-JSON-BUT-SKIPPED, "_id":"cal-before-id"}\n'
        + '{"_id":"test-patient","text": [broken non-dev payload}\n',
        encoding="utf-8",
    )
    qrel_file = raw / "trialgpt" / "sigir" / "qrels" / "test.tsv"
    qrel_file.write_text(
        "query-id\ttrial-id\tgrade\n"
        "dev-patient\ttrial-1\t2\n"
        'cal-patient\t"unterminated calibration row\n'
        "test-patient\tmalformed sealed row\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dev_only, "split_sentences", lambda text, cohort: {0: text})

    patients = dev_only.load_dev_patients(raw, split_dir)
    grades = dev_only.load_dev_gold_grades(raw, split_dir)

    assert set(patients) == {"dev-patient"}
    assert patients["dev-patient"].split == Split.DEV
    assert patients["dev-patient"].sentences == {0: "DEV visible note"}
    assert grades == {
        "dev-patient": {"trial-1": 2},
    }


def test_dev_loaders_read_only_dev_split_id_file(tmp_path, monkeypatch):
    """Do not require or open calibration/test split lists to find DEV patient IDs."""
    split_dir = tmp_path / "splits"
    raw = _source_tree(tmp_path, split_dir)
    (split_dir / "cal.txt").write_text("malformed calibration IDs {", encoding="utf-8")
    (split_dir / "test.txt").write_text("malformed sealed IDs {", encoding="utf-8")
    monkeypatch.setattr(dev_only, "split_sentences", lambda text, cohort: {0: text})
    (raw / "trialgpt" / "sigir" / "queries.jsonl").write_text(
        json.dumps({"_id": "dev-patient", "text": "note"}) + "\n", encoding="utf-8"
    )

    assert dev_only.load_dev_patient_ids(split_dir) == frozenset({"dev-patient"})
    assert set(dev_only.load_dev_patients(raw, split_dir)) == {"dev-patient"}


def test_malformed_dev_note_and_qrel_fail_with_file_line_only(tmp_path, monkeypatch):
    """Reject malformed selected DEV data without echoing note or qrel contents."""
    split_dir = tmp_path / "splits"
    raw = _source_tree(tmp_path, split_dir)
    query_file = raw / "trialgpt" / "sigir" / "queries.jsonl"
    query_file.write_text('{"_id":"dev-patient","text": broken}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"queries\.jsonl:1: malformed DEV JSON record"):
        dev_only.load_dev_patients(raw, split_dir)

    query_file.write_text(
        json.dumps({"_id": "dev-patient", "text": "note"}) + "\n", encoding="utf-8"
    )
    qrel_file = raw / "trialgpt" / "sigir" / "qrels" / "test.tsv"
    qrel_file.write_text("query-id\ttrial-id\tgrade\ndev-patient\ttrial-1\tbad\n", encoding="utf-8")
    monkeypatch.setattr(dev_only, "split_sentences", lambda text, cohort: {0: text})

    with pytest.raises(ValueError, match=r"test\.tsv:2: invalid DEV qrel grade"):
        dev_only.load_dev_gold_grades(raw, split_dir)
