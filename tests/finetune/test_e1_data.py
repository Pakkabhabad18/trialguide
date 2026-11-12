from pathlib import Path

import pandas as pd
import pytest

from trialguide.finetune import e1_data as e1
from trialguide.schemas import Split


def _setup(tmp_path, rows, exclusions=None, columns=None):
    raw = tmp_path / "raw"
    (raw / "pmc_patients").mkdir(parents=True)
    frame = pd.DataFrame(rows, columns=columns or ["patient_uid", "PMID", "patient"])
    frame.to_csv(raw / e1.PMC_RELATIVE_PATH, index=False)
    exclude = tmp_path / "e1_exclude_pmids.txt"
    exclusions = exclusions if exclusions is not None else [str(900000 + i) for i in range(276)]
    exclude.write_text("\n".join(exclusions) + "\n", encoding="utf-8")
    return raw, exclude


def test_actual_frozen_exclusion_list_integrity():
    pmids = e1._read_exclusions(e1.E2_EXCLUSION_RELATIVE_PATH)
    assert len(pmids) == 276


def test_all_e2_pmids_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(e1, "split_sentences", lambda note, cohort: {0: note})
    pmids = [str(900000 + i) for i in range(276)]
    raw, exclusion = _setup(tmp_path, [[f"{p}-1", p, f"note {p}"] for p in pmids], pmids)
    summary = {}
    records = list(e1.iter_pmc_candidates(raw, exclusion, summary))
    assert records == []
    assert summary["excluded_e2_pmid"] == summary["excluded_pmids_in_list"] == 276


def test_missing_exclusion_list_entry_fails(tmp_path):
    pmids = [str(900000 + i) for i in range(275)]
    raw, exclusion = _setup(tmp_path, [["p1", "1", "note"]], pmids)
    with pytest.raises(ValueError, match="expected 276"):
        list(e1.iter_pmc_candidates(raw, exclusion))


@pytest.mark.parametrize("columns", [["patient_uid", "PMID"], ["patient_uid", "patient"]])
def test_missing_columns_fail(tmp_path, columns):
    raw, exclusion = _setup(tmp_path, [["p1", "1"]], columns=columns)
    with pytest.raises(ValueError, match="schema mismatch"):
        list(e1.iter_pmc_candidates(raw, exclusion))


@pytest.mark.parametrize("pmid", ["abc", "12.3", ""])
def test_malformed_or_missing_pmids_fail(tmp_path, pmid):
    raw, exclusion = _setup(tmp_path, [["p1", pmid, "note"]])
    with pytest.raises(ValueError, match="PMID"):
        list(e1.iter_pmc_candidates(raw, exclusion))


@pytest.mark.parametrize("note", [None, "", "   "])
def test_missing_or_empty_notes_fail(tmp_path, note):
    raw, exclusion = _setup(tmp_path, [["p1", "1", note]])
    with pytest.raises(ValueError, match="empty patient note"):
        list(e1.iter_pmc_candidates(raw, exclusion))


def test_duplicate_patient_ids_fail_across_rows(tmp_path):
    raw, exclusion = _setup(tmp_path, [["p1", "1", "note"], ["p1", "2", "another note"]])
    with pytest.raises(ValueError, match="duplicate"):
        list(e1.iter_pmc_candidates(raw, exclusion))


def test_evidence_candidate_binding_ids_text_and_required_evidence():
    candidate = e1.CandidateRecord("patient-a", {0: "Age is 54.", 2: "No prior treatment."}, "source", "id")
    valid = e1.Evidence("patient-a", 2, "No prior treatment.")
    e1.validate_evidence(candidate, [valid])
    with pytest.raises(ValueError, match="different candidate"):
        e1.validate_evidence(candidate, [e1.Evidence("patient-b", 2, valid.text)])
    with pytest.raises(ValueError, match="does not exist"):
        e1.validate_evidence(candidate, [e1.Evidence("patient-a", 1, "missing")])
    with pytest.raises(ValueError, match="verbatim"):
        e1.validate_evidence(candidate, [e1.Evidence("patient-a", 0, "Age is 54")])
    with pytest.raises(ValueError, match="evidence is required"):
        e1.validate_evidence(candidate, [])
    e1.validate_evidence(candidate, [], required=False)


def test_deterministic_iteration_and_source_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(e1, "split_sentences", lambda note, cohort: {0: note})
    raw, exclusion = _setup(tmp_path, [["22-1", "22", "first note"], ["11-1", "11", "second note"]])
    summary1, summary2 = {}, {}
    first = list(e1.iter_pmc_candidates(raw, exclusion, summary1))
    second = list(e1.iter_pmc_candidates(raw, exclusion, summary2))
    assert first == second and summary1 == summary2
    assert [(r.patient_id, r.sentences, r.source, r.source_record_id, r.source_pmid) for r in first] == [
        ("22-1", {0: "first note"}, "PMC-Patients-V1", "22-1", "22"),
        ("11-1", {0: "second note"}, "PMC-Patients-V1", "11-1", "11"),
    ]


def test_iterator_reads_chunks_and_yields_without_accumulating_notes(tmp_path, monkeypatch):
    monkeypatch.setattr(e1, "split_sentences", lambda note, cohort: {0: note})
    raw, exclusion = _setup(tmp_path, [[f"p{i}", str(i + 1), f"note {i}"] for i in range(6)])
    original = pd.read_csv
    calls = []

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("chunksize"):
            calls.append(kwargs["chunksize"])
        return result

    monkeypatch.setattr(e1.pd, "read_csv", spy)
    iterator = e1.iter_pmc_candidates(raw, exclusion)
    assert iter(iterator) is iterator
    first = next(iterator)
    assert first.patient_id == "p0"
    assert calls == [8192]


def _write_synthetic_splits(directory):
    import json

    from trialguide.data.splits import file_hashes

    directory.mkdir(parents=True)
    values = {
        "dev": [f"dev-{i}" for i in range(37)],
        "cal": [f"cal-{i}" for i in range(73)],
        "test": [f"test-{i}" for i in range(73)],
    }
    for split, ids in values.items():
        (directory / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (directory / "split_info.json").write_text(
        json.dumps({"sha256": file_hashes(directory)}), encoding="utf-8")
    return values


def test_dev_loader_fails_before_reading_mixed_split_query_notes(tmp_path, monkeypatch):
    """Cohort JSONL contains mixed splits, so do not even open it without an index."""
    from pathlib import Path

    split_dir = tmp_path / "splits"
    _write_synthetic_splits(split_dir)
    raw = tmp_path / "raw"
    for cohort in ("sigir", "trec_2021", "trec_2022"):
        path = raw / "trialgpt" / cohort / "queries.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"_id":"dev-0","text":"synthetic development sentinel"}\n'
            '{"_id":"cal-0","text":"synthetic calibration sentinel"}\n'
            '{"_id":"test-0","text":"synthetic test sentinel"}\n', encoding="utf-8")

    opened = []
    original_open = Path.open

    def track_open(path, *args, **kwargs):
        if "queries.jsonl" in str(path):
            opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track_open)
    with pytest.raises(RuntimeError, match="no split-selective index"):
        e1.load_development_candidates(raw, split_dir)
    assert opened == []


@pytest.mark.parametrize("bad_metadata", ["missing", "malformed", "wrong_counts", "hash_mismatch"])
def test_dev_loader_rejects_bad_split_metadata_before_opening_queries(tmp_path, bad_metadata,
                                                                      monkeypatch):
    import json

    from pathlib import Path

    split_dir = tmp_path / "splits"
    _write_synthetic_splits(split_dir)
    if bad_metadata == "missing":
        (split_dir / "split_info.json").unlink()
    elif bad_metadata == "malformed":
        (split_dir / "split_info.json").write_text("{broken", encoding="utf-8")
    elif bad_metadata == "wrong_counts":
        (split_dir / "dev.txt").write_text("only-one\n", encoding="utf-8")
        from trialguide.data.splits import file_hashes
        (split_dir / "split_info.json").write_text(
            json.dumps({"sha256": file_hashes(split_dir)}), encoding="utf-8")
    else:
        (split_dir / "split_info.json").write_text('{"sha256": {}}', encoding="utf-8")

    opened = []
    original_open = Path.open

    def track_open(path, *args, **kwargs):
        if "queries.jsonl" in str(path):
            opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track_open)
    with pytest.raises(ValueError, match="split metadata"):
        e1.load_development_candidates(tmp_path / "raw", split_dir)
    assert opened == []


def test_in_memory_development_candidate_evidence_validation_still_works():
    candidate = e1.CandidateRecord(
        "synthetic-dev", {0: "Synthetic development sentence."},
        "TrialGPT-synthetic", "synthetic-dev", split=Split.DEV.value)
    evidence = e1.Evidence("synthetic-dev", 0, "Synthetic development sentence.")
    e1.validate_evidence(candidate, [evidence])


def test_identifier_overlap_reads_frozen_ids_only(monkeypatch):
    monkeypatch.setattr(e1, "load_split", lambda _directory: {
        Split.DEV: ["dev1"], Split.CAL: ["cal1"], Split.TEST: ["test1"]})
    candidates = [e1.CandidateRecord("pmc-1", {}, "PMC-Patients-V1", "pmc-1")]
    status = e1.check_identifier_overlap(candidates, Path("unused"))
    assert status == {"exact_id_overlap": 0, "evaluation_id_count": 2, "content_overlap_verified": False}
    with pytest.raises(ValueError, match="overlap"):
        e1.check_identifier_overlap([e1.CandidateRecord("cal1", {}, "x", "x")], Path("unused"))
