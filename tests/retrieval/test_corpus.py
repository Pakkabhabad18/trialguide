import json
import tempfile
from pathlib import Path

import pytest

from trialguide.retrieval import keyword_retrieve
from trialguide.retrieval.corpus import SUPPORTED_COHORTS, load_trial_corpus
from trialguide.schemas import Split, VisibleNote


@pytest.fixture
def synthetic_raw_root():
    """Create and remove a temporary root for synthetic shared-corpus files."""
    with tempfile.TemporaryDirectory(prefix="t18-corpus-", dir=Path(__file__).parent) as temp:
        yield Path(temp)


def _write_corpus(raw, cohort, records):
    """Write synthetic trial records at the shared corpus path for a cohort."""
    path = raw / "trialgpt" / cohort / "corpus.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


@pytest.mark.parametrize("cohort", SUPPORTED_COHORTS)
def test_each_cohort_loads_its_own_corpus_path(synthetic_raw_root, cohort):
    """Select the corpus file named by the supplied cohort."""
    expected_id = f"{cohort}-trial"
    _write_corpus(synthetic_raw_root, cohort, [{"_id": expected_id, "title": "Title", "text": "Body"}])

    assert load_trial_corpus(synthetic_raw_root, cohort)[0]["_id"] == expected_id


def test_valid_records_preserve_metadata_and_extra_fields(synthetic_raw_root):
    """Keep optional metadata and unknown source fields unchanged."""
    record = {
        "_id": "NCT000001",
        "title": "Lung cancer study",
        "text": "Study body",
        "metadata": {"diseases_list": ["lung cancer"], "phase": "2"},
        "extra": {"source": "synthetic"},
    }
    _write_corpus(synthetic_raw_root, "sigir", [record])

    assert load_trial_corpus(synthetic_raw_root, "sigir") == [record]


@pytest.mark.parametrize("trial_id", [None, 17, "", "   "])
def test_missing_or_invalid_ids_fail_with_file_and_line(synthetic_raw_root, trial_id):
    """Reject IDs that are absent, non-string, or blank."""
    record = {"_id": trial_id, "title": "Title", "text": "Body"}
    _write_corpus(synthetic_raw_root, "sigir", [record])

    with pytest.raises(ValueError, match=r"corpus\.jsonl:1: _id must be a non-empty string"):
        load_trial_corpus(synthetic_raw_root, "sigir")


def test_missing_id_fails(synthetic_raw_root):
    """Reject a record with no trial identifier."""
    _write_corpus(synthetic_raw_root, "sigir", [{"title": "Title", "text": "Body"}])

    with pytest.raises(ValueError, match=r"corpus\.jsonl:1: _id must be a non-empty string"):
        load_trial_corpus(synthetic_raw_root, "sigir")


def test_duplicate_ids_fail_and_report_both_lines(synthetic_raw_root):
    """Report duplicate IDs with the current and original line numbers."""
    _write_corpus(synthetic_raw_root, "sigir", [
        {"_id": "NCT000001", "title": "One", "text": "Body"},
        {"_id": "NCT000001", "title": "Two", "text": "Body"},
    ])

    with pytest.raises(ValueError, match=r"corpus\.jsonl:2: duplicate trial _id .* line 1"):
        load_trial_corpus(synthetic_raw_root, "sigir")


@pytest.mark.parametrize("field", ["title", "text"])
@pytest.mark.parametrize("invalid", [None, 3, "", "   "])
def test_missing_or_invalid_required_text_fields_fail(synthetic_raw_root, field, invalid):
    """Reject required title and body values that are absent or not usable strings."""
    record = {"_id": "NCT000001", "title": "Title", "text": "Body"}
    record[field] = invalid
    _write_corpus(synthetic_raw_root, "sigir", [record])

    with pytest.raises(ValueError, match=rf"corpus\.jsonl:1: {field} must be a non-empty string"):
        load_trial_corpus(synthetic_raw_root, "sigir")


@pytest.mark.parametrize("field", ["title", "text"])
def test_missing_required_text_fields_fail(synthetic_raw_root, field):
    """Reject a missing required title or body field."""
    record = {"_id": "NCT000001", "title": "Title", "text": "Body"}
    del record[field]
    _write_corpus(synthetic_raw_root, "sigir", [record])

    with pytest.raises(ValueError, match=rf"corpus\.jsonl:1: {field} must be a non-empty string"):
        load_trial_corpus(synthetic_raw_root, "sigir")


def test_malformed_json_reports_file_and_line(synthetic_raw_root):
    """Include the malformed JSON line in the loader error."""
    path = _write_corpus(synthetic_raw_root, "sigir", [{"_id": "NCT000001", "title": "Title", "text": "Body"}])
    with path.open("a", encoding="utf-8") as corpus_file:
        corpus_file.write('{"_id": "broken"\n')

    with pytest.raises(ValueError, match=r"corpus\.jsonl:2: malformed JSON"):
        load_trial_corpus(synthetic_raw_root, "sigir")


def test_non_object_json_record_fails_with_line(synthetic_raw_root):
    """Reject valid JSON values that are not trial-record objects."""
    path = _write_corpus(synthetic_raw_root, "sigir", [{"_id": "NCT000001", "title": "Title", "text": "Body"}])
    with path.open("a", encoding="utf-8") as corpus_file:
        corpus_file.write('[]\n')

    with pytest.raises(ValueError, match=r"corpus\.jsonl:2: record must be a JSON object"):
        load_trial_corpus(synthetic_raw_root, "sigir")


def test_unsupported_cohort_fails_clearly(synthetic_raw_root):
    """Explain which cohort names the loader accepts."""
    with pytest.raises(ValueError, match="unsupported cohort 'unknown'"):
        load_trial_corpus(synthetic_raw_root, "unknown")


def test_loaded_records_remain_compatible_with_retrieval_api(synthetic_raw_root):
    """Pass the loaded dictionaries directly into the existing retrieval interface."""
    records = [
        {"_id": "NCT000001", "title": "Lung cancer study", "text": "EGFR treatment"},
        {"_id": "NCT000002", "title": "Diabetes study", "text": "Glucose treatment"},
    ]
    _write_corpus(synthetic_raw_root, "sigir", records)
    note = VisibleNote("synthetic-dev", Split.DEV, {0: "lung cancer EGFR"})

    results = keyword_retrieve(note, load_trial_corpus(synthetic_raw_root, "sigir"), top_k=2)

    assert results
    assert results[0].trial_id == "NCT000001"
