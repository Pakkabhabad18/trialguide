"""E1 candidate validation (no labels, training examples, or model execution).

PMC-Patients V1 has no expert trial-eligibility labels. This module streams unlabelled,
source-traceable candidate records only. Record-level PMC/evaluation overlap is unresolved;
see docs/e1_data_preparation.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import pandas as pd

from trialguide.data.patients import split_sentences
from trialguide.data.splits import SPLIT_DIR, file_hashes, load_split
from trialguide.schemas import Split

PMC_RELATIVE_PATH = Path("pmc_patients") / "PMC-Patients.csv"
E2_EXCLUSION_RELATIVE_PATH = SPLIT_DIR / "e2" / "e1_exclude_pmids.txt"
E2_INFO_PATH = SPLIT_DIR / "e2" / "e2_info.json"
REQUIRED_PMC_COLUMNS = {"patient_uid", "PMID", "patient"}
EXPECTED_E2_PMID_COUNT = 276
EXPECTED_SPLIT_SIZES = {Split.DEV: 37, Split.CAL: 73, Split.TEST: 73}


@dataclass(frozen=True)
class CandidateRecord:
    """A note represented by sentence IDs and source metadata; contains no labels."""

    patient_id: str
    sentences: Mapping[int, str]
    source: str
    source_record_id: str
    source_pmid: str | None = None
    split: str | None = None


@dataclass(frozen=True)
class Evidence:
    candidate_id: str
    sentence_id: int
    text: str


def _read_exclusions(path: Path) -> set[str]:
    """Read the frozen list, validating its repository-pinned digest when applicable.

    Explicit alternate paths support isolated synthetic fixtures; production defaults to
    the repository list and verifies its bytes against committed e2_info.json metadata.
    """
    if not path.is_file():
        raise FileNotFoundError(f"E1 PMID exclusion list is required: {path}")
    if path.resolve() == E2_EXCLUSION_RELATIVE_PATH.resolve():
        if not E2_INFO_PATH.is_file():
            raise FileNotFoundError(f"E2 list integrity metadata is required: {E2_INFO_PATH}")
        info = json.loads(E2_INFO_PATH.read_text(encoding="utf-8"))
        expected = info.get("sha256", {}).get(path.name)
        if not expected:
            raise ValueError(f"E2 metadata has no checksum for {path.name}")
        from trialguide.data.e2 import _hash
        if _hash(path) != expected:
            raise ValueError("frozen E1 PMID exclusion list does not match splits/e2/e2_info.json")
    pmids = path.read_text(encoding="utf-8").split()
    if len(pmids) != EXPECTED_E2_PMID_COUNT or len(set(pmids)) != len(pmids):
        raise ValueError(f"expected {EXPECTED_E2_PMID_COUNT} unique frozen E2 PMIDs; got {len(pmids)} entries")
    if any(not p.isdecimal() for p in pmids):
        raise ValueError("E2 exclusion list contains a malformed PMID")
    return set(pmids)


def iter_pmc_candidates(raw: Path, exclusion_file: Path = E2_EXCLUSION_RELATIVE_PATH,
                        summary: dict | None = None) -> Iterator[CandidateRecord]:
    """Yield PMC V1 candidates in CSV order, retaining only a chunk and IDs in memory.

    The optional `summary` dict is populated as iteration proceeds and is complete only
    after exhaustion. Memory use is O(chunk size + unique record IDs + exclusion IDs),
    not O(total note text). Since duplicate IDs are rejected, the ID set is necessary for
    fail-closed validation across chunks.
    """
    csv_path = raw / PMC_RELATIVE_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(f"pinned PMC-Patients V1 CSV is required: {csv_path}")
    exclusions = _read_exclusions(exclusion_file)
    try:
        header = pd.read_csv(csv_path, nrows=0).columns
    except Exception as exc:
        raise ValueError(f"cannot read PMC-Patients CSV header: {exc}") from exc
    missing = REQUIRED_PMC_COLUMNS - set(header)
    if missing:
        raise ValueError(f"PMC-Patients schema mismatch; missing columns: {sorted(missing)}")

    counts = {"rows": 0, "excluded_e2_pmid": 0, "retained": 0,
              "empty_note": 0, "excluded_pmids_in_list": len(exclusions)}
    seen: set[str] = set()
    if summary is not None:
        summary.clear()
        summary.update(counts)
    try:
        chunks = pd.read_csv(csv_path, usecols=sorted(REQUIRED_PMC_COLUMNS), dtype=str, chunksize=8192)
        for chunk in chunks:
            for row in chunk.itertuples(index=False, name=None):
                values = dict(zip(chunk.columns, row))
                uid, pmid, note = (values[k] for k in ("patient_uid", "PMID", "patient"))
                counts["rows"] += 1
                if pd.isna(uid) or not str(uid).strip() or pd.isna(pmid) or not str(pmid).strip():
                    raise ValueError("PMC row has a missing patient_uid or PMID")
                uid, pmid = str(uid).strip(), str(pmid).strip()
                if not pmid.isdecimal():
                    raise ValueError(f"PMC record {uid} has a malformed PMID")
                if uid in seen:
                    raise ValueError(f"duplicate PMC patient_uid: {uid}")
                seen.add(uid)
                if pd.isna(note) or not str(note).strip():
                    counts["empty_note"] += 1
                    raise ValueError(f"PMC record {uid} has an empty patient note")
                if pmid in exclusions:
                    counts["excluded_e2_pmid"] += 1
                    if summary is not None:
                        summary.update(counts)
                    continue
                record = CandidateRecord(uid, split_sentences(str(note), "pmc_patients"),
                                         "PMC-Patients-V1", uid, pmid)
                counts["retained"] += 1
                if summary is not None:
                    summary.update(counts)
                yield record
    except (ValueError, FileNotFoundError):
        raise
    except Exception as exc:
        raise ValueError(f"malformed PMC-Patients CSV: {exc}") from exc
    finally:
        if summary is not None:
            summary.update(counts)


def _verified_split_ids(split_dir: Path) -> dict[Split, list[str]]:
    """Load membership metadata and verify its frozen counts, disjointness, and hashes."""
    try:
        split_ids = load_split(split_dir)
        info = json.loads((split_dir / "split_info.json").read_text(encoding="utf-8"))
        hashes = file_hashes(split_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("frozen split metadata is missing or malformed") from exc
    if not isinstance(info, dict) or info.get("sha256") != hashes:
        raise ValueError("frozen split metadata hashes do not match split files")
    if {split: len(ids) for split, ids in split_ids.items()} != EXPECTED_SPLIT_SIZES:
        raise ValueError("frozen split metadata must contain 37/73/73 patient IDs")
    all_ids = [patient_id for ids in split_ids.values() for patient_id in ids]
    if any(not isinstance(patient_id, str) or not patient_id for patient_id in all_ids):
        raise ValueError("frozen split metadata contains a malformed patient ID")
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("frozen split metadata contains duplicate patient IDs")
    return split_ids


def load_development_candidates(raw: Path, split_dir: Path = SPLIT_DIR) -> list[CandidateRecord]:
    """Fail closed because cohort JSONL has no safe split-selective note access path.

    The shared patient loader parses every cohort query record before applying split IDs.
    Since JSONL has no split index, using it would materialize calibration/test note text.
    Validate frozen membership metadata first, then refuse before opening query files.
    """
    _verified_split_ids(split_dir)
    raise RuntimeError(
        "cannot load development notes safely: cohort queries.jsonl files mix splits and "
        "have no split-selective index; loading them would read calibration/test note contents"
    )


def validate_evidence(candidate: CandidateRecord, evidence: Iterable[Evidence], *,
                      required: bool = True) -> None:
    """Validate candidate binding, non-empty evidence when required, IDs, and verbatim text."""
    evidence = list(evidence)
    if required and not evidence:
        raise ValueError("evidence is required for this candidate")
    seen: set[int] = set()
    for item in evidence:
        if item.candidate_id != candidate.patient_id:
            raise ValueError("evidence belongs to a different candidate record")
        if item.sentence_id in seen:
            raise ValueError(f"duplicate evidence sentence id: {item.sentence_id}")
        seen.add(item.sentence_id)
        if item.sentence_id not in candidate.sentences:
            raise ValueError(f"evidence sentence id does not exist: {item.sentence_id}")
        if item.text != candidate.sentences[item.sentence_id]:
            raise ValueError(f"evidence text is not verbatim for sentence id {item.sentence_id}")


def check_identifier_overlap(candidates: Iterable[CandidateRecord], split_dir: Path) -> dict:
    """Check exact ID collisions without reading calibration/test note contents.

    This does not establish content-level overlap; PMC and evaluation IDs use different
    namespaces. A zero match must not be treated as clearance for training.
    """
    frozen = load_split(split_dir)
    evaluation_ids = set(frozen[Split.CAL]) | set(frozen[Split.TEST])
    overlap = sorted({r.patient_id for r in candidates} & evaluation_ids)
    if overlap:
        raise ValueError(f"candidate IDs overlap frozen evaluation IDs ({len(overlap)} records)")
    return {"exact_id_overlap": 0, "evaluation_id_count": len(evaluation_ids),
            "content_overlap_verified": False}
