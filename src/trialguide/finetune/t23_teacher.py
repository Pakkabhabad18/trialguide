"""Fail-closed T23 teacher-label contracts; no inference or label generation is performed.

T23 intends to use Qwen3-8B, routed through the shared LLM client. The repository currently
has no verifiable approval-artifact format, approved reference/independent label sources,
PMC overlap clearance, or client-returned model revision. Prompts and label serialization
therefore remain blocked.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping

from trialguide.data.splits import file_hashes, load_split
from trialguide.finetune import e1_data
from trialguide.schemas import InfoCategory, RuleKind, RuleStatus, Split, TrialRule

INTENDED_TEACHER_MODEL = "Qwen/Qwen3-8B"
VERIFIED_TEACHER_REVISION: None = None
APPROVAL_ARTIFACT_FORMAT: None = None
_CANDIDATE_SOURCE_VERIFIER_AVAILABLE = False
_EXPERT_REFERENCE_SOURCE_AVAILABLE = False
_INDEPENDENT_SAMPLE_SOURCE_AVAILABLE = False
_PMC_OVERLAP_CLEARANCE_AVAILABLE = False
_MODEL_REVISION_EXPOSED_BY_CLIENT = False
_VALID_STATUSES = {status.value for status in RuleStatus}
_VALID_CATEGORIES = {category.value for category in InfoCategory}


class PatientSource(str, Enum):
    DEVELOPMENT = "development"
    PMC_PATIENTS_V1 = "pmc_patients_v1"


@dataclass(frozen=True)
class PatientProvenance:
    """Origin claim for a note. It is metadata, not proof that the source was verified."""

    source: PatientSource
    patient_id: str
    source_record_id: str
    split: str | None = None
    source_pmid: str | None = None

    def validate(self) -> None:
        if not isinstance(self.source, PatientSource):
            raise ValueError("unsupported patient source")
        if not isinstance(self.patient_id, str) or not self.patient_id.strip() \
                or not isinstance(self.source_record_id, str) or not self.source_record_id.strip():
            raise ValueError("patient provenance requires patient_id and source_record_id")
        if self.source == PatientSource.DEVELOPMENT:
            if self.split != Split.DEV.value or self.source_record_id != self.patient_id:
                raise ValueError("development provenance must identify a dev split record")
            if self.source_pmid is not None:
                raise ValueError("development provenance cannot include a PMC PMID")
        elif self.split is not None or not isinstance(self.source_pmid, str) \
                or not self.source_pmid.isdecimal():
            raise ValueError("PMC provenance requires no split and a numeric PMID")


@dataclass(frozen=True)
class AgreementMetadata:
    """Descriptive agreement fields only; never treated as proof of approval."""

    patient_id: str
    trial_id: str
    rule_id: str
    reference_source: str
    reference_record_id: str
    reference_status: RuleStatus
    independent_sample_id: str
    independent_procedure_id: str
    independent_seed: int
    independent_status: RuleStatus


@dataclass(frozen=True)
class TeacherLabelRecord:
    """T12-compatible label plus provenance; revision remains unknown until verified."""

    provenance: PatientProvenance
    trial_id: str
    rule_id: str
    teacher_model: str
    teacher_version: str | None
    seed: int
    status: RuleStatus
    category: InfoCategory
    evidence_sentence_ids: tuple[int, ...]
    reasoning: str
    fact_key: str | None = None
    agreement: AgreementMetadata | None = None

    def validate(self) -> None:
        self.provenance.validate()
        if not isinstance(self.trial_id, str) or not self.trial_id.strip() \
                or not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise ValueError("trial_id and rule_id are required")
        if self.teacher_model != INTENDED_TEACHER_MODEL:
            raise ValueError(f"T23 intends {INTENDED_TEACHER_MODEL}; no fallback model is allowed")
        if self.teacher_version is not None:
            raise ValueError("actual teacher revision is unresolved and must not be claimed")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.status, RuleStatus) or not isinstance(self.category, InfoCategory):
            raise ValueError("status and category must use the shared schemas")
        if not isinstance(self.reasoning, str) or not self.reasoning.strip():
            raise ValueError("reasoning is required")
        if self.fact_key is not None and (not isinstance(self.fact_key, str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.fact_key)):
            raise ValueError("fact_key must be null or normalized snake_case")
        if not isinstance(self.evidence_sentence_ids, tuple) or any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0
                for i in self.evidence_sentence_ids):
            raise ValueError("evidence sentence IDs must be a tuple of non-negative integers")
        if len(set(self.evidence_sentence_ids)) != len(self.evidence_sentence_ids):
            raise ValueError("evidence sentence IDs must be unique")
        if self.category in (InfoCategory.STATED, InfoCategory.INFERRED) \
                and not self.evidence_sentence_ids:
            raise ValueError("stated/inferred labels require evidence sentence IDs")
        if self.category in (InfoCategory.ASSUMPTION, InfoCategory.MISSING) and (
                self.status != RuleStatus.NOT_ENOUGH_INFO or self.evidence_sentence_ids):
            raise ValueError("assumption/missing labels require not_enough_info and empty evidence")
        if self.agreement is not None:
            raise ValueError("agreement metadata cannot be trusted without approved source verification")


@dataclass(frozen=True)
class T23PreflightReport:
    ready: bool
    blockers: tuple[str, ...]
    checks: Mapping[str, bool]


@dataclass(frozen=True)
class ParsedTeacherOutput:
    """Parsed T12 response tied to one candidate's sentence IDs; not an approved label."""

    status: RuleStatus
    category: InfoCategory
    evidence_sentence_ids: tuple[int, ...]
    reasoning: str
    fact_key: str | None


@dataclass(frozen=True)
class ReferenceLabel:
    record_id: str
    patient_id: str
    trial_id: str
    rule_id: str
    status: RuleStatus


@dataclass(frozen=True)
class IndependentSampleLabel:
    sample_id: str
    procedure_id: str
    patient_id: str
    trial_id: str
    rule_id: str
    seed: int
    status: RuleStatus


def preflight_t23(raw: Path, split_dir: Path | None = None) -> T23PreflightReport:
    """Inspect split/data metadata without reading any patient-note contents.

    No approval artifact format exists in this repository, so this preflight cannot become
    ready from caller booleans, arbitrary files, or a fabricated report.
    """
    split_dir = split_dir or e1_data.E2_EXCLUSION_RELATIVE_PATH.parents[1]
    blockers: list[str] = []
    checks: dict[str, bool] = {}
    try:
        ids = load_split(split_dir)
        sizes_ok = {s: len(ids[s]) for s in Split} == {
            Split.DEV: 37, Split.CAL: 73, Split.TEST: 73}
        all_ids = ids[Split.DEV] + ids[Split.CAL] + ids[Split.TEST]
        checks["frozen_split_sizes_valid"] = sizes_ok
        checks["frozen_split_ids_disjoint"] = len(all_ids) == len(set(all_ids))
        info_path = split_dir / "split_info.json"
        if info_path.is_file():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            checks["frozen_split_hashes_valid"] = file_hashes(split_dir) == info.get("sha256")
        else:
            checks["frozen_split_hashes_valid"] = False
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        checks.update(frozen_split_sizes_valid=False, frozen_split_ids_disjoint=False,
                      frozen_split_hashes_valid=False)
    for key, message in (
            ("frozen_split_sizes_valid", "frozen split sizes must be 37/73/73"),
            ("frozen_split_ids_disjoint", "frozen split patient IDs must be disjoint"),
            ("frozen_split_hashes_valid", "frozen split hashes must match split_info.json")):
        if not checks[key]:
            blockers.append(message)

    query_paths = [raw / "trialgpt" / cohort / "queries.jsonl"
                   for cohort in ("sigir", "trec_2021", "trec_2022")]
    checks["development_note_sources_present"] = all(path.is_file() for path in query_paths)
    if not checks["development_note_sources_present"]:
        blockers.append("one or more TrialGPT note source files are missing")
    checks["pinned_pmc_csv_present"] = (raw / e1_data.PMC_RELATIVE_PATH).is_file()
    if not checks["pinned_pmc_csv_present"]:
        blockers.append("pinned PMC-Patients V1 CSV is missing")
    try:
        e1_data._read_exclusions(split_dir / "e2" / "e1_exclude_pmids.txt")
        checks["frozen_e2_exclusions_valid"] = True
    except (OSError, ValueError, json.JSONDecodeError):
        checks["frozen_e2_exclusions_valid"] = False
    if not checks["frozen_e2_exclusions_valid"]:
        blockers.append("frozen 276-PMID E2 exclusion list is missing or invalid")

    checks.update(
        approval_artifact_format_defined=APPROVAL_ARTIFACT_FORMAT is not None,
        intended_teacher_is_qwen3_8b=INTENDED_TEACHER_MODEL == "Qwen/Qwen3-8B",
        actual_teacher_revision_verified=(VERIFIED_TEACHER_REVISION is not None
                                         and _MODEL_REVISION_EXPOSED_BY_CLIENT),
        candidate_source_verifier_available=_CANDIDATE_SOURCE_VERIFIER_AVAILABLE,
        expert_reference_source_available=_EXPERT_REFERENCE_SOURCE_AVAILABLE,
        independent_sample_source_available=_INDEPENDENT_SAMPLE_SOURCE_AVAILABLE,
        pmc_overlap_clearance_available=_PMC_OVERLAP_CLEARANCE_AVAILABLE,
    )
    blockers.extend((
        "no repository-defined, verifiable approval-artifact format or actual approval evidence exists",
        "Qwen3-8B is the intended T23 teacher, but the actual served model and immutable revision are unverified",
        "no candidate-source verifier can establish prompt provenance without opening sealed note contents",
        "no approved expert reference-label source is available; the annotation parquet remains forbidden",
        "PMC-Patients content-level overlap with evaluation notes has not been cleared",
        "no approved independent second-sample source or procedure is available",
    ))
    return T23PreflightReport(False, tuple(blockers), checks)


def _require_verified_preflight(report: T23PreflightReport | None, operation: str) -> None:
    """Reject even caller-constructed ready reports while approval verification is undefined."""
    if not isinstance(report, T23PreflightReport):
        raise ValueError(f"{operation} requires a T23 preflight report")
    if APPROVAL_ARTIFACT_FORMAT is None:
        raise ValueError("T23 is blocked: no verifiable approval-artifact format is defined")
    if not report.ready or report.blockers:
        raise ValueError(f"T23 is blocked: preflight is not ready for {operation}")


def build_teacher_prompt(candidate: e1_data.CandidateRecord, rule: TrialRule,
                         preflight: T23PreflightReport | None = None) -> list[dict[str, str]]:
    """Build a T12-compatible prompt only after source and approval verifiers exist."""
    _require_verified_preflight(preflight, "prompt construction")
    if not _CANDIDATE_SOURCE_VERIFIER_AVAILABLE:
        raise ValueError("T23 is blocked: candidate text is not verifiably bound to an approved source")
    if not isinstance(candidate, e1_data.CandidateRecord):
        raise ValueError("prompt input must be a validated E1 candidate record")
    if candidate.split != Split.DEV.value:
        raise ValueError("T23 prompt candidates must come from the development split")
    if candidate.source == "PMC-Patients-V1" and not _PMC_OVERLAP_CLEARANCE_AVAILABLE:
        raise ValueError("PMC prompt source has not been cleared for evaluation overlap")
    if candidate.source != "PMC-Patients-V1" and not candidate.source.startswith("TrialGPT-"):
        raise ValueError("unverified prompt source")
    if not candidate.sentences:
        raise ValueError("teacher prompt requires a non-empty candidate note")
    if rule.kind not in (RuleKind.INCLUSION, RuleKind.EXCLUSION):
        raise ValueError("unsupported trial rule kind")
    note_text = "\n".join(f"{sid}. {text}" for sid, text in sorted(candidate.sentences.items()))
    system = """You are a conservative clinical-trial rule checker. Use only the supplied note.
Return exactly one JSON object with keys status, category, evidence_ids, explanation, fact_key.
Use statuses met, not_met, not_enough_info, not_applicable and categories stated, inferred,
assumption, missing. Stated/inferred labels cite evidence sentence IDs from this note. For
assumption/missing use not_enough_info and an empty evidence_ids list. Do not include identifiers."""
    user = f"""Rule kind: {rule.kind.value}
Rule text: {rule.text}

Visible patient note:
{note_text}

Return the required JSON object only."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """JSON object hook that rejects duplicate keys instead of silently taking the last."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def parse_teacher_output(raw: str, candidate: e1_data.CandidateRecord) -> ParsedTeacherOutput:
    """Strictly parse T12 output and bind evidence IDs to this candidate's sentence map."""
    if not isinstance(candidate, e1_data.CandidateRecord):
        raise ValueError("teacher output must be bound to a candidate record")
    try:
        data = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("teacher response must be one JSON object without duplicate keys") from exc
    required = {"status", "category", "evidence_ids", "explanation", "fact_key"}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError("teacher response must contain exactly the T12 checker fields")
    if not isinstance(data["status"], str) or data["status"] not in _VALID_STATUSES:
        raise ValueError("unsupported teacher label status")
    if not isinstance(data["category"], str) or data["category"] not in _VALID_CATEGORIES:
        raise ValueError("unsupported teacher label category")
    ids = data["evidence_ids"]
    if not isinstance(ids, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
        raise ValueError("evidence_ids must be integer sentence IDs")
    if len(ids) != len(set(ids)) or any(i not in candidate.sentences for i in ids):
        raise ValueError("evidence IDs must be unique and belong to this candidate")
    status, category = RuleStatus(data["status"]), InfoCategory(data["category"])
    if category in (InfoCategory.STATED, InfoCategory.INFERRED) and not ids:
        raise ValueError("stated/inferred output requires candidate evidence")
    if category in (InfoCategory.ASSUMPTION, InfoCategory.MISSING) and (
            status != RuleStatus.NOT_ENOUGH_INFO or ids):
        raise ValueError("assumption/missing output requires not_enough_info and no evidence")
    explanation = data["explanation"]
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("explanation must be non-empty text")
    key = data["fact_key"]
    if key is not None and (not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key)):
        raise ValueError("fact_key must be null or normalized snake_case")
    return ParsedTeacherOutput(status, category, tuple(sorted(ids)), explanation, key)


def validate_teacher_output(raw: str, candidate: e1_data.CandidateRecord, rule: TrialRule,
                            seed: int) -> TeacherLabelRecord:
    """Validate a candidate-bound response, without asserting actual model identity/revision."""
    parsed = parse_teacher_output(raw, candidate)
    if not isinstance(rule.trial_id, str) or not rule.trial_id.strip() \
            or not isinstance(rule.rule_id, str) or not rule.rule_id.strip():
        raise ValueError("trial and rule identifiers are required")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if candidate.split == Split.DEV.value and candidate.source.startswith("TrialGPT-"):
        source = PatientSource.DEVELOPMENT
        provenance = PatientProvenance(source, candidate.patient_id, candidate.source_record_id,
                                       split=Split.DEV.value)
    elif candidate.source == "PMC-Patients-V1":
        source = PatientSource.PMC_PATIENTS_V1
        provenance = PatientProvenance(source, candidate.patient_id, candidate.source_record_id,
                                       source_pmid=candidate.source_pmid)
    else:
        raise ValueError("candidate source or split is not eligible for T23")
    provenance.validate()
    return TeacherLabelRecord(provenance, rule.trial_id, rule.rule_id, INTENDED_TEACHER_MODEL,
                              None, seed, parsed.status, parsed.category,
                              parsed.evidence_sentence_ids, parsed.reasoning, parsed.fact_key)


def apply_agreement_gate(record: TeacherLabelRecord, reference: ReferenceLabel | None,
                         independent: IndependentSampleLabel | None,
                         preflight: T23PreflightReport | None = None) -> TeacherLabelRecord:
    """Agreement cannot be accepted until a verifiable approval format and sources exist."""
    _require_verified_preflight(preflight, "agreement acceptance")
    raise ValueError("T23 is blocked: expert reference and independent sample sources are unavailable")


def _record_dict(record: TeacherLabelRecord) -> dict:
    record.validate()
    return {
        "provenance": {**asdict(record.provenance), "source": record.provenance.source.value},
        "trial_id": record.trial_id,
        "rule_id": record.rule_id,
        "teacher_model": record.teacher_model,
        "teacher_version": record.teacher_version,
        "seed": record.seed,
        "status": record.status.value,
        "category": record.category.value,
        "evidence_sentence_ids": list(record.evidence_sentence_ids),
        "reasoning": record.reasoning,
        "fact_key": record.fact_key,
        "agreement": None,
    }


def _record_identity(record: TeacherLabelRecord) -> tuple[str, str, str, str]:
    return (record.provenance.source.value, record.provenance.source_record_id,
            record.trial_id, record.rule_id)


def canonical_record_order(records: Iterable[TeacherLabelRecord]) -> list[TeacherLabelRecord]:
    """Validate uniqueness and return a stable canonical ordering without approving labels."""
    records = list(records)
    seen = set()
    for record in records:
        record.validate()
        identity = _record_identity(record)
        if identity in seen:
            raise ValueError(f"duplicate or conflicting teacher record: {identity}")
        seen.add(identity)
    return sorted(records, key=lambda record: (
        _record_identity(record),
        json.dumps(_record_dict(record), sort_keys=True, separators=(",", ":"))))


def dumps_jsonl(records: Iterable[TeacherLabelRecord],
                preflight: T23PreflightReport | None = None) -> str:
    """Serialize only after source verification; no such approval verifier exists yet."""
    ordered = canonical_record_order(records)
    if not ordered:
        return ""
    _require_verified_preflight(preflight, "JSONL serialization")
    raise ValueError("T23 is blocked: expert reference and independent sample sources are unavailable")


def dump_jsonl(path: Path, records: Iterable[TeacherLabelRecord],
               preflight: T23PreflightReport | None = None) -> None:
    """Write labels only when an approved serializer is available; currently always blocked."""
    text = dumps_jsonl(records, preflight)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _record_from_dict(data: object) -> TeacherLabelRecord:
    """Decode the exact metadata schema; caller separately verifies external label sources."""
    fields = {"provenance", "trial_id", "rule_id", "teacher_model", "teacher_version", "seed",
              "status", "category", "evidence_sentence_ids", "reasoning", "fact_key", "agreement"}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError("teacher JSONL record has unexpected or missing fields")
    if data["agreement"] is not None:
        raise ValueError("serialized agreement metadata is not source-verifiable")
    p = data["provenance"]
    if not isinstance(p, dict) or set(p) != {"source", "patient_id", "source_record_id", "split", "source_pmid"}:
        raise ValueError("teacher JSONL provenance has unexpected or missing fields")
    provenance = PatientProvenance(PatientSource(p["source"]), p["patient_id"],
                                   p["source_record_id"], p["split"], p["source_pmid"])
    record = TeacherLabelRecord(
        provenance, data["trial_id"], data["rule_id"], data["teacher_model"], data["teacher_version"],
        data["seed"], RuleStatus(data["status"]), InfoCategory(data["category"]),
        tuple(data["evidence_sentence_ids"]), data["reasoning"], data["fact_key"])
    record.validate()
    return record


def loads_jsonl(text: str, preflight: T23PreflightReport | None = None) -> list[TeacherLabelRecord]:
    """Strictly parse JSONL, reject duplicates, and remain blocked without approved sources."""
    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL line at {line_no}")
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            records.append(_record_from_dict(value))
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid teacher-label JSONL record on line {line_no}: {exc}") from exc
    ordered = canonical_record_order(records)
    if not ordered:
        return []
    _require_verified_preflight(preflight, "JSONL loading")
    raise ValueError("T23 is blocked: expert reference and independent sample sources are unavailable")


def load_jsonl(path: Path, preflight: T23PreflightReport | None = None) -> list[TeacherLabelRecord]:
    """Load metadata but refuse label records until sources can be validated."""
    return loads_jsonl(path.read_text(encoding="utf-8"), preflight)


__all__ = ["INTENDED_TEACHER_MODEL", "VERIFIED_TEACHER_REVISION", "PatientSource",
           "PatientProvenance", "AgreementMetadata", "TeacherLabelRecord", "T23PreflightReport",
           "ParsedTeacherOutput", "ReferenceLabel", "IndependentSampleLabel", "preflight_t23",
           "build_teacher_prompt", "parse_teacher_output", "validate_teacher_output",
           "apply_agreement_gate", "canonical_record_order", "dumps_jsonl", "dump_jsonl",
           "loads_jsonl", "load_jsonl"]
