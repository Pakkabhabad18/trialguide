"""Synthetic-only regression tests for the fail-closed T23 interfaces."""

import json

import pytest

from trialguide.finetune import e1_data, t23_teacher as t23
from trialguide.schemas import InfoCategory, RuleKind, RuleStatus, TrialRule


def candidate(**changes):
    values = dict(patient_id="synthetic-p1", sentences={0: "Synthetic sentence."},
                  source="TrialGPT-synthetic", source_record_id="synthetic-p1", split="dev")
    values.update(changes)
    return e1_data.CandidateRecord(**values)


def record(**changes):
    values = dict(
        provenance=t23.PatientProvenance(t23.PatientSource.DEVELOPMENT, "synthetic-p1",
                                         "synthetic-p1", split="dev"),
        trial_id="NCT-synthetic", rule_id="rule-synthetic",
        teacher_model="Qwen/Qwen3-8B", teacher_version=None, seed=7,
        status=RuleStatus.MET, category=InfoCategory.STATED,
        evidence_sentence_ids=(0,), reasoning="Synthetic rationale.", fact_key="age")
    values.update(changes)
    return t23.TeacherLabelRecord(**values)


def test_intended_model_is_qwen3_8b_revision_unverified():
    assert t23.INTENDED_TEACHER_MODEL == "Qwen/Qwen3-8B"
    assert t23.VERIFIED_TEACHER_REVISION is None
    record().validate()
    with pytest.raises(ValueError, match="no fallback model"):
        record(teacher_model="Qwen/Qwen3-32B").validate()


def test_preflight_never_marks_ready_from_arbitrary_files_or_caller_flags(tmp_path):
    # Neither unrelated files nor obsolete caller-supplied approval flags are evidence.
    (tmp_path / "approval.json").write_text('{"approved": true}')
    with pytest.raises(TypeError):
        t23.preflight_t23(tmp_path, teacher_model_approved=True)
    report = t23.preflight_t23(tmp_path, tmp_path / "splits")
    assert not report.ready
    assert not report.checks["approval_artifact_format_defined"]
    assert not report.checks["expert_reference_source_available"]
    assert not report.checks["independent_sample_source_available"]


def test_forged_ready_preflight_is_rejected_for_prompt_and_serialization():
    forged = t23.T23PreflightReport(True, (), {"everything": True})
    rule = TrialRule("NCT-synthetic", "rule-synthetic", RuleKind.INCLUSION, "Synthetic rule")
    with pytest.raises(ValueError, match="no verifiable approval-artifact format"):
        t23.build_teacher_prompt(candidate(), rule, forged)
    with pytest.raises(ValueError, match="no verifiable approval-artifact format"):
        t23.dumps_jsonl([record()], forged)


@pytest.mark.parametrize("split", ["cal", "test"])
def test_spoofed_development_provenance_cannot_prompt_calibration_or_test_text(split):
    forged = t23.T23PreflightReport(True, (), {"all": True})
    # The synthetic string deliberately represents sealed-source text without opening it.
    cal_or_test = candidate(sentences={0: f"synthetic {split} note sentinel"}, split=split)
    dev_spoof = candidate(sentences=cal_or_test.sentences, split="dev")
    rule = TrialRule("NCT-synthetic", "rule-synthetic", RuleKind.INCLUSION, "Synthetic rule")
    for item in (cal_or_test, dev_spoof):
        with pytest.raises(ValueError, match="no verifiable approval-artifact format"):
            t23.build_teacher_prompt(item, rule, forged)


def test_fake_agreement_metadata_cannot_be_validated_or_serialized():
    fake = t23.AgreementMetadata(
        "synthetic-p1", "NCT-synthetic", "rule-synthetic", "invented", "fake-ref",
        RuleStatus.MET, "fake-sample", "invented-procedure", 8, RuleStatus.MET)
    with pytest.raises(ValueError, match="cannot be trusted"):
        record(agreement=fake).validate()
    with pytest.raises(ValueError, match="cannot be trusted"):
        t23.dumps_jsonl([record(agreement=fake)])


def _output(**changes):
    value = {"status": "met", "category": "stated", "evidence_ids": [0],
             "explanation": "Synthetic rationale.", "fact_key": "age"}
    value.update(changes)
    return json.dumps(value)


def test_evidence_ids_must_exist_on_the_validated_candidate():
    with pytest.raises(ValueError, match="belong to this candidate"):
        t23.parse_teacher_output(_output(evidence_ids=[99]), candidate())
    other_candidate = candidate(sentences={9: "A different synthetic note."})
    with pytest.raises(ValueError, match="belong to this candidate"):
        t23.parse_teacher_output(_output(evidence_ids=[0]), other_candidate)


def test_duplicate_conflicting_records_are_rejected():
    with pytest.raises(ValueError, match="duplicate or conflicting"):
        t23.canonical_record_order([record(), record(reasoning="Different synthetic rationale.")])


def test_canonical_order_is_stable_for_distinct_synthetic_records():
    first = record()
    second = record(provenance=t23.PatientProvenance(
        t23.PatientSource.DEVELOPMENT, "synthetic-p2", "synthetic-p2", split="dev"))
    a = t23.canonical_record_order([second, first])
    b = t23.canonical_record_order([first, second])
    assert [t23._record_identity(x) for x in a] == [t23._record_identity(x) for x in b]


def test_duplicate_json_object_keys_are_rejected():
    duplicate = ('{"status":"met","status":"not_met","category":"stated",'
                 '"evidence_ids":[0],"explanation":"x","fact_key":null}')
    with pytest.raises(ValueError, match="duplicate keys"):
        t23.parse_teacher_output(duplicate, candidate())


def test_jsonl_duplicate_or_unapproved_records_remain_blocked():
    encoded = json.dumps(t23._record_dict(record()), sort_keys=True)
    with pytest.raises(ValueError, match="duplicate or conflicting"):
        t23.loads_jsonl(encoded + "\n" + encoded)
    forged = t23.T23PreflightReport(True, (), {"all": True})
    with pytest.raises(ValueError, match="no verifiable approval-artifact format"):
        t23.loads_jsonl(encoded, forged)
    assert t23.loads_jsonl("") == []


def test_missing_approval_and_independent_sample_evidence_keep_preflight_blocked(tmp_path):
    report = t23.preflight_t23(tmp_path, tmp_path / "splits")
    assert not report.ready
    assert not report.checks["approval_artifact_format_defined"]
    assert not report.checks["independent_sample_source_available"]
    assert any("approval-artifact format" in item for item in report.blockers)
    assert any("independent second-sample" in item for item in report.blockers)
