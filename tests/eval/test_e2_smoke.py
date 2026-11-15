"""T38: the E2 smoke runner, with a scripted checker and two made-up trials."""

import json

import pytest

from trialguide.data.patients import Patient
from trialguide.eval import e2_smoke as sm
from trialguide.schemas import InfoCategory, RuleCheck, RuleKind, RuleStatus, Split


def _trial(tid, inc, exc, text):
    return {"_id": tid, "title": text, "text": text,
            "metadata": {"diseases_list": [text], "inclusion_criteria": "inclusion criteria: \n\n " + " \n\n ".join(inc),
                         "exclusion_criteria": "exclusion criteria: \n\n " + " \n\n ".join(exc)}}


TRIALS = [_trial("NCT1", ["Adults with breast cancer", "ECOG 0-1"], ["Prior chemotherapy"], "breast cancer chemotherapy"),
          _trial("NCT2", ["Lung adenocarcinoma"], ["Pregnancy"], "lung adenocarcinoma"),
          _trial("NCT3", ["Children with leukaemia"], ["Heart failure"], "leukaemia children")]


def _patient(pid, sents):
    return Patient(pid, "pmc_patients", Split.CAL, dict(enumerate(sents)))


PATIENTS = {"p1": _patient("p1", ["A 52 year old woman with breast cancer.", "She has ECOG 1.", "She denies chemotherapy.", "BP is normal."]),
            "p2": _patient("p2", ["A 60 year old man with lung adenocarcinoma.", "He is a smoker.", "No pregnancy.", "Scan done."])}


def fake_check(note, rule):
    """Settles every rule while sentence 0 is visible; otherwise reports not enough information."""
    if 0 in note.sentences:
        status = RuleStatus.MET if rule.kind == RuleKind.INCLUSION else RuleStatus.NOT_MET
        return RuleCheck(patient_id=note.patient_id, trial_id=rule.trial_id, rule_id=rule.rule_id, kind=rule.kind,
                         status=status, category=InfoCategory.STATED, evidence_ids=[0], confidence=0.99, n_samples=1)
    return RuleCheck(patient_id=note.patient_id, trial_id=rule.trial_id, rule_id=rule.rule_id, kind=rule.kind,
                     status=RuleStatus.NOT_ENOUGH_INFO, category=InfoCategory.MISSING, confidence=0.5, n_samples=1)


def test_sampling_is_seeded_and_bounded():
    ids = [f"p{i}" for i in range(30)]
    assert sm.sample_patients(ids, 5) == sm.sample_patients(list(reversed(ids)), 5)
    assert len(sm.sample_patients(ids, 100)) == 30
    assert sm.sample_patients(ids, 5, seed=1) != sm.sample_patients(ids, 5, seed=2)


def test_top_trials_finds_the_matching_trial():
    assert sm.top_trials("woman with breast cancer after chemotherapy", TRIALS, 1)[0]["_id"] == "NCT1"
    assert sm.top_trials("adenocarcinoma of the lung", TRIALS, 1)[0]["_id"] == "NCT2"
    assert [t["_id"] for t in sm.top_trials("anything", TRIALS, 2)] and len(sm.top_trials("anything", TRIALS, 2)) == 2


def test_trialgpt_form_feeds_rules_of_trial():
    from trialguide.pipeline import rules_of_trial
    rules = rules_of_trial(sm.trialgpt_form(TRIALS[0]))
    assert [r.rule_id for r in rules] == ["inc_0", "inc_1", "exc_0"]


def test_run_writes_both_conditions_with_all_policies(tmp_path):
    chosen = sm.run(fake_check, PATIENTS, TRIALS, 1, 0.5, tmp_path, workers=2)
    assert chosen == {"p1": ["NCT1"], "p2": ["NCT2"]}
    for cond in ("full", "rule50"):
        rows = [json.loads(l) for l in open(tmp_path / f"checks_{cond}.jsonl", encoding="utf-8")]
        assert {r["patient_id"] for r in rows} == {"p1", "p2"}
        assert set(rows[0]["actions"]) == {"B6", "B2", "B5"}


def test_summary_counts_lost_rules_against_the_full_note(tmp_path):
    sm.run(fake_check, PATIENTS, TRIALS, 1, 0.5, tmp_path, workers=2)
    s = sm.summarise(tmp_path, "rule50", n_boot=100)
    assert s["patients"] == 2 and s["full_note_settled_share"] == 1.0           # the fake settles everything on the full note
    assert 0.0 <= s["masked_unresolved_share_b6"] <= 1.0
    # rules lost to masking only when sentence 0 was hidden; B6 never infers, so it recovers nothing there
    for p in ("B2", "B5"):
        assert s[p]["lost_rules"] >= 0 and s[p]["recovered"][0] == 0.0


def test_merge_parts_summarises_all_patients(tmp_path):
    for part, pid in (("a", "p1"), ("b", "p2")):
        d = tmp_path / part
        chosen = sm.run(fake_check, {pid: PATIENTS[pid]}, TRIALS, 1, 0.5, d, workers=2)
        (d / "chosen.json").write_text(json.dumps({"seed": 1, "patients": chosen, "fraction": 0.5, "trials_per_patient": 1}))
    res = sm.merge_parts([tmp_path / "a", tmp_path / "b"], tmp_path / "merged", "rule50", n_boot=50)
    assert res["patients"] == 2
    assert json.loads((tmp_path / "merged" / "chosen.json").read_text())["patients"] == {"p1": ["NCT1"], "p2": ["NCT2"]}
