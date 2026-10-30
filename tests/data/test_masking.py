"""T10: masking and fact splitting."""

import json
from pathlib import Path

import pandas as pd
import pytest

from trialguide.data import masking as mk
from trialguide.data import patients as pt
from trialguide.schemas import FactSource, MaskType, Split


def _patient(n=10, split=Split.DEV, pid="trec-20211"):
    return pt.Patient(pid, "trec_2021", split, {i: f"sentence number {i} about topic {i}." for i in range(n)})


# ---- how many to hide, seeds ----

@pytest.mark.parametrize("n,f,want", [(10, 0.4, 4), (10, 0.2, 2), (10, 0.6, 6), (6, 0.2, 1),
                                      (3, 0.6, 2), (2, 0.9, 1), (1, 0.4, 0), (10, 0.0, 0)])
def test_n_to_hide(n, f, want):
    assert mk.n_to_hide(n, f) == want       # at least 1, never the whole note


def test_seed_differs_by_split_patient_and_fraction():
    base = mk.derive_seed(0, "dev", "p1", "random", "0.40")
    assert base == mk.derive_seed(0, "dev", "p1", "random", "0.40")
    others = [mk.derive_seed(0, "cal", "p1", "random", "0.40"), mk.derive_seed(0, "dev", "p2", "random", "0.40"),
              mk.derive_seed(0, "dev", "p1", "random", "0.20"), mk.derive_seed(1, "dev", "p1", "random", "0.40")]
    assert len({base, *others}) == 5


# ---- random masking ----

def test_random_mask_hides_exactly_and_is_reproducible():
    p = _patient(10)
    a, b = mk.mask_random(p, 0.4), mk.mask_random(p, 0.4)
    assert a == b and len(a.hidden_ids) == 4
    assert a.visible.mask_type == MaskType.RANDOM and a.visible.mask_fraction == 0.4


def test_visible_note_never_contains_hidden_text():
    p = _patient(10)
    m = mk.mask_random(p, 0.6)
    shown = m.visible.text()
    for i in m.hidden_ids:
        assert p.sentences[i] not in shown
    assert set(m.visible.sentences) == set(p.sentences) - set(m.hidden_ids)
    assert m.visible.sentences.keys().isdisjoint(m.hidden_ids)   # original sentence ids are kept


def test_fraction_is_nested_not_required_but_masks_differ_by_split():
    dev, cal = _patient(20, Split.DEV), _patient(20, Split.CAL)
    assert mk.mask_random(dev, 0.4).hidden_ids != mk.mask_random(cal, 0.4).hidden_ids


def test_mask_none_is_the_full_note():
    p = _patient(5)
    m = mk.mask_none(p)
    assert m.hidden_ids == () and m.visible.sentences == p.sentences and m.visible.mask_type == MaskType.NONE


# ---- rule-targeted ----

def test_split_criteria_keeps_first_criterion_after_header():
    text = "inclusion criteria: \n\n Patients with chronic kidney disease \n\n Age over 18"
    assert mk.split_criteria(text) == ["Patients with chronic kidney disease", "Age over 18"]
    assert mk.split_criteria("") == mk.split_criteria(None) == []
    # header and first criterion in one chunk must not lose the criterion
    assert mk.split_criteria("inclusion criteria: acute myocardial infarction, age 40") == \
        ["acute myocardial infarction, age 40"]


def test_rule_demand_prefers_sentences_that_rules_ask_about():
    sents = {0: "She has chronic kidney disease with reduced eGFR.", 1: "She enjoys gardening on weekends.",
             2: "Her creatinine is elevated."}
    rules = [["Patients with chronic kidney disease", "eGFR below 30"], ["Elevated creatinine at screening"]]
    d = mk.rule_demand(sents, rules)
    assert d[1] < d[0] and d[1] < d[2]


def test_rule_mask_hides_highest_demand_and_breaks_ties_by_seed():
    p = _patient(5)
    m = mk.mask_rule(p, {0: 0.1, 1: 0.9, 2: 0.2, 3: 0.8, 4: 0.0}, 0.4)
    assert m.hidden_ids == (1, 3)
    flat = {i: 0.0 for i in range(10)}
    q = _patient(10)
    assert mk.mask_rule(q, flat, 0.4) == mk.mask_rule(q, flat, 0.4)
    assert len(mk.mask_rule(q, flat, 0.4).hidden_ids) == 4


def test_rule_demand_without_rules_is_zero():
    assert set(mk.rule_demand({0: "a b", 1: "c d"}, []).values()) == {0.0}


# ---- evidence-targeted ----

def test_evidence_mask_hides_exactly_the_cited_sentences():
    p = _patient(6)
    m = mk.mask_evidence(p, frozenset({1, 4, 99}))        # 99 does not exist and is ignored
    assert m.hidden_ids == (1, 4) and m.visible.mask_type == MaskType.EVIDENCE
    assert mk.mask_evidence(p, frozenset()).hidden_ids == ()


def test_evidence_sets_use_definite_rows_only(tmp_path):
    raw = tmp_path
    (raw / "annotations").mkdir()
    pd.DataFrame({"annotation_id": [1, 2, 3], "patient_id": ["p"] * 3, "trial_id": ["T1", "T1", "T2"],
                  "expert_eligibility": ["included", "not enough information", "excluded"],
                  "expert_sentences": ["[0, 2]", "[5]", "[3]"]}
                 ).to_parquet(raw / "annotations" / "criterion_annotations.parquet")
    assert mk.evidence_sets(raw, "pair") == {("p", "T1"): frozenset({0, 2}), ("p", "T2"): frozenset({3})}
    assert mk.evidence_sets(raw, "patient") == {"p": frozenset({0, 2, 3})}
    assert set(mk.evidence_sets(raw, "rule")) == {("p", "T1", 1), ("p", "T2", 3)}


# ---- patients and facts ----

def test_facts_one_per_sentence():
    facts = pt.facts_of(_patient(3))
    assert [f.sentence_id for f in facts] == [0, 1, 2] and all(f.source == FactSource.NOTE for f in facts)
    assert facts[0].fact_id == "trec-20211#s0"


def test_test_split_is_sealed():
    with pytest.raises(PermissionError):
        pt.load_patients(Path("data/raw"), Split.TEST)


# ---- real data (skipped where it is not downloaded) ----

RAW = Path("data/raw")
needs_data = pytest.mark.skipif(not (RAW / "annotations").exists(), reason="run scripts/download_data.sh first")


def _punkt():
    try:
        pt.split_sentences("One. Two.", "trec_2021")
        return True
    except Exception:
        return False


@needs_data
@pytest.mark.skipif(not _punkt(), reason="nltk punkt_tab not available")
def test_sentence_ids_match_the_experts_numbering():
    # The annotation file stores each note numbered "0. ...", the same ids as expert_sentences.
    # Only note text is compared here, no labels.
    ann = pd.read_parquet(RAW / "annotations" / "criterion_annotations.parquet", columns=["patient_id", "note"])
    queries = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(RAW / "trialgpt/sigir/queries.jsonl")}
    checked = 0
    for pid, note in ann.drop_duplicates("patient_id").itertuples(index=False):
        if pid not in queries:
            continue
        gold = {int(s.split(". ", 1)[0]): s.split(". ", 1)[1] for s in note.split("\n")}
        assert pt.split_sentences(queries[pid], "sigir") == gold, pid
        checked += 1
    assert checked == 52


@needs_data
@pytest.mark.skipif(not _punkt(), reason="nltk punkt_tab not available")
def test_dev_masks_on_real_notes():
    dev = pt.load_patients(RAW, Split.DEV)
    assert len(dev) == 37
    p = next(iter(dev.values()))
    demand = mk.rule_demand(p.sentences, mk.candidate_rules(RAW, p.cohort, p.patient_id))
    for m in (mk.mask_random(p, 0.4), mk.mask_rule(p, demand, 0.4)):
        assert 0 < len(m.hidden_ids) < len(p.sentences)
        assert all(p.sentences[i] not in m.visible.text() for i in m.hidden_ids)
