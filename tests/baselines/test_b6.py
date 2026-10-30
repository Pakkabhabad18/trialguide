"""B6 data preparation (T16): patient choice, trial cap, visible note, output layout."""

import json

import pytest

from trialguide.baselines.b6 import cap_trials, choose_patients, visible_text, write_condition

DEV = {**{f"sigir-{i}": "sigir" for i in range(2)},
       **{f"trec-2021{i}": "trec_2021" for i in range(21)},
       **{f"trec-2022{i}": "trec_2022" for i in range(14)}}


def test_choose_patients_is_proportional_and_reproducible():
    chosen = choose_patients(DEV, 5)
    assert len(chosen) == 5 and chosen == choose_patients(DEV, 5)
    cohorts = [DEV[p] for p in chosen]
    assert cohorts.count("trec_2021") == 3 and cohorts.count("trec_2022") == 2   # 5*21/37=2.8, 5*14/37=1.9


def test_cap_trials_keeps_order_and_other_fields():
    entry = {"patient_id": "p", "patient": "note", "0": [1, 2, 3], "2": [7]}
    assert cap_trials(entry, 2) == {"patient_id": "p", "patient": "note", "0": [1, 2], "2": [7]}
    assert entry["0"] == [1, 2, 3]                                                # input untouched


def test_visible_text_keeps_order_and_drops_the_added_sentence():
    sents = {2: "C.", 0: "A.", 5: "Consent."}
    assert visible_text(sents) == "A. C. Consent."
    assert visible_text(sents, drop="Consent.") == "A. C."


def test_write_condition_replaces_the_note_everywhere(tmp_path):
    rows = {"sigir": [{"patient_id": "p1", "patient": "FULL NOTE", "1": [{"NCTID": "NCT1"}]}]}
    write_condition(tmp_path, rows, {"p1": "visible part"})
    r = json.loads((tmp_path / "sigir" / "retrieved_trials.json").read_text())
    q = [json.loads(line) for line in (tmp_path / "sigir" / "queries.jsonl").read_text().splitlines()]
    assert r[0]["patient"] == "visible part" and r[0]["1"] == [{"NCTID": "NCT1"}]
    assert q == [{"_id": "p1", "text": "visible part"}]
    assert rows["sigir"][0]["patient"] == "FULL NOTE"


def test_summarise_scores_each_patient_per_condition(tmp_path):
    pytest.importorskip("trialguide.baselines.trialgpt.evaluate")   # T4 code, PR #1
    from trialguide.baselines.b6 import summarise
    rows = {"trec_2021": [{"patient_id": "p1", "patient": "x", "0": [{"NCTID": "A"}], "2": [{"NCTID": "B"}]}]}
    side = {"0": ["", [], "included"], "1": ["", [], "not enough information"]}
    for cond, order in (("full", {"B": 2.0, "A": 1.0}), ("rule40", {"A": 2.0, "B": 1.0})):
        write_condition(tmp_path / "data" / cond, rows, {"p1": "x"})
        res = tmp_path / "results" / cond
        res.mkdir(parents=True)
        (res / "ranking_trec_2021_Qwen_Qwen3-8B.json").write_text(json.dumps({"p1": order}))
        (res / "matching_results_trec_2021_Qwen_Qwen3-8B.json").write_text(json.dumps(
            {"p1": {"0": {"A": {"inclusion": side, "exclusion": side}}, "1": {}, "2": {}}}))
    s = summarise(tmp_path, "Qwen/Qwen3-8B")["conditions"]
    assert s["full"]["NDCG@10"] == 1.0                      # relevant trial ranked first
    assert s["rule40"]["NDCG@10"] < 1.0
    assert s["full"]["inclusion_not_enough_information_rate"] == 0.5
