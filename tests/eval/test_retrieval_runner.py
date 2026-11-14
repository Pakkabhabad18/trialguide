"""Synthetic DEV-only checks for the T18 retrieval evaluation runner."""

from pathlib import Path

import pytest

from trialguide.data.patients import Patient
from trialguide.eval import retrieval_runner as runner
from trialguide.retrieval import RetrievalResult
from trialguide.schemas import Split, VisibleNote


def _result(trial_id, score):
    """Build a small retrieval result with no patient-specific data."""
    trial = {"_id": trial_id, "title": trial_id, "text": "synthetic trial"}
    return RetrievalResult(trial_id, score, trial, ("synthetic",), {})


def test_grade2_recall_uses_actual_top_k_and_skips_patients_without_grade2():
    """Measure recall over grade-2 qrels and return None for an ineligible denominator."""
    ranked = [_result(f"T{i}", 10 - i) for i in range(11)]

    assert runner.eligible_trial_recall_at_k(ranked, {"T0": 2, "T10": 2}) == 0.5
    assert runner.eligible_trial_recall_at_k(ranked, {"T0": 1, "T10": 0}) is None


def test_aggregate_metrics_use_ranking_api_and_macro_recall():
    """Aggregate per-patient ranking metrics and grade-2 recall without IDs in output."""
    patient_results = {
        "p1": [_result("A", 3.0), _result("C", 2.0)],
        "p2": [_result("D", 1.0)],
    }
    results_by_system = {system: patient_results for system in runner.SYSTEMS}
    gold = {
        "p1": {"A": 2, "B": 2, "C": 1},
        "p2": {"D": 1},
    }

    aggregate = runner.aggregate_retrieval_results(results_by_system, gold)

    expected_recall = 0.5
    for metrics in aggregate.values():
        assert metrics["eligible_trial_recall@10"] == pytest.approx(expected_recall)
        assert metrics["n_patients_with_grade2_qrels"] == 1
        assert metrics["graded_p@10"] == pytest.approx((5 / 20 + 1 / 20) / 2)
        assert "not all clinically eligible trials" in metrics["eligible_trial_recall_definition"]
    assert set(aggregate) == set(runner.SYSTEMS)


def test_dev_protocol_masks_visible_notes_and_routes_each_cohort_once(monkeypatch, tmp_path):
    """Use deterministic 40% masks and load/reuse one corpus and index per cohort."""
    cohorts = ("sigir", "trec_2021", "trec_2022")
    patients = {
        f"synthetic-{cohort}": Patient(
            f"synthetic-{cohort}",
            cohort,
            Split.DEV,
            {i: f"visible-fact-{i}" for i in range(10)},
        )
        for cohort in cohorts
    }
    gold = {patient_id: {f"T-{cohort}": 2} for cohort, patient_id in
            zip(cohorts, patients)}
    loaded_corpora = {}
    corpus_loads = []
    index_loads = []
    note_calls = []
    dense_queries = []
    routed_calls = []
    loader_calls = []

    def load_dev_patients(raw):
        """Record the isolated patient-loader call and return synthetic DEV records."""
        loader_calls.append("patients")
        return patients

    def load_dev_gold(raw):
        """Record the isolated grade-loader call and return synthetic DEV qrels."""
        loader_calls.append("gold")
        return gold

    monkeypatch.setattr(runner, "load_dev_patients", load_dev_patients)
    monkeypatch.setattr(runner, "load_dev_gold_grades", load_dev_gold)

    def load_corpus(raw, cohort):
        """Return one synthetic shared corpus per cohort and count its load."""
        corpus_loads.append(cohort)
        loaded_corpora[cohort] = [
            {"_id": f"T-{cohort}", "title": "Synthetic title", "text": "Synthetic body"}
        ]
        return loaded_corpora[cohort]

    def get_index(cache_dir, cohort, config, *, trials):
        """Stand in for a validated existing index without model or corpus data."""
        index_loads.append(cohort)
        return cohort

    class FakeScorer:
        """Capture visible query text while scoring the routed synthetic corpus."""

        def __init__(self, cohort):
            self.cohort = cohort

        def score_trials(self, query, trials):
            """Return one deterministic score per synthetic shared trial."""
            dense_queries.append((self.cohort, query))
            return [1.0] * len(trials)

    def make_scorer(index, config, **kwargs):
        """Bind a synthetic scorer to the cohort represented by its fake index."""
        return FakeScorer(index)

    def lexical(note, trials, top_k):
        """Capture VisibleNote and shared-corpus routing for lexical methods."""
        note_calls.append(note)
        routed_calls.append((note, trials))
        return [_result(trials[0]["_id"], 1.0)]

    def fused(note, trials, top_k, *, use_medcpt=False, medcpt_scorer=None):
        """Capture the existing RRF call shape and return its synthetic rank."""
        note_calls.append(note)
        routed_calls.append((note, trials))
        return [_result(trials[0]["_id"], 1.0)]

    def two_way_fused(note, trials, top_k, *, medcpt_scorer):
        """Capture the separate BM25-plus-MedCPT RRF path."""
        note_calls.append(note)
        routed_calls.append((note, trials))
        return [_result(trials[0]["_id"], 1.0)]

    monkeypatch.setattr(runner, "load_trial_corpus", load_corpus)
    monkeypatch.setattr(runner, "load_corpus_index", get_index)
    monkeypatch.setattr(runner, "make_medcpt_scorer", make_scorer)
    monkeypatch.setattr(runner, "keyword_retrieve", lexical)
    monkeypatch.setattr(runner, "bm25_retrieve", lexical)
    monkeypatch.setattr(runner, "retrieve_trials", fused)
    monkeypatch.setattr(runner, "bm25_medcpt_rrf_retrieve", two_way_fused)

    manifest = {"split": "dev", "mask": "rule", "fraction": .4, "patients": {
        pid: {"cohort": p.cohort, "sentences": 10, "hidden": 2, "hidden_ids": [1, 4]}
        for pid, p in patients.items()}}
    manifest_path = tmp_path / "manifest.json"
    import json
    manifest_path.write_text(json.dumps(manifest))
    summary = runner.run_dev_evaluation(Path("synthetic-raw"), tmp_path, device="cpu",
                                        manifest_path=manifest_path)

    assert loader_calls == ["patients", "gold"]
    assert corpus_loads == list(cohorts)
    assert index_loads == list(cohorts)
    assert all(note.split == Split.DEV for note in note_calls)
    expected_notes = {pid: runner.visible_from_manifest(patient, [1, 4])
                      for pid, patient in patients.items()}
    cohort_by_note = {
        note.patient_id: patients[patient_id].cohort
        for patient_id, note in expected_notes.items()
    }
    assert note_calls and all(note in expected_notes.values() for note in note_calls)
    for note, corpus in routed_calls:
        assert corpus is loaded_corpora[cohort_by_note[note.patient_id]]
    assert len(dense_queries) == 2 * len(cohorts)
    assert set(summary["systems"]) == set(runner.SYSTEMS)
    assert summary["systems"]["bm25_medcpt_rrf"]["label"] == "BM25 + MedCPT RRF"
    for cohort, query in dense_queries:
        patient_id = next(pid for pid in patients if patients[pid].cohort == cohort)
        expected = {
            runner.build_medcpt_query(expected_notes[patient_id]),
            runner.build_medcpt_query(VisibleNote(
                patient_id, Split.DEV, patients[patient_id].sentences
            )),
        }
        assert query in expected
        assert "synthetic-" not in query
    serialized_summary = str(summary)
    assert all(patient_id not in serialized_summary for patient_id in patients)
    assert "hidden_ids" not in serialized_summary


def test_manifest_validation_and_exact_hidden_ids(tmp_path):
    patient = Patient("p1", "sigir", Split.DEV, {0: "a", 1: "b", 3: "c"})
    manifest = {"split": "dev", "mask": "rule", "fraction": .4, "patients": {
        "p1": {"cohort": "sigir", "sentences": 3, "hidden": 1, "hidden_ids": [1]}}}
    path = tmp_path / "manifest.json"
    import json
    path.write_text(json.dumps(manifest))
    loaded = runner.load_rule_manifest(path, {"p1": patient})
    note = runner.visible_from_manifest(patient, loaded["patients"]["p1"]["hidden_ids"])
    assert note.sentences == {0: "a", 3: "c"}
    assert note.patient_id == "p1" and note.split == Split.DEV
    assert note.mask_fraction == .4
    for field, value in (("split", "test"), ("mask", "random"), ("fraction", .2)):
        broken = dict(manifest, **{field: value})
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match="DEV manifest"):
            runner.load_rule_manifest(path, {"p1": patient})


def test_medcpt_metric_definitions_and_paired_bootstrap_are_aggregate_only():
    ranking = [_result(f"T{i}", 100 - i) for i in range(1001)]
    eligible = {"T0", "T10", "T100", "T1000", "outside"}
    scores = runner.retrieval_scores(ranking, eligible)
    assert scores["recall@10"] == pytest.approx(1 / 5)
    assert scores["recall@100"] == pytest.approx(2 / 5)
    assert scores["recall@1000"] == pytest.approx(3 / 5)
    assert scores["P@10"] == pytest.approx(1 / 10)
    rows = [{"full": scores, "visible": dict(scores, **{"recall@10": 0.0, "P@10": 0.0})}
            for _ in range(4)]
    summary = runner.aggregate_medcpt_comparison(rows)
    assert summary["recall@10"]["visible_minus_full"] == pytest.approx(-.2)
    assert summary["recall@10"]["visible_minus_full_ci95"] == pytest.approx([-.2, -.2])
    serialized = str(summary)
    assert "patient_id" not in serialized and "hidden_ids" not in serialized
    assert "T0" not in serialized and "eligible" not in serialized
