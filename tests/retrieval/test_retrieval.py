import pytest

import trialguide.retrieval as retrieval
from trialguide.retrieval import (
    RetrievalResult,
    bm25_medcpt_rrf_retrieve,
    bm25_retrieve,
    keyword_retrieve,
    retrieve_trials,
)
from trialguide.schemas import MaskType, Split, VisibleNote


def note(sentences=None):
    return VisibleNote("dev-patient", Split.DEV,
                       {0: "advanced lung cancer EGFR mutation"} if sentences is None else sentences,
                       mask_type=MaskType.EVIDENCE)


def trials():
    return [
        {"_id": "NCT2", "title": "Lung cancer", "text": "EGFR targeted therapy",
         "metadata": {"inclusion_criteria": "EGFR mutation"}},
        {"_id": "NCT1", "title": "Diabetes trial", "text": "type 2 diabetes therapy",
         "metadata": {"exclusion_criteria": "renal failure"}},
        {"_id": "NCT3", "title": "Breast cancer", "text": "HER2 positive disease",
         "metadata": {"inclusion_criteria": "HER2"}},
    ]


def test_only_visible_note_is_accepted_and_used():
    try:
        retrieve_trials("full patient note", trials())
    except TypeError as exc:
        assert "VisibleNote" in str(exc)
    else:
        raise AssertionError("raw note text should not be accepted")
    results = retrieve_trials(note({3: "advanced lung cancer"}), trials())
    assert results and results[0].trial_id == "NCT2"
    assert "hidden" not in str(results)


def test_keyword_retrieval_returns_uniform_results_and_trials():
    results = keyword_retrieve(note(), trials(), top_k=2)
    assert results[0].trial_id == "NCT2"
    assert results[0].trial["_id"] == results[0].trial_id
    assert results[0].methods == ("keyword",)
    assert "keyword_score" in results[0].metadata


def test_bm25_ranks_relevant_trial_first_and_has_consistent_schema():
    results = bm25_retrieve(note(), trials(), top_k=2)
    assert results[0].trial_id == "NCT2"
    assert results[0].methods[0] in {"bm25", "keyword_fallback"}
    assert isinstance(results[0].score, float)
    assert isinstance(results[0].metadata, dict)


def test_optional_medcpt_injected_scorer_and_configuration():
    captured = {}

    def scorer(query, documents):
        captured["query"] = query
        captured["documents"] = documents
        return [0.1, 0.9, 0.2]

    results = retrieve_trials(note(), trials(), use_medcpt=True,
                              medcpt_query_model="query-model", medcpt_article_model="article-model",
                              medcpt_scorer=scorer)
    assert len(captured["documents"]) == 3
    assert "advanced lung cancer" in captured["query"]
    med_result = next(r for r in results if r.trial_id == "NCT1")
    assert med_result.metadata["medcpt_score"] == 0.9
    assert "medcpt" in med_result.methods


def test_missing_optional_medcpt_dependency_falls_back(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ImportError("optional dependencies absent")

    monkeypatch.setattr("trialguide.retrieval._medcpt_scores", unavailable)
    results = retrieve_trials(note(), trials(), use_medcpt=True)
    assert results
    assert all(r.metadata["medcpt_status"] == "unavailable_optional_dependency" for r in results)


def test_empty_notes_and_trials_return_empty_lists():
    empty_note = note({})
    assert keyword_retrieve(empty_note, trials()) == []
    assert bm25_retrieve(empty_note, trials()) == []
    assert retrieve_trials(empty_note, trials()) == []
    assert keyword_retrieve(note(), []) == []
    assert bm25_retrieve(note(), []) == []
    assert retrieve_trials(note(), []) == []
    unrelated = note({0: "xylophone quantum avocado"})
    assert retrieve_trials(unrelated, trials()) == []


def test_duplicate_trial_ids_fail_closed():
    duplicate = [{"_id": "NCT1", "title": "a"}, {"_id": "NCT1", "title": "b"}]
    try:
        retrieve_trials(note(), duplicate)
    except ValueError as exc:
        assert "duplicate trial ID" in str(exc)
    else:
        raise AssertionError("duplicate trial IDs should fail")


def test_zero_score_lexical_trials_are_omitted_from_results_and_fusion(monkeypatch):
    """Keep lexical candidate lists limited to trials with positive evidence."""
    records = trials()

    def scores(query_tokens, documents):
        """Give one positive BM25 score and deterministic zero scores for the rest."""
        return [1.0, 0.0, 0.0], "rank_bm25"

    monkeypatch.setattr(retrieval, "_bm25_scores", scores)
    query = note({0: "EGFR mutation"})

    assert [r.trial_id for r in keyword_retrieve(query, records, top_k=3)] == ["NCT2"]
    assert [r.trial_id for r in bm25_retrieve(query, records, top_k=3)] == ["NCT2"]
    assert [r.trial_id for r in retrieve_trials(query, records, top_k=3)] == ["NCT2"]


def test_keyword_and_bm25_ties_are_broken_by_trial_id(monkeypatch):
    """Order equal lexical scores reproducibly by trial ID."""
    records = [
        {"_id": "NCT9", "title": "Lung cancer", "text": "Study body"},
        {"_id": "NCT1", "title": "Lung cancer", "text": "Study body"},
    ]
    monkeypatch.setattr(retrieval, "_bm25_scores", lambda query, docs: ([1.0, 1.0], "rank_bm25"))
    query = note({0: "lung cancer"})

    assert [r.trial_id for r in keyword_retrieve(query, records, top_k=2)] == ["NCT1", "NCT9"]
    assert [r.trial_id for r in bm25_retrieve(query, records, top_k=2)] == ["NCT1", "NCT9"]


def test_rank_fusion_uses_candidate_union_offset_and_deterministic_ties(monkeypatch):
    """Fuse both component candidate lists with offset RRF and stable ID tie-breaking."""
    records = [
        {"_id": "A", "title": "Trial A", "text": "Body"},
        {"_id": "B", "title": "Trial B", "text": "Body"},
        {"_id": "C", "title": "Trial C", "text": "Body"},
    ]
    by_id = {record["_id"]: record for record in records}

    def component_results(order, method, score_key):
        """Build controlled ranked outputs to isolate the fusion calculation."""
        return [
            RetrievalResult(tid, float(rank), by_id[tid], (method,), {score_key: float(rank)})
            for rank, tid in enumerate(order, start=1)
        ]

    monkeypatch.setattr(
        retrieval, "keyword_retrieve",
        lambda note, corpus, top_k: component_results(["A", "B"], "keyword", "keyword_score"),
    )
    monkeypatch.setattr(
        retrieval, "bm25_retrieve",
        lambda note, corpus, top_k: component_results(["B", "A", "C"], "bm25", "bm25_score"),
    )

    results = retrieve_trials(note({0: "cancer"}), records, top_k=3)

    assert [result.trial_id for result in results] == ["A", "B", "C"]
    assert results[0].metadata["rank_fusion_score"] == pytest.approx(1 / 21 + 1 / 22)
    assert results[1].metadata["rank_fusion_score"] == pytest.approx(1 / 21 + 1 / 22)
    assert results[2].metadata["rank_fusion_score"] == pytest.approx(1 / 23)
    assert results[2].methods == ("bm25",)


def test_unavailable_bm25_keyword_fallback_contributes_only_one_vote(monkeypatch):
    """Avoid double-counting keyword results when BM25 falls back to keyword search."""
    records = [{"_id": "NCT1", "title": "Lung cancer", "text": "EGFR study"}]
    monkeypatch.setattr(retrieval, "_bm25_scores", lambda query, docs: ([0.0], "unavailable"))

    result = retrieve_trials(note({0: "lung cancer EGFR"}), records)[0]

    assert result.methods == ("keyword",)
    assert result.score == pytest.approx(1 / 21)
    assert result.metadata["rank_fusion_score"] == pytest.approx(1 / 21)


def test_bm25_medcpt_rrf_uses_union_and_rank_scores_including_negative_dense(monkeypatch):
    """Fuse the union and retain negative MedCPT results as ranked candidates."""
    records = [
        {"_id": "A", "title": "A trial", "text": "A body"},
        {"_id": "B", "title": "B trial", "text": "B body"},
        {"_id": "C", "title": "C trial", "text": "C body"},
    ]
    monkeypatch.setattr(
        retrieval, "_bm25_scores", lambda query, docs: ([2.0, 1.0, 0.0], "rank_bm25")
    )

    results = bm25_medcpt_rrf_retrieve(
        note({0: "oncology mutation"}),
        records,
        top_k=3,
        medcpt_scorer=lambda query, docs: [-3.0, -2.0, -1.0],
    )

    assert [result.trial_id for result in results] == ["A", "B", "C"]
    assert results[0].score == pytest.approx(1 / 21 + 1 / 23)
    assert results[1].score == pytest.approx(1 / 22 + 1 / 22)
    assert results[2].score == pytest.approx(1 / 21)
    assert results[2].methods == ("medcpt",)
    assert results[2].metadata["medcpt_score"] == -1.0


def test_bm25_medcpt_rrf_ties_use_trial_id(monkeypatch):
    """Break equal two-method fusion scores by trial ID, independent of input order."""
    records = [
        {"_id": "B", "title": "B trial", "text": "B body"},
        {"_id": "A", "title": "A trial", "text": "A body"},
    ]
    monkeypatch.setattr(
        retrieval, "_bm25_scores", lambda query, docs: ([2.0, 1.0], "rank_bm25")
    )

    results = bm25_medcpt_rrf_retrieve(
        note({0: "oncology mutation"}),
        records,
        medcpt_scorer=lambda query, docs: [0.0, 1.0],
    )

    assert [result.trial_id for result in results] == ["A", "B"]
    assert results[0].score == pytest.approx(1 / 21 + 1 / 22)
    assert results[1].score == pytest.approx(1 / 21 + 1 / 22)


def test_bm25_medcpt_rrf_handles_empty_component_rankings(monkeypatch):
    """Return dense-only, BM25-only, or empty results when either rank list is empty."""
    records = [
        {"_id": "A", "title": "A trial", "text": "A body"},
        {"_id": "B", "title": "B trial", "text": "B body"},
    ]
    query = note({0: "oncology mutation"})
    monkeypatch.setattr(
        retrieval, "_bm25_scores", lambda q, docs: ([0.0, 0.0], "rank_bm25")
    )
    dense_only = bm25_medcpt_rrf_retrieve(
        query, records, medcpt_scorer=lambda q, docs: [-2.0, -1.0]
    )
    assert [result.trial_id for result in dense_only] == ["B", "A"]
    assert all(result.methods == ("medcpt",) for result in dense_only)

    monkeypatch.setattr(
        retrieval, "_bm25_scores", lambda q, docs: ([2.0, 0.0], "rank_bm25")
    )

    def unavailable_query_encoder(query_text, documents):
        """Simulate an unavailable optional dense dependency without model loading."""
        raise ImportError("synthetic missing MedCPT dependency")

    bm25_only = bm25_medcpt_rrf_retrieve(query, records, medcpt_scorer=unavailable_query_encoder)
    assert [result.trial_id for result in bm25_only] == ["A"]
    assert bm25_only[0].methods == ("bm25",)

    monkeypatch.setattr(
        retrieval, "_bm25_scores", lambda q, docs: ([0.0, 0.0], "rank_bm25")
    )
    assert bm25_medcpt_rrf_retrieve(query, records, medcpt_scorer=unavailable_query_encoder) == []


def test_bm25_medcpt_rrf_fails_clearly_on_keyword_fallback(monkeypatch):
    """Keep BM25 fallback visible and refuse to present it as an independent BM25 vote."""
    records = [{"_id": "A", "title": "Lung cancer", "text": "EGFR study"}]
    query = note({0: "lung cancer EGFR"})
    monkeypatch.setattr(retrieval, "_bm25_scores", lambda q, docs: ([0.0], "unavailable"))

    fallback = bm25_retrieve(query, records)
    assert fallback[0].methods == ("keyword_fallback",)
    assert fallback[0].metadata["bm25_backend"] == "unavailable"
    with pytest.raises(RuntimeError, match="BM25 retrieval is using keyword fallback"):
        bm25_medcpt_rrf_retrieve(query, records, medcpt_scorer=lambda q, docs: [1.0])


def test_existing_three_way_rrf_retains_its_exact_component_votes(monkeypatch):
    """Preserve the prior keyword-plus-BM25-plus-MedCPT candidate and vote behavior."""
    records = [
        {"_id": "A", "title": "A trial", "text": "A body"},
        {"_id": "B", "title": "B trial", "text": "B body"},
        {"_id": "C", "title": "C trial", "text": "C body"},
    ]
    by_id = {record["_id"]: record for record in records}

    def make_results(order, method, score_key):
        """Construct fixed lexical ranks for the three-way regression check."""
        return [
            RetrievalResult(tid, float(rank), by_id[tid], (method,), {score_key: float(rank)})
            for rank, tid in enumerate(order, start=1)
        ]

    monkeypatch.setattr(
        retrieval,
        "keyword_retrieve",
        lambda n, ts, k: make_results(["A", "B"], "keyword", "keyword_score"),
    )
    monkeypatch.setattr(
        retrieval,
        "bm25_retrieve",
        lambda n, ts, k: make_results(["B", "C"], "bm25", "bm25_score"),
    )

    results = retrieve_trials(
        note({0: "oncology mutation"}),
        records,
        top_k=3,
        use_medcpt=True,
        medcpt_scorer=lambda query, docs: [2.0, 1.0, 3.0],
    )

    assert [result.trial_id for result in results] == ["B", "A", "C"]
    assert results[0].score == pytest.approx(1 / 22 + 1 / 21 + 1 / 23)
    assert results[0].methods == ("keyword", "bm25", "medcpt")
    assert results[1].score == pytest.approx(1 / 21 + 1 / 22)
    assert results[1].methods == ("keyword", "medcpt")
    assert results[2].score == pytest.approx(1 / 22 + 1 / 21)
    assert results[2].methods == ("bm25", "medcpt")
