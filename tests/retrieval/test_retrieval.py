from trialguide.retrieval import bm25_retrieve, keyword_retrieve, retrieve_trials
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
