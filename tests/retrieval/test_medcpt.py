from dataclasses import replace

import numpy as np
import pytest

import trialguide.retrieval.medcpt as medcpt
from trialguide.retrieval import retrieve_trials
from trialguide.schemas import Split, VisibleNote


def _trials():
    """Return small synthetic trial dictionaries with distinct shared-corpus IDs."""
    return [
        {"_id": "NCT2", "title": "Trial two", "text": "Second body", "metadata": {"phase": "2"}},
        {"_id": "NCT1", "title": "Trial one", "text": "First body", "metadata": {"phase": "1"}},
        {"_id": "NCT3", "title": "Trial three", "text": "Third body"},
    ]


def _config():
    """Use a tiny embedding dimension and synthetic revisions for model-free tests."""
    return medcpt.MedCPTConfig(
        query_revision="query-test-revision",
        article_revision="article-test-revision",
        embedding_dimension=3,
    )


def _fake_encoder_factory(monkeypatch, on_article=None, query_vector=None):
    """Replace HF model loading with deterministic fake batch encoders."""
    loaded = []

    def load(model_id, revision, max_length, device, fp16):
        loaded.append((model_id, revision, max_length, device, fp16))
        if model_id == medcpt.DEFAULT_ARTICLE_MODEL:
            def encode_articles(pairs):
                """Return stable vectors derived from synthetic article fields."""
                if on_article is not None:
                    on_article(pairs)
                return np.asarray([
                    [float(len(title)), float(len(body)), 1.0]
                    for title, body in pairs
                ], dtype=np.float32)

            return encode_articles, "cpu"

        def encode_queries(queries):
            """Return the configured synthetic query vector for each query."""
            return np.tile(np.asarray(query_vector or [1.0, 0.0, 0.0], dtype=np.float32),
                           (len(queries), 1))

        return encode_queries, "cpu"

    monkeypatch.setattr(medcpt, "_load_hf_encoder", load)
    return loaded


def test_visible_query_sorts_sentences_and_excludes_prefixes_and_patient_id():
    """Format only visible sentence contents in ID order without IDs or number prefixes."""
    note = VisibleNote(
        "patient-secret-42",
        Split.DEV,
        {5: "Later visible fact.", 1: "First visible fact."},
    )

    query = medcpt.build_medcpt_query(note)

    assert query == "First visible fact.\nLater visible fact."
    assert "1." not in query and "5." not in query
    assert "patient-secret-42" not in query
    assert "hidden phrase" not in query


def test_query_constructor_rejects_raw_text_and_handles_empty_visible_note():
    """Enforce the VisibleNote boundary and produce an empty query for no visible sentences."""
    with pytest.raises(TypeError, match="VisibleNote"):
        medcpt.build_medcpt_query("full note")
    assert medcpt.build_medcpt_query(VisibleNote("p", Split.DEV, {})) == ""


def test_article_encoding_uses_paired_fields_and_configured_batches(tmp_path, monkeypatch):
    """Encode `[title, text]` pairs in bounded batches and preserve source metadata."""
    observed = []
    _fake_encoder_factory(monkeypatch, observed.append)

    index = medcpt.build_corpus_index(
        _trials(), "sigir", tmp_path / "sigir", _config(), batch_size=2, device="cpu"
    )

    assert observed == [
        [["Trial two", "Second body"], ["Trial one", "First body"]],
        [["Trial three", "Third body"]],
    ]
    assert index.embeddings.shape == (3, 3)
    assert index.embeddings.dtype == np.float32
    assert index.trial_ids == ["NCT2", "NCT1", "NCT3"]
    assert index.metadata["trial_count"] == 3
    assert index.metadata["cohort"] == "sigir"
    assert index.metadata["tokenizer_configuration"]["article"]["sentence_pair"] == ["title", "text"]


@pytest.mark.parametrize("bad_trials, message", [
    ([{"title": "T", "text": "B"}], "_id"),
    ([{"_id": "", "title": "T", "text": "B"}], "_id"),
    ([{"_id": 7, "title": "T", "text": "B"}], "_id"),
    ([{"_id": "NCT1", "text": "B"}], "title"),
    ([{"_id": "NCT1", "title": "T"}], "text"),
    ([{"_id": "NCT1", "title": "T", "text": "B"},
      {"_id": "NCT1", "title": "T2", "text": "B2"}], "duplicate"),
])
def test_embedding_builder_rejects_duplicate_ids_and_invalid_records(
    tmp_path, monkeypatch, bad_trials, message
):
    """Reject malformed records before loading an encoder or writing a cache."""
    loaded = _fake_encoder_factory(monkeypatch)

    with pytest.raises((TypeError, ValueError), match=message):
        medcpt.build_corpus_index(bad_trials, "sigir", tmp_path / "bad", _config())
    assert loaded == []


@pytest.mark.parametrize("returned", [
    np.ones((1, 3), dtype=np.float32),
    np.asarray([[1.0, np.nan, 2.0], [1.0, 2.0, 3.0]], dtype=np.float32),
])
def test_article_encoder_shape_and_finite_values_are_validated(tmp_path, monkeypatch, returned):
    """Reject wrong batch shapes and non-finite embedding values."""
    monkeypatch.setattr(medcpt, "_load_hf_encoder", lambda *args: (lambda pairs: returned, "cpu"))

    with pytest.raises(ValueError, match="shape|non-finite"):
        medcpt.build_corpus_index(_trials()[:2], "sigir", tmp_path / "invalid", _config())


def test_load_reuses_cache_without_loading_models_and_maps_ids(tmp_path, monkeypatch):
    """Reload persisted vectors without model loading and select them by trial ID."""
    _fake_encoder_factory(monkeypatch)
    trials = _trials()
    cache = tmp_path / "index"
    built = medcpt.build_corpus_index(trials, "trec_2021", cache, _config(), device="cpu")
    monkeypatch.setattr(
        medcpt, "_load_hf_encoder",
        lambda *args: pytest.fail("valid cache reuse must not load an encoder"),
    )

    loaded = medcpt.get_or_build_corpus_index(trials, "trec_2021", cache, _config())

    assert np.array_equal(loaded.embeddings, built.embeddings)
    assert np.array_equal(loaded.vectors_for(["NCT1", "NCT2"]), built.embeddings[[1, 0]])


def test_cache_rejects_wrong_cohort_changed_corpus_revision_and_tokenizer(tmp_path, monkeypatch):
    """Reject caches whose cohort, records, model revision, or token limit differs."""
    _fake_encoder_factory(monkeypatch)
    trials = _trials()
    cache = tmp_path / "index"
    medcpt.build_corpus_index(trials, "sigir", cache, _config())

    with pytest.raises(ValueError, match="cohort"):
        medcpt.load_corpus_index(cache, "trec_2021", _config())
    changed = [dict(trial) for trial in trials]
    changed[0] = {**changed[0], "text": "changed article body"}
    with pytest.raises(ValueError, match="fingerprint"):
        medcpt.load_corpus_index(cache, "sigir", _config(), trials=changed)
    with pytest.raises(ValueError, match="article_revision"):
        medcpt.load_corpus_index(cache, "sigir", replace(_config(), article_revision="other-revision"))
    with pytest.raises(ValueError, match="tokenizer_configuration"):
        medcpt.load_corpus_index(cache, "sigir", replace(_config(), query_max_length=64))


def test_cache_rejects_corruption_and_incomplete_write(tmp_path, monkeypatch):
    """Detect altered vector bytes and a cache missing its metadata commit marker."""
    _fake_encoder_factory(monkeypatch)
    cache = tmp_path / "index"
    medcpt.build_corpus_index(_trials(), "sigir", cache, _config())
    embeddings = cache / "article_embeddings.npy"
    embeddings.write_bytes(embeddings.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        medcpt.load_corpus_index(cache, "sigir", _config())

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "article_embeddings.npy").write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete"):
        medcpt.load_corpus_index(partial, "sigir", _config())


def test_metadata_is_written_last_when_cache_save_is_interrupted(tmp_path, monkeypatch):
    """Leave no metadata that could make a partially written index appear valid."""
    _fake_encoder_factory(monkeypatch)
    cache = tmp_path / "interrupted"

    def fail_metadata_write(path, value):
        """Simulate interruption before the cache commit marker is written."""
        raise OSError("simulated interruption")

    monkeypatch.setattr(medcpt, "_atomic_write_json", fail_metadata_write)
    with pytest.raises(OSError, match="simulated interruption"):
        medcpt.build_corpus_index(_trials(), "sigir", cache, _config())
    assert (cache / "article_embeddings.npy").is_file()
    assert not (cache / "metadata.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        medcpt.load_corpus_index(cache, "sigir", _config())


def test_empty_corpus_and_nonpositive_batch_behavior(tmp_path, monkeypatch):
    """Persist an empty index without model loading and reject invalid batch sizes."""
    loaded = _fake_encoder_factory(monkeypatch)
    empty = medcpt.build_corpus_index([], "sigir", tmp_path / "empty", _config())
    assert empty.embeddings.shape == (0, 3)
    assert loaded == []
    with pytest.raises(ValueError, match="batch_size"):
        medcpt.build_corpus_index(_trials(), "sigir", tmp_path / "bad-batch", _config(), batch_size=0)


def test_index_scorer_uses_dot_products_negative_scores_and_id_alignment(tmp_path, monkeypatch):
    """Score negative and positive inner products while returning scores in request order."""
    loaded = []

    def load(model_id, revision, max_length, device, fp16):
        loaded.append(model_id)
        if model_id == medcpt.DEFAULT_ARTICLE_MODEL:
            def articles(pairs):
                """Return deterministic signed vectors based on article titles."""
                return np.asarray([
                    [-1.0, 0.0, 0.0] if title == "Trial two" else [2.0, 0.0, 0.0]
                    for title, _ in pairs
                ], dtype=np.float32)
            return articles, "cpu"
        return lambda queries: np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), "cpu"

    monkeypatch.setattr(medcpt, "_load_hf_encoder", load)
    trials = _trials()[:2]
    index = medcpt.build_corpus_index(trials, "sigir", tmp_path / "index", _config())
    scorer = medcpt.make_medcpt_scorer(index, _config(), device="cpu")

    scores = scorer.score_trials("visible query", [trials[1], trials[0]])

    assert scores == [2.0, -1.0]
    assert loaded == [medcpt.DEFAULT_ARTICLE_MODEL, medcpt.DEFAULT_QUERY_MODEL]


def test_query_encoder_is_loaded_once_and_article_encoder_is_not_loaded_for_cached_scoring(
    tmp_path, monkeypatch
):
    """Reuse one query encoder and score cached article vectors without loading the article model."""
    loaded = _fake_encoder_factory(monkeypatch, query_vector=[0.0, 1.0, 0.0])
    medcpt._QUERY_ENCODERS.clear()
    trials = _trials()
    index = medcpt.build_corpus_index(trials, "sigir", tmp_path / "index", _config())
    scorer = medcpt.make_medcpt_scorer(index, _config(), device="cpu")

    scorer.score_trials("first query", trials)
    scorer.score_trials("second query", trials)

    assert [item[0] for item in loaded].count(medcpt.DEFAULT_ARTICLE_MODEL) == 1
    assert [item[0] for item in loaded].count(medcpt.DEFAULT_QUERY_MODEL) == 1


def test_retrieve_trials_includes_dense_candidates_when_lexical_methods_are_empty(
    tmp_path, monkeypatch
):
    """Allow dense candidates to populate the union even when keyword and BM25 return nothing."""
    docs = [
        {"_id": "NCT9", "title": "Alpha trial", "text": "Article one"},
        {"_id": "NCT1", "title": "Beta trial", "text": "Article two"},
    ]
    loaded = []

    def load(model_id, revision, max_length, device, fp16):
        loaded.append(model_id)
        if model_id == medcpt.DEFAULT_ARTICLE_MODEL:
            def articles(pairs):
                """Return one negative and one positive synthetic article vector."""
                return np.asarray([[-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

            return articles, "cpu"

        def queries(query_batch):
            """Return a synthetic query vector for dense candidate union coverage."""
            return np.asarray([[1.0, 0.0, 0.0]])

        return queries, "cpu"

    monkeypatch.setattr(medcpt, "_load_hf_encoder", load)
    medcpt._QUERY_ENCODERS.clear()
    index = medcpt.build_corpus_index(docs, "sigir", tmp_path / "index", _config())
    scorer = medcpt.make_medcpt_scorer(index, _config(), device="cpu")
    note = VisibleNote("synthetic-dev", Split.DEV, {0: "qz"})

    results = retrieve_trials(note, docs, use_medcpt=True, medcpt_scorer=scorer)

    assert [result.trial_id for result in results] == ["NCT1", "NCT9"]
    assert results[-1].metadata["medcpt_score"] < 0
    assert results[-1].methods == ("medcpt",)


def test_empty_visible_note_does_not_load_or_invoke_medcpt():
    """Return no dense candidates for an empty VisibleNote without loading model weights."""
    note = VisibleNote("synthetic-dev", Split.DEV, {})
    assert retrieve_trials(note, _trials(), use_medcpt=True, medcpt_scorer=lambda q, d: pytest.fail()) == []


def test_get_or_build_rejects_nonempty_incomplete_cache_instead_of_overwriting(tmp_path, monkeypatch):
    """Refuse an existing partial directory rather than replacing its contents."""
    loaded = _fake_encoder_factory(monkeypatch)
    cache = tmp_path / "partial"
    cache.mkdir()
    marker = cache / "unrelated.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete"):
        medcpt.get_or_build_corpus_index(_trials(), "sigir", cache, _config())
    assert marker.read_text(encoding="utf-8") == "keep"
    assert loaded == []
