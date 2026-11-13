"""Tests for the RAG check (T18): the cached corpus index must rank exactly like the retriever."""

import pytest

from trialguide.eval import retrieval_check as rc
from trialguide.retrieval import bm25_retrieve, keyword_retrieve, retrieve_trials
from trialguide.schemas import Split, VisibleNote

pytest.importorskip("rank_bm25")

TRIALS = [
    {"_id": "NCT1", "title": "Metformin in type 2 diabetes", "text": "Adults with type 2 diabetes and obesity."},
    {"_id": "NCT2", "title": "Asthma inhaler study", "text": "Children with asthma."},
    {"_id": "NCT3", "title": "Diabetes and kidney disease", "text": "Chronic kidney disease in diabetes patients."},
    {"_id": "NCT4", "title": "Breast cancer chemotherapy", "text": "Women with breast cancer after surgery."},
    {"_id": "NCT5", "title": "Obesity diet trial", "text": "Weight loss in obesity, no diabetes."},
]
NOTE = VisibleNote("p1", Split.DEV, {0: "A 60-year-old man with type 2 diabetes.", 1: "He has obesity and kidney disease."})


def test_index_ranks_like_the_retriever():
    index = rc.CorpusIndex(TRIALS)
    assert index.keyword(NOTE) == [r.trial_id for r in keyword_retrieve(NOTE, TRIALS, len(TRIALS))]
    assert index.bm25_rank(NOTE) == [r.trial_id for r in bm25_retrieve(NOTE, TRIALS, len(TRIALS))]
    assert index.fused(NOTE) == [r.trial_id for r in retrieve_trials(NOTE, TRIALS, len(TRIALS))]


def test_scores():
    s = rc.scores(["NCT1", "NCT3", "NCT5"], {"NCT3", "NCT9"})
    assert s["recall@10"] == 0.5 and s["P@10"] == pytest.approx(0.1)


def test_note_of_drops_hidden_sentences():
    class P:
        patient_id, split, sentences = "p", Split.DEV, {0: "a.", 1: "b.", 2: "c."}
    assert rc.note_of(P(), [1]).sentences == {0: "a.", 2: "c."}
