"""VisibleNote-only trial retrieval: lexical keywords, BM25, and optional MedCPT (T18).

This module never loads patient records, cached GPT-4 queries, labels, or hidden sentences.
Owner: Vidur.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from trialguide.schemas import VisibleNote

DEFAULT_MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
DEFAULT_MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
_TOKEN_RE = re.compile(r"[a-z][a-z0-9]+", re.I)
_STOPWORDS = {"about", "after", "also", "and", "are", "been", "but", "for", "from", "has", "have", "his", "her", "into", "not", "of", "on", "or", "patient", "the", "their", "this", "was", "with"}


@dataclass(frozen=True)
class RetrievalResult:
    """Uniform ranked trial output with original trial data and method scores."""

    trial_id: str
    score: float
    trial: dict
    methods: tuple[str, ...]
    metadata: dict = field(default_factory=dict)


def _note_text(note: VisibleNote) -> str:
    if not isinstance(note, VisibleNote):
        raise TypeError("retrieval accepts only a VisibleNote")
    return "\n".join(f"{sid}. {text}" for sid, text in sorted(note.sentences.items()))


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOPWORDS]


def _trial_id(trial: dict) -> str:
    for key in ("NCTID", "_id", "trial_id", "id"):
        value = trial.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("trial record has no recognized identifier (NCTID, _id, trial_id, id)")


def _flatten(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [part for v in value.values() for part in _flatten(v)]
    if isinstance(value, (list, tuple)):
        return [part for v in value for part in _flatten(v)]
    return []


def _trial_text(trial: dict) -> str:
    # Include corpus title, summary, and eligibility criteria without label fields.
    values = []
    for key in ("title", "brief_title", "text", "inclusion_criteria", "exclusion_criteria"):
        values.extend(_flatten(trial.get(key)))
    metadata = trial.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("brief_title", "diseases_list", "inclusion_criteria", "exclusion_criteria"):
            values.extend(_flatten(metadata.get(key)))
    return " ".join(values)


def _validate_trials(trials: Iterable[dict]) -> list[tuple[str, dict, str, list[str]]]:
    prepared = []
    seen = set()
    for trial in trials:
        if not isinstance(trial, dict):
            raise TypeError("each trial must be a dictionary")
        tid = _trial_id(trial)
        if tid in seen:
            raise ValueError(f"duplicate trial ID: {tid}")
        seen.add(tid)
        text = _trial_text(trial)
        prepared.append((tid, trial, text, _tokens(text)))
    return prepared


def keyword_retrieve(note: VisibleNote, trials: Iterable[dict], top_k: int = 10) -> list[RetrievalResult]:
    """Rank by visible-note keyword overlap; deterministic and dependency-free."""
    query = set(_tokens(_note_text(note)))
    prepared = _validate_trials(trials)
    if top_k <= 0 or not query or not prepared:
        return []
    scored = []
    for tid, trial, _, tokens in prepared:
        counts = set(tokens)
        matched = query & counts
        score = len(matched) / len(query)
        scored.append((tid, trial, score, sorted(matched)))
    scored.sort(key=lambda row: (-row[2], row[0]))
    return [RetrievalResult(tid, score, trial, ("keyword",),
                            {"keyword_score": score, "matched_keywords": matched})
            for tid, trial, score, matched in scored[:top_k] if score > 0]


def _bm25_scores(query_tokens: list[str], documents: list[list[str]]) -> tuple[list[float], str]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return [0.0] * len(documents), "unavailable"
    if not documents or not query_tokens:
        return [0.0] * len(documents), "rank_bm25"
    bm25 = BM25Okapi(documents)
    return [float(score) for score in bm25.get_scores(query_tokens)], "rank_bm25"


def bm25_retrieve(note: VisibleNote, trials: Iterable[dict], top_k: int = 10) -> list[RetrievalResult]:
    """BM25 retrieval with a dependency-free keyword fallback if rank-bm25 is unavailable."""
    prepared = _validate_trials(trials)
    query_tokens = _tokens(_note_text(note))
    if top_k <= 0 or not prepared or not query_tokens:
        return []
    scores, backend = _bm25_scores(query_tokens, [row[3] for row in prepared])
    if backend == "unavailable":
        fallback = keyword_retrieve(note, [row[1] for row in prepared], top_k)
        return [RetrievalResult(r.trial_id, r.score, r.trial, ("keyword_fallback",),
                                {**r.metadata, "bm25_backend": backend}) for r in fallback]
    ranked = sorted(zip(prepared, scores), key=lambda item: (-item[1], item[0][0]))
    return [RetrievalResult(tid, score, trial, ("bm25",),
                            {"bm25_score": score, "bm25_backend": backend})
            for (tid, trial, _, _), score in ranked[:top_k] if score > 0]


def _medcpt_scores(note_text: str, trial_texts: list[str], *, query_model: str,
                   article_model: str, device: str = "cpu", batch_size: int = 16) -> list[float]:
    """Lazy optional MedCPT encoder; weights load only when explicitly requested."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("MedCPT retrieval requires optional torch and transformers packages") from exc
    q_tokenizer = AutoTokenizer.from_pretrained(query_model)
    a_tokenizer = AutoTokenizer.from_pretrained(article_model)
    q_model = AutoModel.from_pretrained(query_model).to(device).eval()
    a_model = AutoModel.from_pretrained(article_model).to(device).eval()
    with torch.no_grad():
        query = q_tokenizer(note_text, truncation=True, max_length=512, return_tensors="pt").to(device)
        q_embedding = q_model(**query).last_hidden_state[:, 0, :]
        scores = []
        for start in range(0, len(trial_texts), batch_size):
            batch = a_tokenizer(trial_texts[start:start + batch_size], padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to(device)
            emb = a_model(**batch).last_hidden_state[:, 0, :]
            scores.extend((emb @ q_embedding.T).squeeze(-1).cpu().tolist())
    return [float(score) for score in scores]


def retrieve_trials(note: VisibleNote, trials: Iterable[dict], top_k: int = 10, *,
                    use_medcpt: bool = False,
                    medcpt_query_model: str = DEFAULT_MEDCPT_QUERY_MODEL,
                    medcpt_article_model: str = DEFAULT_MEDCPT_ARTICLE_MODEL,
                    medcpt_device: str = "cpu",
                    medcpt_scorer: Callable[[str, list[str]], list[float]] | None = None) -> list[RetrievalResult]:
    """Fuse keyword and BM25 ranks, optionally adding MedCPT scores.

    The scorer injection supports lightweight integration tests. The built-in MedCPT path
    is lazy and may download weights only after `use_medcpt=True` is explicitly requested.
    """
    if top_k <= 0:
        return []
    visible_text = _note_text(note)
    prepared = _validate_trials(trials)
    if not prepared or not _tokens(visible_text):
        return []
    keywords = keyword_retrieve(note, [row[1] for row in prepared], len(prepared))
    bm25 = bm25_retrieve(note, [row[1] for row in prepared], len(prepared))
    kw_rank = {r.trial_id: i + 1 for i, r in enumerate(keywords)}
    bm_rank = {r.trial_id: i + 1 for i, r in enumerate(bm25)}
    kw_scores = {r.trial_id: r.metadata["keyword_score"] for r in keywords}
    bm_scores = {r.trial_id: r.metadata.get("bm25_score", 0.0) for r in bm25}
    med_scores = {}
    med_status = "not_requested"
    if use_medcpt:
        scorer = medcpt_scorer
        if scorer is None:
            try:
                scores = _medcpt_scores(visible_text, [r[2] for r in prepared],
                                        query_model=medcpt_query_model,
                                        article_model=medcpt_article_model,
                                        device=medcpt_device)
            except ImportError:
                scores, med_status = [], "unavailable_optional_dependency"
            else:
                med_status = "available"
        else:
            scores = scorer(visible_text, [r[2] for r in prepared])
            med_status = "injected_scorer"
        if scores is not None and len(scores) > 0:
            if len(scores) != len(prepared):
                raise ValueError("MedCPT scorer must return one score per trial")
            med_scores = {row[0]: float(score) for row, score in zip(prepared, scores)}
    med_rank = {tid: i + 1 for i, (tid, _) in enumerate(
        sorted(med_scores.items(), key=lambda item: (-item[1], item[0])))}
    available = [(kw_rank, "keyword"), (bm_rank, "bm25")]
    if med_rank:
        available.append((med_rank, "medcpt"))
    ranked = []
    for tid, trial, _, _ in prepared:
        ranks = [(rank[tid], name) for rank, name in available if tid in rank]
        if not ranks:
            continue
        fused = sum(1 / rank for rank, _ in ranks)
        ranked.append((tid, trial, fused, ranks))
    ranked.sort(key=lambda row: (-row[2], row[0]))
    return [RetrievalResult(tid, score, trial, tuple(name for _, name in ranks), {
        "keyword_score": kw_scores.get(tid, 0.0), "bm25_score": bm_scores.get(tid, 0.0),
        "medcpt_score": med_scores.get(tid), "medcpt_status": med_status,
        "rank_fusion_score": score,
    }) for tid, trial, score, ranks in ranked[:top_k]]


__all__ = ["RetrievalResult", "keyword_retrieve", "bm25_retrieve", "retrieve_trials",
           "DEFAULT_MEDCPT_QUERY_MODEL", "DEFAULT_MEDCPT_ARTICLE_MODEL"]
