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
RRF_OFFSET = 20
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


# Rank BM25 candidates once and return backend status for fallback-aware callers.
def _bm25_retrieve_prepared(note: VisibleNote, prepared: list[tuple], query_tokens: list[str],
                            top_k: int) -> tuple[list[RetrievalResult], str | None]:
    """Return ranked BM25 results and backend status for validated records."""
    if top_k <= 0 or not prepared or not query_tokens:
        return [], None
    scores, backend = _bm25_scores(query_tokens, [row[3] for row in prepared])
    if backend == "unavailable":
        fallback = keyword_retrieve(note, [row[1] for row in prepared], top_k)
        return ([RetrievalResult(r.trial_id, r.score, r.trial, ("keyword_fallback",),
                                 {**r.metadata, "bm25_backend": backend}) for r in fallback],
                backend)
    ranked = sorted(zip(prepared, scores), key=lambda item: (-item[1], item[0][0]))
    return ([RetrievalResult(tid, score, trial, ("bm25",),
                             {"bm25_score": score, "bm25_backend": backend})
             for (tid, trial, _, _), score in ranked[:top_k] if score > 0], backend)


def bm25_retrieve(note: VisibleNote, trials: Iterable[dict], top_k: int = 10) -> list[RetrievalResult]:
    """BM25 retrieval with a dependency-free keyword fallback if rank-bm25 is unavailable."""
    prepared = _validate_trials(trials)
    query_tokens = _tokens(_note_text(note))
    results, _ = _bm25_retrieve_prepared(note, prepared, query_tokens, top_k)
    return results


def _medcpt_scores(note_text: str, trial_records: list[dict], *, query_model: str,
                   article_model: str, device: str = "cpu", batch_size: int = 16) -> list[float]:
    """Use the reusable MedCPT subsystem for the default no-index scorer path."""
    from trialguide.retrieval.medcpt import (
        DEFAULT_ARTICLE_MODEL,
        DEFAULT_QUERY_MODEL,
        MedCPTConfig,
        score_medcpt_trials,
    )

    if query_model != DEFAULT_QUERY_MODEL or article_model != DEFAULT_ARTICLE_MODEL:
        raise ValueError(
            "custom MedCPT model IDs require an explicitly revision-pinned MedCPTConfig and index scorer"
        )
    return score_medcpt_trials(
        note_text,
        trial_records,
        MedCPTConfig(),
        batch_size=batch_size,
        device=device,
    )


# Fuse method rank maps over the same candidate union and RRF offset.
def _rrf_fuse(prepared: list[tuple], method_ranks: list[tuple[dict, str]],
              top_k: int) -> list[tuple]:
    """Apply equal-weight reciprocal-rank fusion and deterministic trial-ID tie-breaking."""
    ranked = []
    for tid, trial, _, _ in prepared:
        ranks = [(rank[tid], name) for rank, name in method_ranks if tid in rank]
        if not ranks:
            continue
        fused = sum(1 / (RRF_OFFSET + rank) for rank, _ in ranks)
        ranked.append((tid, trial, fused, ranks))
    ranked.sort(key=lambda row: (-row[2], row[0]))
    return ranked[:top_k]


# Share BM25 and dense ranks without changing retrieve_trials' existing three-method fusion.
def bm25_medcpt_rrf_retrieve(
    note: VisibleNote,
    trials: Iterable[dict],
    top_k: int = 10,
    *,
    medcpt_query_model: str = DEFAULT_MEDCPT_QUERY_MODEL,
    medcpt_article_model: str = DEFAULT_MEDCPT_ARTICLE_MODEL,
    medcpt_device: str = "cpu",
    medcpt_scorer: Callable[[str, list[str]], list[float]] | None = None,
) -> list[RetrievalResult]:
    """Fuse only BM25 and MedCPT ranks; reject keyword fallback as a false BM25 vote."""
    if top_k <= 0:
        return []
    visible_text = _note_text(note)
    prepared = _validate_trials(trials)
    if not prepared:
        return []
    lexical_query_tokens = _tokens(visible_text)
    from trialguide.retrieval.medcpt import build_medcpt_query

    medcpt_query = build_medcpt_query(note)
    if not lexical_query_tokens and not medcpt_query.strip():
        return []

    bm25, bm25_backend = _bm25_retrieve_prepared(
        note, prepared, lexical_query_tokens, len(prepared)
    )
    if bm25_backend == "unavailable":
        raise RuntimeError(
            "BM25 + MedCPT RRF requires rank-bm25; BM25 retrieval is using keyword fallback"
        )
    bm_rank = {result.trial_id: rank for rank, result in enumerate(bm25, start=1)}
    bm_scores = {result.trial_id: result.metadata.get("bm25_score", 0.0) for result in bm25}

    med_status = "available"
    try:
        if medcpt_scorer is None:
            scores = _medcpt_scores(
                medcpt_query,
                [row[1] for row in prepared],
                query_model=medcpt_query_model,
                article_model=medcpt_article_model,
                device=medcpt_device,
            )
        elif hasattr(medcpt_scorer, "score_trials"):
            scores = medcpt_scorer.score_trials(medcpt_query, [row[1] for row in prepared])
            med_status = "indexed_scorer"
        else:
            scores = medcpt_scorer(medcpt_query, [row[2] for row in prepared])
            med_status = "injected_scorer"
    except ImportError:
        scores, med_status = [], "unavailable_optional_dependency"
    if scores is not None and len(scores) > 0:
        if len(scores) != len(prepared):
            raise ValueError("MedCPT scorer must return one score per trial")
        med_scores = {row[0]: float(score) for row, score in zip(prepared, scores)}
    else:
        med_scores = {}
    med_rank = {
        tid: rank
        for rank, (tid, _) in enumerate(
            sorted(med_scores.items(), key=lambda item: (-item[1], item[0])),
            start=1,
        )
    }
    method_ranks = [(bm_rank, "bm25")]
    if med_rank:
        method_ranks.append((med_rank, "medcpt"))
    fused = _rrf_fuse(prepared, method_ranks, top_k)
    return [
        RetrievalResult(
            tid,
            score,
            trial,
            tuple(name for _, name in ranks),
            {
                "keyword_score": 0.0,
                "bm25_score": bm_scores.get(tid, 0.0),
                "medcpt_score": med_scores.get(tid),
                "medcpt_status": med_status,
                "bm25_backend": bm25_backend,
                "rank_fusion_score": score,
            },
        )
        for tid, trial, score, ranks in fused
    ]


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
    if not prepared:
        return []
    lexical_query_tokens = _tokens(visible_text)
    medcpt_query = ""
    if use_medcpt:
        from trialguide.retrieval.medcpt import build_medcpt_query

        medcpt_query = build_medcpt_query(note)
    if not lexical_query_tokens and not medcpt_query.strip():
        return []
    keywords = keyword_retrieve(note, [row[1] for row in prepared], len(prepared))
    bm25 = bm25_retrieve(note, [row[1] for row in prepared], len(prepared))
    kw_rank = {r.trial_id: i + 1 for i, r in enumerate(keywords)}
    bm_rank = {r.trial_id: i + 1 for i, r in enumerate(bm25)}
    kw_scores = {r.trial_id: r.metadata["keyword_score"] for r in keywords}
    bm_scores = {r.trial_id: r.metadata.get("bm25_score", 0.0) for r in bm25}
    bm25_is_keyword_fallback = any(
        r.metadata.get("bm25_backend") == "unavailable" for r in bm25
    )
    med_scores = {}
    med_status = "not_requested"
    if use_medcpt:
        scorer = medcpt_scorer
        try:
            if scorer is None:
                scores = _medcpt_scores(medcpt_query, [r[1] for r in prepared],
                                        query_model=medcpt_query_model,
                                        article_model=medcpt_article_model,
                                        device=medcpt_device)
                med_status = "available"
            elif hasattr(scorer, "score_trials"):
                scores = scorer.score_trials(medcpt_query, [r[1] for r in prepared])
                med_status = "indexed_scorer"
            else:
                # Keep the original injected callable contract for lightweight test scorers.
                scores = scorer(medcpt_query, [r[2] for r in prepared])
                med_status = "injected_scorer"
        except ImportError:
            scores, med_status = [], "unavailable_optional_dependency"
        if scores is not None and len(scores) > 0:
            if len(scores) != len(prepared):
                raise ValueError("MedCPT scorer must return one score per trial")
            med_scores = {row[0]: float(score) for row, score in zip(prepared, scores)}
    med_rank = {tid: i + 1 for i, (tid, _) in enumerate(
        sorted(med_scores.items(), key=lambda item: (-item[1], item[0])))}
    available = [(kw_rank, "keyword"), (bm_rank, "bm25")]
    if bm25_is_keyword_fallback:
        available.remove((bm_rank, "bm25"))
    if med_rank:
        available.append((med_rank, "medcpt"))
    ranked = _rrf_fuse(prepared, available, top_k)
    return [RetrievalResult(tid, score, trial, tuple(name for _, name in ranks), {
        "keyword_score": kw_scores.get(tid, 0.0), "bm25_score": bm_scores.get(tid, 0.0),
        "medcpt_score": med_scores.get(tid), "medcpt_status": med_status,
        "rank_fusion_score": score,
    }) for tid, trial, score, ranks in ranked]


__all__ = ["RetrievalResult", "keyword_retrieve", "bm25_retrieve", "retrieve_trials",
           "bm25_medcpt_rrf_retrieve",
           "DEFAULT_MEDCPT_QUERY_MODEL", "DEFAULT_MEDCPT_ARTICLE_MODEL"]
