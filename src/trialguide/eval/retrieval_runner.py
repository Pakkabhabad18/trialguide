"""Aggregate-only DEV evaluation for T18 lexical and MedCPT retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from trialguide.data.masking import mask_random
from trialguide.data.dev_only import load_dev_gold_grades, load_dev_patients
from trialguide.eval.ranking import per_patient_metrics
from trialguide.retrieval import (
    RetrievalResult,
    bm25_retrieve,
    bm25_medcpt_rrf_retrieve,
    keyword_retrieve,
    retrieve_trials,
)
from trialguide.retrieval.corpus import SUPPORTED_COHORTS, load_trial_corpus
from trialguide.retrieval.medcpt import (
    MedCPTConfig,
    build_medcpt_query,
    get_or_build_corpus_index,
    make_medcpt_scorer,
)
from trialguide.schemas import Split

SYSTEMS = (
    "keyword",
    "bm25",
    "medcpt",
    "keyword_bm25_rrf",
    "keyword_bm25_medcpt_rrf",
    "bm25_medcpt_rrf",
)
SYSTEM_LABELS = {
    "keyword": "Keyword",
    "bm25": "BM25",
    "medcpt": "MedCPT",
    "keyword_bm25_rrf": "Keyword + BM25 RRF",
    "keyword_bm25_medcpt_rrf": "Keyword + BM25 + MedCPT RRF",
    "bm25_medcpt_rrf": "BM25 + MedCPT RRF",
}


# Rank the full shared corpus with the already-built dense article index.
def _medcpt_results(note, trials: list[dict], scorer) -> list[RetrievalResult]:
    """Return deterministic MedCPT results using visible text and the reusable scorer."""
    query = build_medcpt_query(note)
    if not query.strip():
        return []
    scores = scorer.score_trials(query, trials)
    if len(scores) != len(trials):
        raise ValueError("MedCPT index scorer must return one score per trial")
    ranked = sorted(
        zip(trials, scores),
        key=lambda item: (-float(item[1]), str(item[0]["_id"])),
    )
    return [
        RetrievalResult(
            str(trial["_id"]),
            float(score),
            trial,
            ("medcpt",),
            {"medcpt_score": float(score)},
        )
        for trial, score in ranked
    ]


# Compute grade-2 recall over the actual retrieved top-k list.
def eligible_trial_recall_at_k(results: list[RetrievalResult], grades: dict,
                               k: int = 10) -> float | None:
    """Return recall for benchmark-labelled grade-2 trials, or None with no grade-2 qrel."""
    eligible_ids = {trial_id for trial_id, grade in grades.items() if grade == 2}
    if not eligible_ids:
        return None
    retrieved_ids = {result.trial_id for result in results[:k]}
    return len(eligible_ids & retrieved_ids) / len(eligible_ids)


# Aggregate retrieval rankings without emitting per-patient or qrel-level output.
def aggregate_retrieval_results(results_by_system: dict, gold: dict, k: int = 10) -> dict:
    """Summarize macro NDCG, graded P@10, and grade-2 qrel recall by system."""
    summary = {}
    for system in SYSTEMS:
        patient_results = results_by_system[system]
        labelled_rankings = {}
        recalls = []
        eligible_patient_count = 0
        for patient_id, grades in gold.items():
            results = patient_results.get(patient_id, [])
            scores = {result.trial_id: float(result.score) for result in results}
            # Keep every labelled trial in the metric ranking; unretrieved labels rank last.
            for trial_id in grades:
                scores.setdefault(trial_id, float("-inf"))
            labelled_rankings[patient_id] = scores

            recall = eligible_trial_recall_at_k(results, grades, k)
            if recall is not None:
                eligible_patient_count += 1
                recalls.append(recall)

        patient_metrics = per_patient_metrics(labelled_rankings, gold, k=k)
        metric_rows = list(patient_metrics.values())
        summary[system] = {
            "label": SYSTEM_LABELS[system],
            "n_patients_with_defined_ranking_metrics": len(metric_rows),
            "ndcg@10": mean(row["ndcg"] for row in metric_rows) if metric_rows else None,
            "graded_p@10": mean(row["p10"] for row in metric_rows) if metric_rows else None,
            "n_patients_with_grade2_qrels": eligible_patient_count,
            "eligible_trial_recall@10": mean(recalls) if recalls else None,
            "eligible_trial_recall_definition": (
                "macro recall over benchmark-labelled grade-2 trials; not all clinically eligible trials"
            ),
        }
    return summary


# Evaluate already-loaded DEV records against their cohort-specific shared corpora.
def evaluate_loaded_dev(patients: dict, gold: dict, corpora: dict, scorers: dict,
                        k: int = 10) -> dict:
    """Mask each DEV note, run the four T18 systems, and return aggregate-only metrics."""
    results_by_system = {system: {} for system in SYSTEMS}
    for patient_id, patient in patients.items():
        visible_note = mask_random(patient, 0.4, base_seed=0).visible
        if patient.cohort not in corpora or patient.cohort not in scorers:
            raise ValueError(f"missing shared corpus or MedCPT index for cohort {patient.cohort!r}")
        trials = corpora[patient.cohort]

        # Retrieval calls receive only the masked VisibleNote and the matching shared corpus.
        results_by_system["keyword"][patient_id] = keyword_retrieve(
            visible_note, trials, top_k=len(trials)
        )
        results_by_system["bm25"][patient_id] = bm25_retrieve(
            visible_note, trials, top_k=len(trials)
        )
        results_by_system["medcpt"][patient_id] = _medcpt_results(
            visible_note, trials, scorers[patient.cohort]
        )
        results_by_system["keyword_bm25_rrf"][patient_id] = retrieve_trials(
            visible_note, trials, top_k=len(trials)
        )
        # Preserve retrieve_trials' existing equal-weight RRF, including its keyword vote.
        results_by_system["keyword_bm25_medcpt_rrf"][patient_id] = retrieve_trials(
            visible_note,
            trials,
            top_k=len(trials),
            use_medcpt=True,
            medcpt_scorer=scorers[patient.cohort],
        )
        results_by_system["bm25_medcpt_rrf"][patient_id] = bm25_medcpt_rrf_retrieve(
            visible_note,
            trials,
            top_k=len(trials),
            medcpt_scorer=scorers[patient.cohort],
        )

    return aggregate_retrieval_results(results_by_system, gold, k=k)


# Load DEV inputs and each of the three shared corpus indexes once per run.
def run_dev_evaluation(raw: Path, cache_dir: Path, *, device: str = "auto",
                       batch_size: int = 16, fp16: bool = False) -> dict:
    """Load DEV notes/qrels, build or reuse one index per cohort, and aggregate metrics."""
    patients = load_dev_patients(raw)
    gold = load_dev_gold_grades(raw)
    corpora = {}
    scorers = {}
    config = MedCPTConfig()
    for cohort in SUPPORTED_COHORTS:
        corpus = load_trial_corpus(raw, cohort)
        index = get_or_build_corpus_index(
            corpus,
            cohort,
            Path(cache_dir) / cohort,
            config,
            batch_size=batch_size,
            device=device,
            fp16=fp16,
        )
        corpora[cohort] = corpus
        scorers[cohort] = make_medcpt_scorer(index, config, device=device, fp16=fp16)

    return {
        "split": Split.DEV.value,
        "mask": "mask_random(patient, 0.4, base_seed=0)",
        "query": "VisibleNote sentence text in sentence-ID order; no numeric prefixes; max 256 tokens",
        "article": "[trial['title'], trial['text']]; max 512 tokens",
        "medcpt_query_revision": config.query_revision,
        "medcpt_article_revision": config.article_revision,
        "rank_fusion": "existing equal-weight RRF: sum(1 / (20 + rank)), rank starts at 1",
        "systems": evaluate_loaded_dev(patients, gold, corpora, scorers),
    }


# Expose an aggregate-only CLI with an explicit DEV data and index-cache location.
def main(argv=None) -> int:
    """Run the approved DEV protocol and print only aggregate JSON results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/medcpt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fp16", action="store_true", help="use half precision on CUDA")
    args = parser.parse_args(argv)
    result = run_dev_evaluation(
        args.raw,
        args.cache_dir,
        device=args.device,
        batch_size=args.batch_size,
        fp16=args.fp16,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
