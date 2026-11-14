"""Aggregate-only DEV evaluation for T18 lexical and MedCPT retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

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
    load_corpus_index,
    make_medcpt_scorer,
)
from trialguide.schemas import Split, VisibleNote

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
KS = (10, 100, 1000)


def load_rule_manifest(path: Path, patients: dict) -> dict:
    """Load the DEV rule-mask manifest and validate every patient entry before masking."""
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read DEV rule-mask manifest {path}: {exc}") from exc
    if manifest.get("split") != "dev" or manifest.get("mask") != "rule" or manifest.get("fraction") != 0.4:
        raise ValueError("DEV manifest must specify split='dev', mask='rule', fraction=0.4")
    entries = manifest.get("patients")
    if not isinstance(entries, dict):
        raise ValueError("DEV manifest is missing its patients mapping")
    for patient_id, patient in patients.items():
        if getattr(patient.split, "value", patient.split) != "dev":
            raise ValueError(f"DEV manifest patient {patient_id!r} is not a DEV patient")
        entry = entries.get(patient_id)
        if not isinstance(entry, dict):
            raise ValueError(f"DEV manifest is missing patient {patient_id!r}")
        if entry.get("cohort") != patient.cohort:
            raise ValueError(f"DEV manifest cohort mismatch for patient {patient_id!r}")
        hidden_ids = entry.get("hidden_ids")
        if (not isinstance(hidden_ids, list)
                or any(type(i) is not int for i in hidden_ids)
                or len(set(hidden_ids)) != len(hidden_ids)
                or any(i not in patient.sentences for i in hidden_ids)):
            raise ValueError(f"DEV manifest has invalid hidden sentence IDs for patient {patient_id!r}")
        if entry.get("sentences") != len(patient.sentences) or entry.get("hidden") != len(hidden_ids):
            raise ValueError(f"DEV manifest sentence counts mismatch for patient {patient_id!r}")
    extra = set(entries) - set(patients)
    if extra:
        raise ValueError(f"DEV manifest contains patient absent from DEV data: {sorted(extra)[0]!r}")
    return manifest


def visible_from_manifest(patient, hidden_ids: list[int]) -> VisibleNote:
    """Apply exact hidden IDs while retaining IDs and the patient/split metadata."""
    from trialguide.schemas import MaskType
    hidden = set(hidden_ids)
    return VisibleNote(patient.patient_id, patient.split,
                       {i: text for i, text in patient.sentences.items() if i not in hidden},
                       MaskType.RULE, 0.4)


def retrieval_scores(ranking: list[RetrievalResult], eligible: set[str]) -> dict:
    """Compute the shared retrieval_check recall@K and P@10 definitions."""
    ids = [item.trial_id for item in ranking]
    return {**{f"recall@{k}": len(eligible & set(ids[:k])) / len(eligible) for k in KS},
            "P@10": len(eligible & set(ids[:10])) / 10}


def _mean_ci(values):
    from trialguide.eval.stats import paired_bootstrap
    arr = np.asarray(values, dtype=float)
    boot = paired_bootstrap({"x": arr}, lambda d: d["x"].mean(axis=-1))
    return {"estimate": float(arr.mean()), "ci95": [boot.low, boot.high]}


def aggregate_medcpt_comparison(rows: list[dict]) -> dict:
    """Return aggregate-only full/visible metrics and paired patient-bootstrap CIs."""
    from trialguide.eval.stats import paired_bootstrap
    if not rows:
        return {metric: None for metric in [f"recall@{k}" for k in KS] + ["P@10"]}
    out = {}
    for metric in [f"recall@{k}" for k in KS] + ["P@10"]:
        full = np.asarray([row["full"][metric] for row in rows], dtype=float)
        visible = np.asarray([row["visible"][metric] for row in rows], dtype=float)
        delta = paired_bootstrap({"f": full, "v": visible},
                                 lambda d: d["v"].mean(axis=-1) - d["f"].mean(axis=-1))
        out[metric] = {"full": _mean_ci(full), "visible": _mean_ci(visible),
                       "visible_minus_full": delta.estimate,
                       "visible_minus_full_ci95": [delta.low, delta.high]}
    return out


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
                        manifest: dict, k: int = 10) -> dict:
    """Mask each DEV note, run T18 retrieval systems, and return aggregate-only metrics."""
    results_by_system = {system: {} for system in SYSTEMS}
    comparison_rows = []
    for patient_id, patient in patients.items():
        visible_note = visible_from_manifest(patient, manifest["patients"][patient_id]["hidden_ids"])
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
        full_note = VisibleNote(patient.patient_id, patient.split, dict(patient.sentences))
        med_full = _medcpt_results(full_note, trials, scorers[patient.cohort])
        med_visible = _medcpt_results(visible_note, trials, scorers[patient.cohort])
        results_by_system["medcpt"][patient_id] = med_visible
        eligible = {trial_id for trial_id, grade in gold.get(patient_id, {}).items() if grade == 2}
        if eligible:
            comparison_rows.append({"full": retrieval_scores(med_full, eligible),
                                    "visible": retrieval_scores(med_visible, eligible)})
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

    summary = aggregate_retrieval_results(results_by_system, gold, k=k)
    return {"systems": summary, "medcpt_full_vs_visible": aggregate_medcpt_comparison(comparison_rows),
            "medcpt_patients_scored": len(comparison_rows)}


# Load DEV inputs and each of the three shared corpus indexes once per run.
def run_dev_evaluation(raw: Path, cache_dir: Path, *, device: str = "auto",
                       batch_size: int = 16, fp16: bool = False,
                       manifest_path: Path = Path("results/dev_run/manifest.json")) -> dict:
    """Load DEV inputs, validate the rule manifest, and reuse pre-existing indexes only."""
    patients = load_dev_patients(raw)
    gold = load_dev_gold_grades(raw)
    manifest = load_rule_manifest(manifest_path, patients)
    corpora = {}
    scorers = {}
    config = MedCPTConfig()
    for cohort in SUPPORTED_COHORTS:
        corpus = load_trial_corpus(raw, cohort)
        index = load_corpus_index(Path(cache_dir) / cohort, cohort, config, trials=corpus)
        corpora[cohort] = corpus
        scorers[cohort] = make_medcpt_scorer(index, config, device=device, fp16=fp16)

    evaluation = evaluate_loaded_dev(patients, gold, corpora, scorers, manifest)
    return {
        "split": Split.DEV.value,
        "mask": "rule-targeted 40%; exact hidden sentence IDs from the DEV run manifest",
        "query": "VisibleNote sentence text in sentence-ID order; no numeric prefixes; max 256 tokens",
        "article": "[trial['title'], trial['text']]; max 512 tokens",
        "medcpt_query_revision": config.query_revision,
        "medcpt_article_revision": config.article_revision,
        "rank_fusion": "existing equal-weight RRF: sum(1 / (20 + rank)), rank starts at 1",
        **evaluation,
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
