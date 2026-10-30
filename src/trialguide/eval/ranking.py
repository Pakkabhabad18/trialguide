"""Ranking metrics (T17): NDCG@10 and P@10 with TrialGPT's conventions, plus the gold grades.

Gold grade of a (patient, trial) pair: 0 not relevant, 1 excluded (TREC) / potential (SIGIR),
2 eligible, from the qrels. Only labelled pairs are ranked ("unlabelled pairs are not included",
as in the TrialGPT paper), and these formulas are the ones the T4 baseline evaluation uses:

    DCG@k  = sum_x grade_x / log2(x + 1)          x = rank, starting at 1
    NDCG@k = DCG@k / ideal DCG@k                  ideal = all labelled candidates, best first
    P@10   = sum of grades in the top 10 / (2 * 10)

A patient with no graded trial among the candidates has no defined NDCG and is dropped; that
depends only on the gold labels, so every system drops the same patients.
"""

import math
from pathlib import Path

import pandas as pd

from trialguide.data.audit import cohort_of
from trialguide.data.splits import load_split
from trialguide.schemas import Split

MAX_GRADE = 2


def load_gold_grades(raw: Path, split: Split, allow_test: bool = False) -> dict:
    """{patient_id: {trial_id: grade}} for the patients of one frozen split.

    Like the notes, the test labels stay sealed until T33 unless allow_test=True.
    """
    if split == Split.TEST and not allow_test:
        raise PermissionError("The test split is sealed until T33 (opened once, after the v1.0 freeze).")
    wanted = set(load_split()[split])
    gold = {p: {} for p in wanted}
    for cohort in ("sigir", "trec_2021", "trec_2022"):
        qrels = pd.read_csv(Path(raw) / "trialgpt" / cohort / "qrels" / "test.tsv", sep="\t")
        for pid, trial, grade in qrels.itertuples(index=False):
            if pid in wanted:
                gold[pid][trial] = int(grade)
    return gold


def dcg(gains, k: int = 10) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(gains, k: int = 10):
    """NDCG of a ranked list of gains, or None when no candidate has a positive grade."""
    ideal = dcg(sorted(gains, reverse=True), k)
    return dcg(gains, k) / ideal if ideal > 0 else None


def precision_at_k(gains, k: int = 10, max_grade: int = MAX_GRADE) -> float:
    """Graded precision: share of the best possible gain in the top k."""
    return sum(gains[:k]) / (max_grade * k)


def ranked_gains(scores: dict, grades: dict) -> list:
    """Gold grades of the system's trials, best score first. Ties are broken by trial id so a
    run is reproducible. A labelled trial without a score is an error, not a silent drop, because
    dropping it would flatter the system."""
    missing = [t for t in grades if t not in scores]
    if missing:
        raise ValueError(f"{len(missing)} labelled trial(s) have no score, e.g. {missing[0]}")
    return [grades[t] for t in sorted(grades, key=lambda t: (-scores[t], t))]


def per_patient_metrics(rankings: dict, gold: dict, k: int = 10) -> dict:
    """{patient_id: {"ndcg": x, "p10": y}} for patients with a defined NDCG.

    rankings: {patient_id: {trial_id: score}}. Only trials that have a gold grade are ranked.
    """
    out = {}
    for pid, scores in rankings.items():
        grades = {t: gold[pid][t] for t in scores if t in gold[pid]}
        gains = ranked_gains(scores, grades)
        n = ndcg_at_k(gains, k)
        if n is not None:
            out[pid] = {"ndcg": n, "p10": precision_at_k(gains, k)}
    return out


def same_candidates(rankings_by_system: dict) -> None:
    """All systems must rank the same trials per patient (proposal: 'same retrieved trials')."""
    names = list(rankings_by_system)
    ref = rankings_by_system[names[0]]
    for name in names[1:]:
        other = rankings_by_system[name]
        if set(other) != set(ref):
            raise ValueError(f"{name} ranks different patients than {names[0]}")
        for pid in ref:
            if set(other[pid]) != set(ref[pid]):
                raise ValueError(f"{name} ranks different trials than {names[0]} for {pid}")


__all__ = ["load_gold_grades", "dcg", "ndcg_at_k", "precision_at_k", "ranked_gains",
           "per_patient_metrics", "same_candidates", "cohort_of"]
