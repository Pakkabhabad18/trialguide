"""RAG check (T18): how well trial retrieval works from the visible (masked) note.

    python -m trialguide.eval.retrieval_check --manifest results/dev_run/manifest.json
    python -m trialguide.eval.retrieval_check --manifest results/test_run/manifest.json --allow-test

For each patient of the manifest, the trial retriever (retrieval/__init__.py, Vidur) searches the
WHOLE trial corpus of the patient's cohort (SIGIR 3,620 trials; TREC 2021 26,148; TREC 2022 26,580)
twice: once with the full note and once with the visible note, i.e. the same rule-targeted 40% mask
every other run uses (the manifest's hidden sentence ids). We score each ranking against the expert
trial labels (qrels: 2 = eligible):

    recall@K   share of the patient's eligible trials found in the top K (K = 10, 100, 1000)
    P@10       share of the top 10 that are eligible

and report the mean over patients and the visible-minus-full difference with a patient-level paired
bootstrap CI. The difference is what masking costs retrieval: the selector's questions can only help
with trials that were retrieved in the first place.

Three retrievers, all from retrieval/__init__.py: keyword overlap, BM25, and their rank fusion (the
default stack). MedCPT is left out here: encoding 50,000 trials needs a GPU.

The retriever reads only the note text and the trial records; the qrels are used for scoring only.
Writes results/retrieval/retrieval_{split}.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trialguide.retrieval import _note_text, _tokens, _validate_trials
from trialguide.schemas import VisibleNote

KS = (10, 100, 1000)
METHODS = ("keyword", "bm25", "fused")


class CorpusIndex:
    """The retriever's three rankings over one fixed corpus, with the corpus prepared once.

    retrieval.retrieve_trials re-tokenises every trial on every call, which is fine for a patient's
    few dozen candidates but takes minutes per query on a 26,000-trial corpus. This class does the
    same computation (same tokens, same scores, same tie-breaks, same fusion) with the corpus work
    done once; a test checks it gives the same order as retrieve_trials on a small corpus.
    """

    def __init__(self, trials):
        from rank_bm25 import BM25Okapi
        prepared = _validate_trials(trials)
        self.ids = [row[0] for row in prepared]
        self.token_sets = [set(row[3]) for row in prepared]
        self.bm25 = BM25Okapi([row[3] for row in prepared])

    def keyword(self, note: VisibleNote) -> list[str]:
        query = set(_tokens(_note_text(note)))
        if not query:
            return []
        scored = [(len(query & toks) / len(query), tid) for tid, toks in zip(self.ids, self.token_sets)]
        return [tid for score, tid in sorted(scored, key=lambda x: (-x[0], x[1])) if score > 0]

    def bm25_rank(self, note: VisibleNote) -> list[str]:
        query = _tokens(_note_text(note))
        if not query:
            return []
        scores = self.bm25.get_scores(query)
        order = sorted(zip(scores, self.ids), key=lambda x: (-x[0], x[1]))
        return [tid for score, tid in order if score > 0]

    def fused(self, note: VisibleNote) -> list[str]:
        """Rank fusion as in retrieve_trials: sum of 1/rank over the rankings a trial appears in."""
        ranks = [{tid: i + 1 for i, tid in enumerate(r)} for r in (self.keyword(note), self.bm25_rank(note))]
        scored = [(sum(1 / r[tid] for r in ranks if tid in r), tid) for tid in self.ids]
        return [tid for score, tid in sorted(scored, key=lambda x: (-x[0], x[1])) if score > 0]

    def rank(self, note: VisibleNote, method: str) -> list[str]:
        return {"keyword": self.keyword, "bm25": self.bm25_rank, "fused": self.fused}[method](note)


def scores(ranking: list[str], eligible: set) -> dict:
    """recall@K and P@10 of one ranking against one patient's eligible trials."""
    out = {f"recall@{k}": len(eligible & set(ranking[:k])) / len(eligible) for k in KS}
    out["P@10"] = len(eligible & set(ranking[:10])) / 10
    return out


def read_corpus(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_eligible(path: Path) -> dict:
    """qrels -> {patient: set of trials labelled eligible (score 2)}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)                                           # header: query-id corpus-id score
        for line in f:
            pid, tid, score = line.rstrip("\n").split("\t")
            if score == "2":
                out.setdefault(pid, set()).add(tid)
    return out


def note_of(patient, hidden_ids) -> VisibleNote:
    hidden = set(hidden_ids)
    return VisibleNote(patient_id=patient.patient_id, split=patient.split,
                       sentences={i: s for i, s in patient.sentences.items() if i not in hidden})


def main(argv=None) -> None:
    from trialguide.data.patients import load_patients
    from trialguide.eval.stats import paired_bootstrap
    from trialguide.schemas import Split

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/retrieval"))
    args = ap.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    split = Split(manifest["split"])
    if split == Split.TEST and not args.allow_test:
        raise SystemExit("test split: pass --allow-test")
    patients = load_patients(args.raw, split, allow_test=args.allow_test)
    by_cohort = {}
    for pid, info in manifest["patients"].items():
        by_cohort.setdefault(info["cohort"], []).append(pid)

    rows, skipped = [], []
    for cohort, pids in sorted(by_cohort.items()):
        index = CorpusIndex(read_corpus(args.raw / "trialgpt" / cohort / "corpus.jsonl"))
        eligible = read_eligible(args.raw / "trialgpt" / cohort / "qrels" / "test.tsv")
        print(f"{cohort}: {len(index.ids)} trials, {len(pids)} patients", flush=True)
        for pid in sorted(pids):
            if not eligible.get(pid):
                skipped.append(pid)                       # no eligible trial: recall is undefined
                continue
            full = note_of(patients[pid], [])
            visible = note_of(patients[pid], manifest["patients"][pid]["hidden_ids"])
            row = {"patient_id": pid, "cohort": cohort, "eligible": len(eligible[pid])}
            for method in METHODS:
                for cond, note in (("full", full), ("visible", visible)):
                    for k, v in scores(index.rank(note, method), eligible[pid]).items():
                        row[f"{method}|{cond}|{k}"] = v
            rows.append(row)

    metrics = [f"recall@{k}" for k in KS] + ["P@10"]
    table = {}
    for method in METHODS:
        for m in metrics:
            full = np.array([r[f"{method}|full|{m}"] for r in rows])
            vis = np.array([r[f"{method}|visible|{m}"] for r in rows])
            diff = paired_bootstrap({"f": full, "v": vis},
                                    lambda d: d["v"].mean(axis=-1) - d["f"].mean(axis=-1))
            table.setdefault(method, {})[m] = {"full": float(full.mean()), "visible": float(vis.mean()),
                                               "visible_minus_full": diff.estimate,
                                               "ci95": [diff.low, diff.high]}
    result = {"manifest": str(args.manifest), "split": split.value, "mask": manifest.get("mask"),
              "fraction": manifest.get("fraction"), "patients_scored": len(rows),
              "skipped_no_eligible_trial": skipped, "table": table, "per_patient": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"retrieval_{split.value}.json").write_text(json.dumps(result, indent=1) + "\n")

    print(f"\n{split.value}: {len(rows)} patients scored ({len(skipped)} without an eligible trial skipped)")
    print(f"{'method':8s} {'metric':10s} {'full':>7s} {'visible':>8s}  visible-full [95% CI]")
    for method in METHODS:
        for m in metrics:
            t = table[method][m]
            print(f"{method:8s} {m:10s} {t['full']:7.3f} {t['visible']:8.3f}  "
                  f"{t['visible_minus_full']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
