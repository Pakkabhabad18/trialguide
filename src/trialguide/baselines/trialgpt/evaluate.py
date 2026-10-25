"""
Evaluate TrialGPT rankings as in the paper (Methods, Eqs. 27-32).

Not part of the released TrialGPT code; written by Team Zenith.

Relevance r: 0 = irrelevant, 1 = excluded (TREC) / potential (SIGIR), 2 = eligible,
taken from the label buckets in dataset/{corpus}/retrieved_trials.json. Only labeled
pairs are ranked, as in the paper ("unlabeled pairs are not included").

	NDCG@10 = DCG@10 / IDCG@10, DCG@k = sum_x r_x / log2(x + 1)
	P@10    = sum_{x<=10} r_x / (max(r) * 10)     (graded precision, max(r) = 2)
	AUROC   = excluding task, TREC only: excluded (r=1) vs eligible (r=2), scored by
	          -trial_score. The paper does not state the negative class; we use
	          eligible because that is the distinction the excluding task is about.

syntax: python -m trialguide.baselines.trialgpt.evaluate ${corpus} ${ranking_path} [${matching_results_path}] [--data_dir DIR]
"""

import argparse
import json
import math
import os
import random

import numpy as np
from sklearn.metrics import roc_auc_score


def dcg(gains, k=10):
	return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg(gains, k=10):
	ideal = dcg(sorted(gains, reverse=True), k)
	return dcg(gains, k) / ideal if ideal > 0 else None


def precision(gains, k=10, max_r=2):
	return sum(gains[:k]) / (max_r * k)


def label_distribution(matching_results):
	counts = {"inclusion": {}, "exclusion": {}}
	unparsed = 0
	sides = 0
	for label2trials in matching_results.values():
		for trials in label2trials.values():
			for results in trials.values():
				for side in ["inclusion", "exclusion"]:
					sides += 1
					preds = results.get(side) if isinstance(results, dict) else None
					if not isinstance(preds, dict):
						unparsed += 1
						continue
					for info in preds.values():
						lab = info[2] if isinstance(info, list) and len(info) == 3 else "malformed"
						counts[side][lab] = counts[side].get(lab, 0) + 1
	return counts, unparsed, sides


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("corpus")
	parser.add_argument("ranking_path")
	parser.add_argument("matching_results_path", nargs="?")
	parser.add_argument("--data_dir", default=os.getenv("TRIALGPT_DATA", "data/trialgpt"))
	args = parser.parse_args()

	corpus = args.corpus
	rankings = json.load(open(args.ranking_path))
	dataset = json.load(open(f"{args.data_dir}/{corpus}/retrieved_trials.json"))
	pid2rel = {
		inst["patient_id"]: {t["NCTID"]: int(l) for l in "012" for t in inst.get(l, [])}
		for inst in dataset
	}

	rng = random.Random(0)
	ndcgs, precs, rand_ndcgs, aurocs = [], [], [], []
	for pid, trial2score in rankings.items():
		rel = pid2rel[pid]
		ranked = [t for t in trial2score if t in rel]
		gains = [rel[t] for t in ranked]

		n = ndcg(gains)
		if n is None:
			continue  # no relevant trial among the candidates
		ndcgs.append(n)
		precs.append(precision(gains))

		shuffled = []
		for _ in range(100):
			g = gains[:]
			rng.shuffle(g)
			shuffled.append(ndcg(g))
		rand_ndcgs.append(np.mean(shuffled))

		if corpus.startswith("trec"):
			pairs = [(rel[t], trial2score[t]) for t in ranked if rel[t] in (1, 2)]
			ys = [int(r == 1) for r, _ in pairs]
			if 0 < sum(ys) < len(ys):
				aurocs.append((ys, [-s for _, s in pairs]))

	report = {
		"corpus": corpus,
		"n_patients": len(ndcgs),
		"n_pairs": sum(len(v) for v in rankings.values()),
		"NDCG@10": float(np.mean(ndcgs)),
		"P@10": float(np.mean(precs)),
		"NDCG@10_random_ranking": float(np.mean(rand_ndcgs)),
	}
	if aurocs:
		ys = [y for pair in aurocs for y in pair[0]]
		ss = [s for pair in aurocs for s in pair[1]]
		report["AUROC_excluding_pooled"] = float(roc_auc_score(ys, ss))

	if args.matching_results_path:
		counts, unparsed, sides = label_distribution(json.load(open(args.matching_results_path)))
		report["criterion_labels"] = counts
		report["unparsed_outputs"] = f"{unparsed}/{sides}"
		for side in counts:
			total = sum(counts[side].values())
			if total:
				report[f"{side}_not_enough_information_rate"] = counts[side].get("not enough information", 0) / total

	print(json.dumps(report, indent=2))
