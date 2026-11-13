# RAG check: trial retrieval from the visible note (13 Nov 2026)

Code: `src/trialguide/eval/retrieval_check.py`, using the retriever in `src/trialguide/retrieval/` unchanged.
CPU only, about 3 minutes per split.

Setting: each patient's note searches the whole trial corpus of its cohort (SIGIR 3,620 trials; TREC 2021
26,148; TREC 2022 26,580) twice: with the full note, and with the visible note (the same rule-targeted 40%
mask as every other run, hidden sentence ids from the run manifests). Rankings are scored against the expert
trial labels (qrels, 2 = eligible). Patients with no eligible trial are skipped (one per split).
Differences are visible minus full, with patient-level paired bootstrap 95% CIs (10,000 resamples).

## Test split (72 patients)

| Retriever | Metric | Full note | Visible note | Visible − full [95% CI] |
|---|---|---|---|---|
| BM25 | P@10 | 0.147 | 0.065 | −0.082 [−0.125, −0.043] |
| BM25 | recall@100 | 0.198 | 0.110 | −0.088 [−0.119, −0.059] |
| BM25 | recall@1000 | 0.481 | 0.311 | −0.170 [−0.215, −0.127] |
| Fused (keyword + BM25, the default) | P@10 | 0.113 | 0.054 | −0.058 [−0.096, −0.026] |
| Fused | recall@1000 | 0.401 | 0.285 | −0.116 [−0.165, −0.066] |
| Keyword overlap | P@10 | 0.039 | 0.022 | −0.017 [−0.046, +0.006] |
| Keyword overlap | recall@1000 | 0.205 | 0.202 | −0.003 [−0.037, +0.033] |

## Development split (36 patients)

| Retriever | Metric | Full note | Visible note | Visible − full [95% CI] |
|---|---|---|---|---|
| BM25 | P@10 | 0.167 | 0.047 | −0.119 [−0.200, −0.047] |
| BM25 | recall@1000 | 0.408 | 0.205 | −0.204 [−0.282, −0.131] |
| Fused | P@10 | 0.094 | 0.031 | −0.064 [−0.117, −0.019] |
| Fused | recall@1000 | 0.330 | 0.152 | −0.178 [−0.247, −0.115] |

All metrics (recall@10, @100, @1000, P@10) for all three retrievers are in `retrieval_test.json` and
`retrieval_dev.json`, with per-patient rows.

## Reading

- **Masking hurts retrieval much more than it hurts ranking.** On test, BM25 finds 48% of the eligible trials
  in its top 1,000 with the full note and 31% with the visible note. In the ranking runs the candidate trials
  are fixed in advance (the benchmark's labelled candidates, as in TrialGPT's ranking stage), and there the
  same mask lowers NDCG@10 by 0.007 on dev and 0.053 on test (B1 vs B6). The rule-targeted mask hides the
  sentences trials ask about, and those are also the words retrieval matches on.
- **So the ranking results are an upper bound for a deployed system.** A trial that retrieval misses never
  reaches the checker, and no question can bring it back. Re-running retrieval after each doctor answer is
  the obvious fix; we did not test it.
- **BM25 alone beats the default fused stack.** Keyword overlap scores the share of note words a trial
  contains, which favours long trials; fusing it with BM25 lowers P@10 (0.147 to 0.113 on full notes).
- **Absolute numbers are low, for two reasons.** Retrieval uses the raw note text; TrialGPT instead
  generates search keywords with GPT-4 and adds a MedCPT dense retriever, and we did neither (MedCPT on
  50,000 trials needs a GPU). And TREC patients have many eligible trials (median 53 to 71), so recall@10
  is small by construction; P@10 and recall@1000 are the useful numbers.
