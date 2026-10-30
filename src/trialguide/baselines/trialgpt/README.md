# TrialGPT baseline (T4, later B6)

A reproduction of [TrialGPT](https://github.com/ncbi-nlp/TrialGPT) (Jin et al.,
*Nature Communications* 2024) that runs on an open model instead of GPT-4. It is
the "standard abstention" system: when a note does not settle a criterion it
answers *not enough information* and never infers or asks. Baseline B6 (T16) is
this code run on the development patients with the team backbone.

The code is TrialGPT's, a US Government public-domain work (`LICENSE_NCBI`).
Prompts and the ranking formula are unchanged.

## Pipeline

| Step | Module | LLM calls per patient–trial pair | Output |
|---|---|---|---|
| Matching | `run_matching` (prompts in `matching.py`) | 2 (inclusion, exclusion) | per criterion: reasoning, evidence sentence ids, label |
| Aggregation | `run_aggregation` (prompt in `ranking.py`) | 1 | relevance R (0–100) and eligibility E (−R to R) |
| Ranking | `rank_results` | 0 | trial score = % inclusion met − 1[any not included] − 1[any excluded] + (R+E)/100 |
| Evaluation | `evaluate` (ours) | 0 | NDCG@10, P@10, AUROC, label counts |

Candidate trials come from TrialGPT's released `retrieved_trials.json`, which holds
only the expert-labelled trials of each patient, grouped by label (0 irrelevant,
1 excluded/potential, 2 eligible). Retrieval itself is not run here.

## Running

Data: `scripts/download_data.sh` puts TrialGPT's patients, labels and candidate
trials in `data/raw/trialgpt/` (the default; override with `TRIALGPT_DATA`).
Sentence splitting needs NLTK's tokenizer data once:
`python -c "import nltk; nltk.download('punkt_tab')"`.

```bash
mkdir -p logs
# smoke test: 3 SIGIR patients, at most 5 trials per label
sbatch -w gnode097 --export=ALL,HF_HOME=/ssd_scratch/$USER/hf scripts/run_trialgpt.sbatch
# chosen patients, all their trials (e.g. the dev split once T8 is frozen)
PYTHONPATH=src python -m trialguide.baselines.trialgpt.run_matching sigir Qwen/Qwen3-8B --patient_ids dev_ids.txt
```

All settings (model, corpus, subset, backend) are variables at the top of
`scripts/run_trialgpt.sbatch`. Outputs go to `outputs/trialgpt/` (git-ignored).
Every LLM call is cached in `outputs/llm_cache.jsonl`, so an interrupted run
resumes where it stopped, and logged with model revision, date and seed in
`outputs/llm_calls.jsonl`.

## Changes from the released code

| Change | Why |
|---|---|
| Model calls go through `trialguide.llm.client` (local transformers, an OpenAI-compatible server such as vLLM, or the original Azure) | No paid API; caching and logging |
| `parse_json` also accepts JSON wrapped in prose or `<think>` blocks | The original only strips backticks; small models add text |
| Qwen3 thinking mode switched off | The prompts ask for plain JSON |
| Output limit 8192 tokens | 4096 cut off answers for trials with 40+ criteria |
| If `parse_criteria` leaves a criteria list empty, that side returns `{}` without a call | See below |
| Aggregation reads trial text from `retrieved_trials.json` instead of `trial_info.json` | Same fields; the 1.2 GB download is very slow on our cluster |
| Patient subsets, per-label trial cap, concurrent requests | Smoke tests and dev-split runs |
| `rank_results` saves the ranking and tolerates unparsed outputs; failed pairs stop the pipeline | The original crashes on a string; partial results must not be evaluated |
| `evaluate.py` added | The repo ships no evaluation code |

**Empty criteria lists.** `parse_criteria` splits criteria on blank lines and
drops every chunk containing "inclusion criteria" or "exclusion criteria",
assuming it is a header. When the header and the first criterion share a chunk
(e.g. `"inclusion criteria: Acute myocardial infarction, ..."`) the criterion is
dropped too, and some trials have no criteria at all. About 1% of trials end up
with an empty list (SIGIR 25, TREC 2021 109, TREC 2022 91 trial sides). Given an
empty list, Qwen3-8B invents 20–150 criteria. We return `{}` for that side, which
keeps the original parser (and so the original prompts) while not scoring
invented criteria. Fixing the parser would change what every system sees, so if
the team wants it, it goes into all systems B1–B7 at once.

## Evaluation

As in the paper (Methods, Eqs. 27–32): gains 0/1/2 from the label buckets,
NDCG@10 and graded P@10 = Σ gains in top 10 / (2 × 10), over labelled trials
only. AUROC (TREC only) separates excluded (1) from eligible (2) trials by the
negated trial score; the paper does not state its negative class. The report
also gives NDCG@10 of a random ranking of the same trials, the criterion label
counts and the *not enough information* rate.

## Smoke test result (T4)

`results/trialgpt_smoke_sigir_qwen3-8b.json`: Qwen3-8B, 3 SIGIR patients, 43
pairs. NDCG@10 0.773 (random ranking 0.620), P@10 0.517, 0/86 unparsed outputs;
*not enough information* on 29% of inclusion and 13% of exclusion criteria. With
3 patients and at most 15 trials each this only shows the pipeline works; it is
not comparable to the paper's 0.7275 (GPT-4, all trials, three cohorts).

## Known issues

- Qwen3-8B labels many criteria *not applicable* (108 of 210 exclusion labels in the smoke test),
  including some that should be *not included*. Those are invisible to an
  infer/ask/abstain policy that acts only on *not enough information*.
- The `hf` backend runs one model per process and only greedy decoding; the
  checker's 5-sample confidence (T12) needs sampling.
- Resuming a run reuses stored matching results as they are; start from an empty
  `outputs/` folder when code that changes matching outputs has changed.
