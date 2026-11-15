# T35 (partial): 60% masking on 24 test patients, plus the 20% run without the doctor

**What this is, and is not.** T35 asks for B4 and B7 at 20% and 60% masking on the 73 test patients. A full run does not
fit the compute available (about 8,600 checker calls per fraction), so this is a **seeded 24-patient subset** of the test
set (9 SIGIR, 9 TREC 2021, 6 TREC 2022; same sampling seed, trials and masks as the 40% test run), run on **Kaggle**
(2 x T4, frozen code commit `55be3fe`, Qwen3-8B checker and the official gated `meta-llama/Llama-3.1-8B-Instruct`
doctor, both 8-bit and greedy, lambda-hat 0.766, answerability prior from development). The 20% run for B4 and B7 was
**not** run (its phase 2 never started); only B6, B2 and B5 exist at 20%. The primary results in the README are the cluster
test run on all 73 patients and are unchanged.

## Read this before the numbers

1. **Two environments.** The cluster (RTX 3080 Ti) and Kaggle (T4) do not give identical outputs: on the same 26 patients
   lambda-hat came out 0.562 on Kaggle against 0.766 on the cluster (`results/resplits/README.md`). Contrasts **within** the
   Kaggle run are clean (same patients, trials and environment, paired). Anything that sets a Kaggle number against a
   cluster number (B1, the 40% run) also contains that difference, and is marked "mixed" below.
2. **24 patients are few, and this subset is not typical.** On these same 24 patients the cluster's 40% run gives
   B7 minus B6 = +0.047 (CI +0.012 to +0.082), but on all 73 patients it is +0.004 (-0.020 to +0.028). A 24-patient draw
   can look clearly better than the whole set, so nothing here is evidence of an effect.
3. No H1 or H2 verdict is given for the subset: the proposal's verdicts are defined on the full test split (`results/verdicts/`).

## Results at 60% masking (24 test patients, NDCG@10)

| System | Mean NDCG@10 (Kaggle, 60%) |
|---|---|
| B6 abstain | 0.712 |
| B2 every inference | 0.704 |
| B5 threshold 0.9 | 0.696 |
| **B7 TrialGuide** | **0.717** |
| B4 fixed-order questions | 0.719 |

**Clean contrasts (same run, paired patient-level bootstrap, 95% CI):**

| | Difference in NDCG@10 |
|---|---|
| B7 minus B6 | +0.005 (-0.039 to +0.048) |
| B7 minus B4 | -0.003 (-0.010 to +0.002) |
| B7 minus B2 | +0.013 (-0.012 to +0.040) |
| B4 minus B6 | +0.008 (-0.039 to +0.053) |

No system differs from B6 beyond noise, and B7 and B4 are indistinguishable (the interval is about one hundredth wide). This
matches the full 40% test run.

**Mixed (Kaggle number against the cluster's full-note B1):** B1 is 0.790 on these 24 patients, so masking at 60% costs
B1 minus B6 = +0.078 (+0.008 to +0.145); at 40% on the same patients (cluster) it is +0.082 (+0.023 to +0.141). Going from
40% to 60% does not visibly cost more on this subset. Recovery against that gap is **not interpretable**: B7 +6% and B4 +10%
with intervals running from about -80% to +70%, because the denominator is a small, noisy gap.

**The doctor** (Kaggle, 60%): B7 asked 2.67 questions per patient (64 answers: 39 "not documented", 19 "probably", 5 "sure",
1 "unsure"); B4 2.79 (67 answers: 38, 22, 7, 0). 61% and 57% of answers were "not documented", against 70% for B7 in the
73-patient 40% run (136 of 194).

## The 20% run, B6, B2 and B5 only (no doctor, Kaggle)

| System | Mean NDCG@10, 20% masking, 24 patients |
|---|---|
| B6 | 0.751 |
| B2 | 0.766 |
| B5 | 0.756 |

B2 minus B6 = +0.015 (-0.009 to +0.038), B5 minus B6 = +0.005 (-0.015 to +0.025); B1 minus B6 at 20% is +0.039
(-0.008 to +0.085, mixed). At 20% there is little hidden, so little to recover. B6 reads 0.751 at 20%, 0.708 at 40% (cluster) and
0.712 at 60% (Kaggle): the last two are not comparable across environments, so do not read a trend into them.

## Files

`subset_run.json` (all contrasts and the per-patient NDCG of every system), `raw/` (Kaggle summaries, the doctor's turns,
manifests, run info), `raw_20pct_phase1/`. Reproduce the table from the Kaggle outputs with
`python -m trialguide.eval.subset_run --fraction 60 --phase1 <phase 1 output> --phase2 <phase 2 output> --out results/t35_60pct`.
Kaggle scripts: `scripts/kaggle/t35_phase1/`, `t35_phase2/` and the `t35_f20_*` copies.
