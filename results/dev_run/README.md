# Development run: all baselines and ablations A1, A2, A3, A5, A6 (12 Nov 2026)

Setting: all 37 development patients, the first 4 trials of each expert label (430 patient-trial
pairs, about 12 trials per patient), rule-targeted masking hiding 40% of sentences. Checker
Qwen3-8B, simulated doctor Llama-3.1-8B-Instruct reading the full note, both 8-bit and greedy, on
one RTX 3080 Ti each. Budget 3 questions (B3: every candidate fact). Code commit ef60178.

Threshold: lambda-hat = 0.766, calibrated on the expert-labelled rules of the 26 calibration
patients (501 rules, alpha = 0.05; `calibration.json`). At lambda-hat the calibration
harmful-inference rate is 1.3%; the conformal bound is met (certifiable with 26 patients).

| System | What it is | NDCG@10 | P@10 | Questions / patient | NDCG@10 minus B6 [95% CI] |
|---|---|---|---|---|---|
| B1 reference | B6 on the full note | 0.770 | 0.497 | 0 | +0.007 [-0.036, +0.048] |
| B6 | Abstain on every gap (masked note) | 0.763 | 0.495 | 0 | 0 |
| B2 | Every inference accepted | 0.755 | 0.499 | 0 | -0.009 [-0.049, +0.030] |
| B5 | Inference if confidence >= 0.9 | 0.752 | 0.499 | 0 | -0.011 [-0.049, +0.026] |
| B7 | TrialGuide | 0.743 | 0.499 | 2.46 | -0.021 [-0.062, +0.020] |
| A1 | B7 without inference | 0.760 | 0.493 | 2.76 | -0.004 [-0.010, +0.002] |
| A2 | B7 with fixed question order | 0.748 | 0.497 | 2.81 | -0.015 [-0.056, +0.024] |
| A3 | B7 without the ranking term | 0.753 | 0.499 | 2.81 | -0.011 [-0.051, +0.029] |
| A5 | B7 without tools | 0.736 | 0.493 | 2.57 | -0.027 [-0.067, +0.012] |
| A6 | B7 without risk control | 0.746 | 0.499 | 2.46 | -0.018 [-0.058, +0.022] |
| B4 | Fixed-order questions | 0.748 | 0.497 | 2.81 | -0.015 [-0.056, +0.024] |
| B3 | Ask every candidate fact | 0.745 | 0.497 | 7.38 | -0.019 [-0.058, +0.020] |

CIs: patient-level paired bootstrap (5,000 resamples). A1 and A6 do not depend on the threshold
and come from the lambda = 0.9 run; in this code B4 and A2 use the same question strategy.

Doctor answers for B7: 67 not documented, 19 probably, 3 sure, 2 unsure (91 questions).

Reading:
- Masking barely moves the ranking: B6 drops from 0.770 (full note) to 0.763. With so little lost,
  there is little for inference or questions to recover on these patients and trials.
- No system differs from B6 beyond noise; every confidence interval includes 0.
- About three quarters of the doctor's answers are "not documented": the selector asks about
  facts many trials need, which the original notes mostly never recorded.
- Removing the tools (A5) gives the lowest score, but within noise; the tools change the prompt
  for 12 of 37 notes.

Files: `systems_B6_B2_B5.json` (full and masked), `asking_lambda_hat.json` (B7, A2, A3, A5, B4),
`asking_B3_lambda_hat.json`, `asking_lambda_0.9_with_A1_A6.json`, `calibration.json`,
`turns_lambda_hat.jsonl` (every question and answer), `manifest.json` (patients and trials),
`checks_full.jsonl` (the checker on every rule of the full notes: status, category, evidence, confidence).

## B7 rerun after the selector fix (answerability weight)

After the run above, one change was made on development data only (commit 2548fc3,
`src/trialguide/selector/answerability.py`): Eq. 1 is weighted by the chance that the doctor can
answer a question about the fact, estimated from the dev turns above and frozen in
`results/answerability_prior.json`. B7, A1, A5 and A6 were rerun with it (A2, A3 and B4 do not use
Eq. 1; B7-eq1 keeps the plain Eq. 1). This is the B7 used on the test split.

| System | NDCG@10 | P@10 | Questions / patient | Doctor answers |
|---|---|---|---|---|
| B7 | 0.743 | 0.497 | 2.43 | 55 not documented, 23 probably, 10 sure, 2 unsure |
| A1 | 0.761 | 0.493 | 2.76 | 54 not documented, 31 probably, 15 sure, 2 unsure |
| A5 | 0.734 | 0.492 | 2.54 | 55 not documented, 26 probably, 10 sure, 3 unsure |
| A6 | 0.743 | 0.497 | 2.43 | 55 not documented, 23 probably, 10 sure, 2 unsure |

The fix raised the share of answered questions (not documented 74% -> 61% for B7) but did not
change NDCG@10 (0.743 before and after).

Rule-level recovery on dev (`recovery.json`; 1,915 rules settled on the full note but left
unresolved by B6 on the masked note; `python -m trialguide.eval.recovery --run <run> --asking <dir>`):
B7 recovers 24.5% [18.8, 30.0] (wrong 6.3%, wrong toward eligibility 1.1%), B2 24.2%, B5 20.0%,
A1 0.8%, A5 29.1%, A6 26.6%, B3 26.1%.

To reproduce: `SYSTEMS="B7 A1 A5 A6" THRESHOLD=0.766 OUT=<dir> MANIFEST=results/dev_run/manifest.json
bash scripts/run_asking.sh` (the prior is read from `results/answerability_prior.json` by default).

Files: `asking_B7_answerability_rerun.json`, `turns_B7_answerability_rerun.jsonl`, `recovery.json`.
