# Development run with 60% masking (13 Nov 2026)

The stronger-masking check: the same 37 development patients and trials as `results/dev_run/`,
the same frozen system (code commit 55be3fe, lambda-hat = 0.766, answerability prior from the 40%
dev turns), but rule-targeted masking hiding 60% of sentences instead of 40%. The question was
whether asking helps more when more is hidden. Run after the test run; nothing was tuned on it.

## Trial ranking against the expert labels

| System | NDCG@10 | P@10 | Questions / patient | NDCG@10 minus B6 [95% CI] |
|---|---|---|---|---|
| B1 reference (full note) | 0.770 | 0.497 | 0 | +0.060 [+0.010, +0.112] |
| B6 abstain | 0.710 | 0.486 | 0 | 0 |
| B2 every inference | 0.715 | 0.493 | 0 | +0.006 [-0.023, +0.032] |
| B5 confidence >= 0.9 | 0.715 | 0.496 | 0 | +0.006 [-0.021, +0.031] |
| B7 TrialGuide | 0.709 | 0.493 | 2.70 | -0.001 [-0.034, +0.031] |
| B7-eq1 (no answerability weight) | 0.705 | 0.496 | 2.70 | -0.005 [-0.039, +0.031] |
| A1 no inference | 0.708 | 0.488 | 2.78 | -0.001 [-0.026, +0.020] |
| A2 fixed order | 0.701 | 0.495 | 2.84 | -0.008 [-0.043, +0.028] |
| A3 no ranking term | 0.701 | 0.492 | 2.84 | -0.008 [-0.040, +0.022] |
| A5 no tools | 0.713 | 0.488 | 2.81 | +0.004 [-0.027, +0.033] |
| A6 no risk control | 0.709 | 0.493 | 2.70 | -0.000 [-0.033, +0.031] |
| B4 fixed-order questions | 0.701 | 0.495 | 2.84 | -0.008 [-0.043, +0.028] |
| B3 ask every candidate | 0.713 | 0.491 | 9.16 | +0.004 [-0.027, +0.035] |

CIs: patient-level paired bootstrap (5,000 resamples). B1 is the full-note run of `results/dev_run/`.
Doctor answers for B7: 63 not documented, 29 probably, 5 sure, 3 unsure (100 questions).

## Recovery of the full-note result (rule level)

2,431 rules were settled on the full note but left unresolved by B6 with 60% hidden (1,915 at 40%).

| System | Recovered [95% CI] | Wrong | Wrong toward eligibility | Kendall tau vs full |
|---|---|---|---|---|
| B6 | 0.000 | 0.000 | 0.000 | 0.217 |
| B2 | 0.097 [0.062, 0.141] | 0.029 | 0.009 | 0.190 |
| B5 | 0.086 [0.054, 0.125] | 0.027 | 0.007 | 0.201 |
| B7 | 0.124 [0.085, 0.177] | 0.038 | 0.011 | 0.184 |
| B7-eq1 | 0.121 [0.082, 0.172] | 0.037 | 0.011 | 0.154 |
| A1 | 0.017 [0.011, 0.024] | 0.005 | 0.001 | 0.207 |
| A2 | 0.121 [0.084, 0.168] | 0.038 | 0.012 | 0.148 |
| A3 | 0.119 [0.084, 0.162] | 0.036 | 0.010 | 0.166 |
| A5 | 0.137 [0.093, 0.190] | 0.039 | 0.012 | 0.217 |
| A6 | 0.128 [0.088, 0.182] | 0.039 | 0.012 | 0.180 |
| B4 | 0.121 [0.084, 0.168] | 0.038 | 0.012 | 0.148 |
| B3 | 0.130 [0.089, 0.184] | 0.041 | 0.013 | 0.205 |

## Reading

- Hiding 60% costs 0.060 NDCG@10 (CI excludes 0), against 0.007 at 40% on the same patients.
- Again no system recovers it: every difference from B6 is within noise.
- Inference recovers a smaller share of lost rules than at 40% (B7 12.4% against 24.5%): with
  less text left there is less to infer from. Questions alone (A1) recover 1.7%; most answers
  are still "not documented".
- So stronger masking does not reveal a benefit of asking that 40% hid.

Files: `ndcg_table.json`, `recovery.json`, `systems_B6_B2_B5.json`, `asking.json`,
`asking_A5.json`, `asking_B3.json`, `turns_asking.jsonl`, `manifest.json`.
