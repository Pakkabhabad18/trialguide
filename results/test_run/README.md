# Test run: frozen system on the held-out test split (13 Nov 2026)

The test split was opened once, after the system was frozen on development data (code commit
55be3fe; answerability prior from dev turns only; lambda-hat = 0.766 from the calibration split).
Nothing was changed after seeing these numbers.

Setting: all 73 test patients, about 12 trials each (826 patient-trial pairs), rule-targeted
masking hiding 40% of sentences. Checker Qwen3-8B, simulated doctor Llama-3.1-8B-Instruct reading
the full note, both 8-bit and greedy. Budget 3 questions (B3: every candidate fact).

## Trial ranking against the expert labels

| System | What it is | NDCG@10 | P@10 | Questions / patient | NDCG@10 minus B6 [95% CI] |
|---|---|---|---|---|---|
| B1 reference | B6 on the full note | 0.784 | 0.487 | 0 | +0.053 [+0.016, +0.091] |
| B6 | Abstain on every gap (masked note) | 0.731 | 0.476 | 0 | 0 |
| B2 | Every inference accepted | 0.726 | 0.466 | 0 | -0.005 [-0.028, +0.019] |
| B5 | Inference if confidence >= 0.9 | 0.724 | 0.466 | 0 | -0.007 [-0.030, +0.018] |
| B7 | TrialGuide | 0.735 | 0.466 | 2.66 | +0.004 [-0.020, +0.028] |
| B7-eq1 | B7 without the answerability weight | 0.736 | 0.465 | 2.66 | +0.005 [-0.018, +0.029] |
| A1 | B7 without inference | 0.735 | 0.475 | 2.84 | +0.004 [-0.001, +0.010] |
| A2 | B7 with fixed question order | 0.733 | 0.466 | 2.81 | +0.002 [-0.021, +0.027] |
| A3 | B7 without the ranking term | 0.734 | 0.465 | 2.81 | +0.004 [-0.019, +0.028] |
| A5 | B7 without tools | 0.727 | 0.463 | 2.66 | -0.003 [-0.027, +0.021] |
| A6 | B7 without risk control | 0.734 | 0.466 | 2.63 | +0.003 [-0.021, +0.027] |
| B4 | Fixed-order questions | 0.733 | 0.466 | 2.81 | +0.002 [-0.021, +0.027] |
| B3 | Ask every candidate fact | 0.734 | 0.466 | 6.67 | +0.003 [-0.021, +0.028] |

CIs: patient-level paired bootstrap (5,000 resamples). Doctor answers for B7: 136 not documented,
35 probably, 20 sure, 3 unsure (194 questions).

## Recovery of the full-note result (rule level)

3,728 rules were settled (met / not met) by the checker on the full note but left unresolved by B6
on the masked note. Share of them each system ends with the full-note status:

| System | Recovered [95% CI] | Wrong | Wrong toward eligibility | Kendall tau vs full-note ranking |
|---|---|---|---|---|
| B6 | 0.000 | 0.000 | 0.000 | 0.294 |
| B2 | 0.214 [0.177, 0.255] | 0.064 | 0.022 | 0.239 |
| B5 | 0.187 [0.153, 0.224] | 0.057 | 0.015 | 0.237 |
| B7 | 0.226 [0.188, 0.267] | 0.064 | 0.019 | 0.247 |
| B7-eq1 | 0.224 [0.185, 0.265] | 0.064 | 0.019 | 0.246 |
| A1 | 0.008 [0.003, 0.012] | 0.005 | 0.001 | 0.294 |
| A2 | 0.225 [0.186, 0.266] | 0.064 | 0.019 | 0.244 |
| A3 | 0.221 [0.182, 0.262] | 0.063 | 0.019 | 0.246 |
| A5 | 0.234 [0.195, 0.277] | 0.065 | 0.019 | 0.253 |
| A6 | 0.236 [0.197, 0.277] | 0.067 | 0.022 | 0.244 |
| B4 | 0.225 [0.186, 0.266] | 0.064 | 0.019 | 0.244 |
| B3 | 0.236 [0.198, 0.277] | 0.065 | 0.020 | 0.243 |

## Harmful inferences against expert rule labels (26 test patients, 496 rules)

| System | Threshold | Accepted inferences | Harmful | Harmful rate [95% CI] |
|---|---|---|---|---|
| B2 | 0 | 50 | 12 | 4.1% [1.4, 7.6] |
| B5 | 0.9 | 44 | 10 | 3.0% [0.8, 6.0] |
| B7 | 0.766 | 46 | 11 | 3.9% [1.3, 7.4] |

B7's harmful rate is below alpha = 5% on test, but the upper end of its CI is above 5%.

## Reading

- Masking costs more on test than on dev: NDCG@10 drops by 0.053 (CI excludes 0) against 0.007
  on dev.
- No system recovers that loss in the ranking. Every system with inference or questions is within
  noise of B6; every CI includes 0.
- At rule level, inference recovers about a fifth of the lost rules (B7 22.6%), at the cost of
  about 6% wrong, 2% wrong toward eligibility. Questions alone (A1) recover under 1%: about 70%
  of the doctor's answers are "not documented", because the original notes rarely record the
  facts that were hidden or that the trials ask about.
- The recovered rules do not move the ranking toward the full-note ranking (tau 0.247 against
  0.294 for B6), so they are not the rules that decide the top 10.
- The answerability weight (B7 vs B7-eq1), question order (A2, B4), the ranking term (A3), tools
  (A5) and risk control (A6) make no measurable difference on test.

Files: `ndcg_table.json`, `recovery.json`, `risk_test.json`, `systems_B6_B2_B5.json`,
`asking.json`, `asking_A5.json`, `asking_B3.json`, `manifest.json`; for error analysis: `checks_full.jsonl`
and `checks_rule40.jsonl` (checker on every rule, full and masked notes), `turns_asking.jsonl`,
`turns_A5.jsonl`, `turns_B3.jsonl` (every question and answer) and `final_{system}.json` (final
status of every rule per patient).
