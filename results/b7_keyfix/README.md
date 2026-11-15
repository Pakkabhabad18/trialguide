# B7-keyfix: B7 without the copied "prior_chemotherapy" fact key (15 Nov 2026)

The fix (Prakash, commit 55a7cbb, `selector.drop_copied_key`) was found in the test-run error
analysis, so its **test result is post hoc** and does not replace B7. To change nothing else, it
was run on the frozen test-run code (55be3fe) with only 55a7cbb applied, reusing the same checker
outputs; lambda-hat = 0.766. 40% rule-targeted masking, budget 3.

| Split | System | NDCG@10 | P@10 | Questions / patient | Not documented | Minus B6 [95% CI] | Minus B7 [95% CI] |
|---|---|---|---|---|---|---|---|
| dev (37) | B7 | 0.743 | 0.497 | 2.43 | 55 of 90 (61%) | -0.021 [-0.061, +0.019] | 0 |
| dev (37) | B7-keyfix | 0.743 | 0.497 | 2.03 | 38 of 75 (51%) | -0.021 [-0.061, +0.019] | +0.000 [+0.000, +0.000] |
| test (73) | B7 | 0.735 | 0.466 | 2.66 | 136 of 194 (70%) | +0.004 [-0.020, +0.028] | 0 |
| test (73) | B7-keyfix | 0.734 | 0.465 | 2.29 | 102 of 167 (61%) | +0.003 [-0.021, +0.027] | -0.001 [-0.003, +0.001] |

CIs: patient-level paired bootstrap, 5,000 resamples.

Reading: the fix removes wasted questions (fewer questions, a smaller share answered "not
documented"), but the ranking is unchanged on both splits. The copied key cost questions, not
ranking quality.

Files: `asking_B7-keyfix_{dev,test}.json`, `turns_B7-keyfix_{dev,test}.jsonl`,
`final_B7-keyfix_{dev,test}.json`.
