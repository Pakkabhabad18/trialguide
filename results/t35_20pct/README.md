# T35 (partial): 20% masking on the same 24 test patients, B6, B2, B5, B7 and B4

Companion to `results/t35_60pct/`: **same 24 test patients, trials and masks logic, same frozen code (`55be3fe`), same Kaggle
setup (2 x T4, Qwen3-8B checker, the official gated Llama-3.1-8B-Instruct doctor, 8-bit greedy, lambda-hat 0.766)**, with
20% of each note hidden instead of 60%. Read the caveats in `results/t35_60pct/README.md` first; they apply here too:
the cluster and Kaggle outputs differ (so only contrasts within this run are clean), 24 patients are few, and this particular
subset behaves differently from all 73 at 40%. No H1/H2 verdict is given for a subset. The primary results in the README
(73 patients, cluster, 40%) are unchanged.

## Results (24 test patients, NDCG@10)

| System | Mean NDCG@10, Kaggle, 20% masking |
|---|---|
| B6 abstain | 0.751 |
| B2 every inference | 0.766 |
| B5 threshold 0.9 | 0.756 |
| **B7 TrialGuide** | **0.762** |
| B4 fixed-order questions | 0.762 |

Clean contrasts (same run, paired patient-level bootstrap, 95% CI):

| | Difference in NDCG@10 |
|---|---|
| B7 minus B6 | +0.010 (-0.012 to +0.033) |
| B7 minus B4 | -0.001 (-0.003 to +0.000) |
| B7 minus B2 | -0.004 (-0.018 to +0.008) |
| B4 minus B6 | +0.011 (-0.011 to +0.034) |

No system differs from B6 beyond noise, and B7 and B4 are practically identical. This is the same picture as at 60% and in the
full 40% run. At 20% little is hidden (B1, the full note, is only 0.039 above B6, with an interval that includes zero:
-0.008 to +0.085, mixed environment), so there is little for questions or inference to recover.
Recovery against that small gap is **not interpretable** (B7 +27% and B4 +29%, intervals about -30% to +90%).

**The doctor** (20%): B7 asked 2.54 questions per patient (61 answers: 36 "not documented" = 59%, 10 "probably", 12 "sure",
3 "unsure"); B4 2.58 (62 answers: 40 = 65%, 11, 10, 1).

## Across the three fractions on these 24 patients (Kaggle unless marked)

| | 20% | 60% | 40% (cluster) |
|---|---|---|---|
| B6 | 0.751 | 0.712 | 0.708 |
| B7 | 0.762 | 0.717 | 0.755 |
| B7 minus B6 | +0.010 (-0.012 to +0.033) | +0.005 (-0.039 to +0.048) | +0.047 (+0.012 to +0.082) |
| B7 minus B4 | -0.001 | -0.003 | +0.002 |

The 40% column comes from the cluster run on the same patients, so it is not comparable with the other two columns. The B7
gain at 40% (+0.047) is the subset anomaly noted in the 60% README (all 73 patients: +0.004); inside the Kaggle runs the
gain is within noise at both 20% and 60%. B6 falls from 0.751 at 20% to about 0.71 at 40% and 60%, so more masking costs ranking
quality only up to a point on this subset, but two of the three columns come from different machines.

Files: `subset_run.json` (all contrasts and per-patient NDCG), `raw/` (Kaggle summaries, the doctor's turns, manifests, run
info). Reproduce with `python -m trialguide.eval.subset_run --fraction 20 --phase1 <phase 1 output> --phase2 <phase 2 output>
--out results/t35_20pct`.
