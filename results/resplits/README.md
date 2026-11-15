# Exploratory resplits (T36)

How much does the harmful-inference rate depend on which of the 52 expert-labelled patients were in
calibration and which in test? The primary result (calibration 26 / test 26) is one draw. Here the
split is redrawn 200 times (seed 20261115, fixed before running), λ̂ is refitted on each calibration half
(α = 5%, `risk_control.calibrate`) and the harmful-inference rate is measured on the other half.
**Exploratory only:** the resplits reuse the test patients, so they are not a second test and were
never used to choose a model, prompt or threshold (proposal, Section 5).

## Where the checks come from (read this first)

The cluster runs kept only totals, not the per-rule checks, so the calibration and test checks were
**re-run on Kaggle** (2 x T4, 74 minutes, `scripts/kaggle/risk_checks/`) with the frozen code commit
`55be3fe`, Qwen3-8B 8-bit, greedy, tools on, rule mask hiding 40% of sentences. **Kaggle did not
reproduce the cluster numbers exactly**, probably because of different GPUs and library versions
(`pip_freeze_kaggle.txt`):

| | Cluster (published) | Kaggle re-run |
|---|---|---|
| λ̂ on the 26 calibration patients | 0.766 | **0.562** |
| Calibration risk at λ̂ | 1.33% | 0.96% |
| Test, unsettled rules | 386 | 397 |
| Test B2 (accept all): accepted / harmful | 50 / 12 (4.11%) | 52 / 14 (4.32%) |
| Test B5 (0.9): accepted / harmful | 44 / 10 (2.97%) | 43 / 10 (2.23%) |
| Test B7 at the frozen λ̂ = 0.766: accepted / harmful | 46 / 11 (3.93%) | 47 / 12 (3.94%) |

So the system's behaviour is close but not identical across machines, and **λ̂ itself moved from 0.766
to 0.562** on the same 26 patients. Every number below uses the Kaggle checks only (calibration and test
from the same environment), and the primary cluster numbers in the README are unchanged.

## Results (200 resplits of the 52 patients, 26 / 26)

| System | Test harmful rate: mean (sd) | 2.5% / median / 97.5% | Resplits at or below 5% | Mean accepted |
|---|---|---|---|---|
| B7, risk-controlled (λ̂ refitted each time) | 1.4% (1.0) | 0.0% / 1.2% / 3.8% | 200 / 200 | 44.5 |
| B5, fixed threshold 0.9 | 1.3% (0.4) | 0.6% / 1.2% / 2.0% | 200 / 200 | 45.7 |
| B2, accept every inference | 2.6% (0.8) | 1.1% / 2.7% / 4.0% | 200 / 200 | 53.1 |

- λ̂ varies a lot between resplits: mean 0.887, sd 0.129, range 0.498 to 0.997. No resplit accepted nothing.
- The largest B7 rate in any resplit is 4.4%.
- **The primary split is an unfavourable draw.** Its B7 test rate (3.9%) is at least as high as in only
  2.0% of the resplits (B2: 1%), and its λ̂ (0.56 to 0.77) is lower than 90% to 94% of the resplit λ̂ values.
  A typical split gives a test rate around 1.2%. The headline 3.9% is therefore not an optimistic estimate.

## How to read this

- The 52 patients are the whole pool, and the primary split is one of the possible halves, so the
  resplits overlap heavily with it and with each other. The spread is real but narrower than the
  uncertainty of one new sample of patients.
- Each half has 26 patients and about 400 unsettled rules, with 44 to 53 accepted inferences. Rates
  are small counts: with a median of 14 unsettled rules per patient, one more harmful inference moves a half's rate by about 0.27 percentage points.
- The expert labels cover SIGIR only. Nothing here says how risk control behaves on TREC patients.
- "At or below 5%" refers to the point estimate of each half, not to its confidence bound. The primary
  result's CI (upper end above 5%) is the statement to quote for H2.

Files: `resplits.json` (summary and 200 rows), `checks_cal.jsonl` and `checks_test.jsonl` (the per-rule checks),
`calibration.json` and `risk_test_kaggle.json` (Kaggle re-run totals), `run_info.json`, `pip_freeze_kaggle.txt`.
Reproduce the analysis: `python -m trialguide.eval.resplits --cal results/resplits/checks_cal.jsonl
--test results/resplits/checks_test.jsonl --out results/resplits`.
