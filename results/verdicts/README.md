# H1, H2 and McNemar on the test run (T36, statistics half)

Computed from the saved outputs of the frozen test run (`results/test_run/`) by
`python -m trialguide.eval.verdicts --out results/verdicts`. No model was called. All intervals are paired
patient-level bootstrap 95% CIs (10,000 resamples; 73 test patients for H1). The verdict wording is the
proposal's (the RQ table with H1 and H2, and the Statistics paragraph).

**Bottom line: neither hypothesis is supported, and neither is contradicted.** Both verdicts are
"inconclusive". For H1 that is a narrow call (see below).

## H1: recovery of B7 at least 10 points above fixed-order questioning (B4), false-eligible rate no higher

Recovery = (NDCG@10 of the system minus B6) / (B1 minus B6). The B1 to B6 gap is **0.053** NDCG, above the
0.02 floor, so recovery is reportable, but it is a small denominator.

| | Estimate | 95% CI |
|---|---|---|
| Recovery B7 | 7.6% | -46.8% to +56.4% |
| Recovery B4 | 4.4% | -51.6% to +51.9% |
| **Recovery gain, B7 minus B4** (the H1 quantity) | **+3.2 points** | **-0.1 to +10.6** |
| NDCG@10 difference, B7 minus B4 | +0.0017 | 0.0000 to +0.0044 |
| False-eligible rate (rule level), B7 / B4 | 1.88% / 1.88% | difference exactly 0 |

- **Verdict: inconclusive.** The estimate (3.2 points) is below the 10-point target, so H1 is not supported.
  It is not "not supported" under the proposal's rule only because the upper end of the CI (10.6) is just above
  10. The false-eligible clause holds (70 rules each, identical).
- **Why the intervals are so wide:** the whole ranking effect of masking is 0.053 NDCG, so a ratio against it
  swings widely between resamples; 415 of 10,000 resamples (4.2%) had a B1 to B6 gap under 0.02 and were
  dropped as the proposal specifies. The raw NDCG difference is the more stable statement: B7 and B4 differ
  by less than half a hundredth.
- The false-eligible rate is the team's rule-level one (`eval/recovery.py`): of the 3,728 rules the full-note
  checker settles and masking made B6 leave open, the share a system settles wrongly toward eligibility.
  B7 and B4 differ on 13 rules, none of them in that class.

## H2: risk-controlled harmful-inference rate within 5% and lower than B5 and an uncorrected tuned threshold

Primary run (cluster, expert labels, 26 test patients, 496 rules; `risk_test.json`):

| System | Threshold | Accepted | Harmful | Harmful rate (95% CI) |
|---|---|---|---|---|
| B2, accept every inference | 0 | 50 | 12 | 4.11% (1.37 to 7.63) |
| B5, fixed threshold | 0.9 | 44 | 10 | 2.97% (0.81 to 6.04) |
| **B7, risk-controlled** | 0.766 | 46 | 11 | **3.93% (1.25 to 7.41)** |

- **Verdict (the proposal's rule on the CI): inconclusive.** The upper bound (7.41%) is above 5%, so "supported"
  fails; the lower bound (1.25%) is below 5%, so "contradicted" fails. The point estimate is under 5%.
- **The "lower than B5" clause is not met.** B7's rate is higher than B5's (3.93% vs 2.97%).

Paired comparisons need per-rule checks, which the cluster run did not save, so they use the **Kaggle re-run**
in `results/resplits/` (same frozen code; it is close to but not identical with the cluster run, see that
README). B7 is evaluated at the frozen 0.766; the uncorrected tuned threshold is the smallest one whose
empirical calibration risk is at most 5% (no conformal correction), 0.562 on this run.

| | Rate | B7 minus it (95% CI) |
|---|---|---|
| B7 (0.766) | 3.94% | |
| B5 (0.9) | 2.23% | +1.71 points (0.00 to +4.70): B7 is higher |
| B2 (accept all) | 4.32% | -0.39 points (-0.97 to 0.00) |
| Uncorrected tuned (0.562) | 4.12% | -0.18 points (-0.55 to 0.00) |

Reading: against accept-everything and the uncorrected threshold, B7 is slightly lower, with intervals that
touch zero. Against the fixed 0.9 threshold it is not lower. **Risk control barely binds on this data**: with
26 calibration patients the conformal correction alone costs 1/27 = 3.7%, so the empirical calibration risk
may be at most about 1.35% to meet the 5% target, and accepting every inference already sits at 1.0% to 1.5%.
On the Kaggle run lambda-hat is simply the lowest candidate (it accepts everything); on the cluster run
accept-everything just failed the corrected bound (5.1%) and lambda-hat moved up to 0.766. That is why the
threshold shifted between runs and why B7 behaves close to B2.

## McNemar (exact), paired rule-level recovery on the 3,728 rules masking made B6 leave open

Outcome per rule: the system ends with the full-note status. Rules of one patient are not independent, so these
p-values overstate certainty; the bootstrap above is the main result.

| B7 against | B7 only right | Other only right | p |
|---|---|---|---|
| B6 (abstain) | 844 | 0 | < 1e-250 |
| A1 (no inference) | 820 | 4 | < 1e-230 |
| B4 (fixed order) | 9 | 4 | 0.27 |
| A6 (no risk control) | 1 | 38 | 1.5e-10 |
| B3 (ask everything) | 0 | 37 | 1.5e-11 |

Reading: inference is what recovers rules (A1, questions alone, recovers almost none). B7 and B4 do not differ
detectably at rule level. Turning risk control off (A6) recovers 37 more rules net, at the cost of more
rules settled wrongly toward eligibility (83 vs 70 false-eligible rules); asking about every fact (B3) recovers 37 more
than B7 at 6.7 questions per patient instead of 2.7.

## What this does not cover

Trial-level false-eligible and false-exclusion rates (the decisions per trial were not saved), the budget sweep,
and verdicts for E1/E2. Per-rule-type breakdowns are in `results/test_run/error_analysis.md`.
