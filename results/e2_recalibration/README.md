# E2 recalibration (T38, exploratory)

The proposal recalibrates the risk threshold on part of E2, because calibration and test patients must be exchangeable and E2 notes are
not like the main benchmark. This is that step on the only E2 checker outputs that exist, the 10 patients of the smoke run
(`results/e2_smoke/`), rebuilt from its committed `rule_table.csv`. Run it with
`python -m trialguide.eval.e2_recalibration --table results/e2_smoke/rule_table.csv --out results/e2_recalibration`.

## Read this first

1. **This shows the mechanics and one qualitative point, not a validated E2 threshold.** There are 10 patients, 170 unsettled rules and
   only **6 harmful inferences in the whole run** (when accepting every inference). Every number below hangs on a handful of rules.
2. **The reference is the checker on the full note** (model-derived; E2 has no expert labels). "Harmful" means a masked-note inference
   toward eligibility that disagrees with that reference, as in `policy/risk_control.py`.
3. **No held-out E2 set.** The E2 test half stays sealed, so the in-sample numbers are optimistic. A leave-one-out version (threshold fitted on 9
   patients, applied to the 10th) is given as the out-of-sample view.

## Recalibration

With n = 10 patients the smallest certifiable target is 1/(n+1) = **9.1%**:

| Target alpha | Certifiable | Threshold |
|---|---|---|
| 2% | no | none, the policy would accept no inference toward eligibility |
| 5% | no | none (the main run's target cannot be met on 10 patients) |
| 10% | yes | lambda-hat = **0.992** (empirical risk 0.36%) |

At alpha = 10% the conformal correction alone uses 9.1 points of the 10, so only about 1% of empirical risk is left, which forces a
very strict threshold. The main run's threshold (0.766) would not carry over to these notes.

## Effect of the thresholds (in-sample, 170 unsettled rules)

| Threshold | Accepted | Harmful | Harmful rate (95% CI) | Coverage (95% CI) |
|---|---|---|---|---|
| B2, accept all (0) | 91 | 6 | 4.0% (0.9 to 7.5) | 53.5% (36.4 to 71.1) |
| B5, fixed 0.9 | 88 | 5 | 3.8% (0.6 to 7.5) | 51.8% (35.6 to 69.5) |
| Frozen main-run lambda-hat, 0.766 | 89 | 5 | 3.8% (0.6 to 7.5) | 52.4% (35.8 to 70.1) |
| **E2 recalibrated, alpha 10%, 0.992** | **71** | **1** | **0.4% (0.0 to 1.1)** | **41.8% (29.2 to 55.3)** |

**Leave-one-out at alpha 10%** (every fold chose about 0.994): the recalibrated threshold accepted 69 inferences with **0 harmful** (0.0% of
unsettled rules), against 5 harmful of 89 at the frozen 0.766 (3.8%) and 6 of 91 when accepting everything (4.0%).

## Reading

On E2 notes the checker's confidence is high even when it is wrong, so the frozen 0.766 accepts almost everything (89 of 91 inferences) and
filters nothing; only a threshold near 1 separates the harmful inferences here. Recalibrating buys about a 10-fold lower harmful rate for
roughly a fifth fewer accepted inferences (91 to 71). That fits the proposal's warning that a threshold does not carry over to patients drawn
differently, but with 6 events it is a hint and not a result. A real E2 recalibration needs the 50-patient calibration half (smallest
certifiable alpha about 2%), expert or hand-checked labels, and the same analysis on a held-out E2 set.
