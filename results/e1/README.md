# E1 (RQ3): QLoRA-tuned vs untuned Qwen3-8B as the rule checker (15 Nov 2026)

Run by Vidur on Kaggle (T4), with `src/trialguide/finetune/` (data, training, comparison).
Files: `e1_compare.json` (all comparison numbers), `untuned/` and `tuned/` (`calibration.json`, `risk_test.json`),
`train_log.json` (loss per step).

## Setup

- **Training data** (development patients only): the input is the rule and the rule-masked note (40%), with exactly
  the checker's prompt; the target is the checker's own answer on the full note when its evidence sentence is still
  visible, or "not enough information" plus the missing fact when that sentence is hidden. 1,702 training examples
  (700 visible, 700 hidden, 300 unresolved, 2 not applicable) and 150 validation examples from 4 held-out
  development patients. These are model-made labels, not expert labels.
- **Training:** QLoRA on Qwen3-8B, 4-bit NF4, LoRA rank 16 on all attention and MLP projections (43.6 M trainable
  parameters, 0.53%), one epoch (213 steps, effective batch 8), about 99 minutes on one T4. Validation loss fell from
  0.114 (step 50) to 0.091 (end).
- **Evaluation:** both models in 4-bit, tools off, greedy, on the same rule-masked notes, against the **expert**
  rule labels of the 26 calibration patients (501 rules) and the 26 test patients (496 rules). No calibration or test
  patient was used in training. Each model gets its own λ̂ from its own calibration run.

## Results

Differences are tuned minus untuned with patient-level paired bootstrap 95% CIs; McNemar is exact on rule-level
agreement.

| Metric (against expert labels) | Calibration: untuned → tuned | Test: untuned → tuned | Test difference [95% CI] |
|---|---|---|---|
| Agreement with the expert status | 32.3% → 40.1% | 31.9% → 36.3% | **+4.4 points [+0.2, +9.0]** |
| McNemar: rules only one model gets right | tuned 54, untuned 15 (p < 0.001) | tuned 46, untuned 24 (p = 0.012) | |
| Rules decided (met / not met) | 25.3% → 20.8% | 20.0% → 16.1% | −3.8 points [−9.7, +1.9] |
| Accuracy of the decided rules | 47.2% → 70.2% | 46.5% → 63.7% | |
| Guesses when the expert's evidence is hidden | 20.3% → 12.7% (79 rules) | 17.6% → 8.2% (85 rules) | |
| Decided wrongly toward eligibility (any category) | 3.2% → 2.6% | 3.0% → 4.0% | +1.0 points [−1.6, +3.9] |

Risk control on test, each model at its own λ̂ (inferences toward eligibility only; harmful = wrong and toward
eligibility):

| Model | λ̂ | Accepted inferences | Harmful | Harmful rate [95% CI] | Inference coverage |
|---|---|---|---|---|---|
| Untuned, accept every inference (B2) | 0 | 61 | 14 | 2.81% [1.15, 4.82] | 13.3% |
| Untuned, risk-controlled (B7) | 0.998 | 37 | 2 | 0.59% [0.00, 1.56] | 8.1% |
| Tuned, accept every inference (B2) | 0 | 15 | 3 | 0.77% [0.00, 1.73] | 3.5% |
| Tuned, risk-controlled (B7) | 0.543 | 13 | 2 | 0.58% [0.00, 1.54] | 3.0% |

## Reading

- **The tuned checker agrees with the experts more often**, on both splits (+7.8 points on calibration, +4.4 on test;
  the test CI only just excludes 0). It gets there by deciding less often but being right more often when it does
  decide (64% vs 47% on test).
- **It learned the behaviour the data targeted:** when the sentence the experts relied on is hidden, it guesses half
  as often (8.2% vs 17.6% on test) and says "not enough information" instead.
- **It is not safer on its own.** Rules decided wrongly toward eligibility went from 3.0% to 4.0% on test (CI
  includes 0). These include "stated" answers, which risk control does not filter.
- **Under risk control both reach about 0.6%** harmful-inference rate, well under the 5% target, but the tuned model
  accepts far fewer inferences (coverage 3.0% vs 8.1%): it rarely infers at all, so there is little left to control.
- **λ̂ differs a lot between the models** (0.998 vs 0.543): the tuned model's confidence values are distributed
  differently. This is why λ̂ is fitted per checker and never carried over.
- **RQ3 answer:** fine-tuning gives a modest but measurable gain in agreement with the experts and halves guessing on
  hidden evidence, without a measurable change in harmful inferences, mostly by making the checker more conservative.

## Limits

- The teacher is the same base model reading the full note, so the tuned model learns to copy that model where the
  evidence is visible, including its mistakes. The labels are model-made.
- Both models run in 4-bit with tools off, unlike the main system (8-bit, tools on), so these numbers are not
  comparable with `results/test_run/`; only the tuned-vs-untuned difference is the result.
- Agreement is low in absolute terms for both, because the experts settle many rules from sentences the mask hides.
- One training run and one seed; 26 patients per split, so the intervals are wide.
