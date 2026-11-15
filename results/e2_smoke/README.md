# E2 smoke run (T38, exploratory): the frozen system on 10 Indian-affiliated case reports

**This is not the E2 evaluation of the plan.** T38 asks for the full system on the 100 E2 patients with recalibration. That did
not fit the time and compute left, so this is a **small exploratory check** that the frozen system runs on this kind of
note and a first look at how it behaves. What is missing: recalibration, B7 and the doctor, any gold labels, and the E2
test half (never opened). The primary results in the README are unchanged.

## What was run

- **Patients:** 10, a seeded sample (seed 20261115) of the E2 **calibration** half (`chosen.json`). Adults, one patient per article
  (`docs/e2_limits.md`).
- **Candidate trials:** the top 2 of the 157 E2 trials per patient by BM25 on the full note, so the full and the masked note are
  checked against the **same** 355 rules (23 to 98 per patient). Retrieval is not what is evaluated.
- **System:** the frozen code (commit `55be3fe`), Qwen3-8B 8-bit greedy, on Kaggle (2 x T4, 78 minutes). Each rule is checked once on the full note and
  once on the rule-targeted note with 40% of the sentences hidden; B6, B2 and B5 (threshold 0.9) are applied to the same checks.
  No doctor, so no B7 or B4.
- **Reference:** E2 has no expert labels, so the reference is the checker's own judgement on the full note (model-derived,
  like the T45 reference in `results/reference_audit/`). "Recovery" means the same as in `results/test_run/recovery.json`: of the rules the
  full-note checker settles and masking made B6 leave open, the share a policy settles the same way, wrongly, and wrongly
  toward eligibility.

## Read this before the numbers

1. **The reference is the checker itself.** Where it infers "not met" from a note's silence on both the full and the masked
   note, the two agree and it counts as recovered. So "recovered" partly measures how consistent the checker is with itself, not
   whether it is right. On the full E2 notes the checker settled 77.5% of rules and **30% of all rules by inference**
   (102 "inferred, not met"), against 16% on the TREC/SIGIR test notes.
2. **The candidate trials are weak.** The 10 patients were matched to only **9 different trials**; one of them (NCT04057209) is a
   candidate for 5 of the 10 patients. Many rules are therefore trivially "not met" (253 of 355 on the full note), unlike the expert-labelled
   trials of the main benchmark.
3. **10 patients, no calibration.** The intervals are wide, and the 0.9 threshold of B5 was not tuned for E2. Nothing here is a
   risk guarantee, and the E2 licence and representativeness limits in `docs/e2_limits.md` apply.

## Results (10 patients, 355 rules, 40% masking)

| | Full note | Masked (40%) |
|---|---|---|
| Settled by the checker (met or not met) | 275 rules (77.5%) | B6 leaves **69.0%** unresolved |
| Stated / inferred / missing | 168 / 108 / 79 | 111 / 96 / 148 |

Rules lost to masking (settled on the full note, left open by B6): **170 of 275**.

| Policy | Recovered | Settled wrongly | Wrongly toward eligibility |
|---|---|---|---|
| B2, every inference | **48.8%** (32.8 to 64.9) | 4.7% (2.4 to 7.8) | 3.5% (1.4 to 6.9) |
| B5, threshold 0.9 | 47.6% (32.3 to 63.1) | 4.1% (2.0 to 7.6) | 2.9% (0.7 to 6.8) |

For comparison, the main test run (73 TREC/SIGIR patients, same definitions, expert-labelled trials): B2 recovers 21.4% (17.7 to 25.5), settles 6.4%
wrongly and 2.2% (1.5 to 3.0) wrongly toward eligibility; B5 18.7% (15.3 to 22.4), 5.7% and 1.5%.

**Reading:** the system runs on Indian-affiliated case reports without problems (both parts exited 0 and no check reported a parse failure). Recovery by inference
is about twice as high as on the main benchmark, and its error rates are of a similar size, but given the first two points above that gap says
more about the longer notes, the weak candidate trials and the self-referential reference than about performance on Indian patients. Inference
recovers rules at a cost of a few percent settled wrongly toward eligibility, as on the main run. A calibrated E2 result would need the
full 50-patient calibration half, expert or hand-checked labels, and B7 with the doctor.

## Files

`summary.json` (all numbers above), `chosen.json` (patients and trials), `rule_table.csv` (every rule: status, category, confidence, evidence ids and each
policy's final status, **without** the model's explanations), `run_info.json`. The raw checks with explanations paraphrase
PMC-Patients text (CC BY-NC-SA 4.0), so they are not committed; they are in the Kaggle notebook output (`trialguide-t38-e2-smoke`).
Reproduce the summary from the two Kaggle parts with `python -m trialguide.eval.e2_smoke --merge <part a> <part b> --out <dir>`; the Kaggle
script is `scripts/kaggle/t38_e2_smoke/`.
