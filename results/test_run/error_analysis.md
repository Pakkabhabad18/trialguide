# Error analysis of the test run (14 Nov 2026)

Code: `src/trialguide/eval/error_analysis.py` (no model calls; reads the saved test-run outputs).
Numbers: `error_analysis.json`.

Reference: the checker on the full note (model-derived), as in the recovery table. A rule is "lost"
if the checker settled it on the full note and B6 leaves it open on the masked note: 3,728 lost rules
across the 73 test patients. The overall outcome matches the recovery table in `README.md`: B7
recovers 844 (22.6%), gets 237 wrong (6.4%), 70 of them toward eligibility (1.9%), and leaves 2,647
open (71%).

## 1. How the rules B7 settled were settled

| Settled by | Rules | Recovered | Wrong toward eligibility | Wrong toward exclusion |
|---|---|---|---|---|
| Doctor's answer | 97 | 86.6% | 1.0% | 12.4% |
| Inference alone | 984 | 77.2% | 7.0% | 15.8% |

Inference does most of the work: it settles ten times as many rules as the answers. It is also where
the harmful errors come from (69 of the 70 wrong-toward-eligibility rules). Answers are more accurate
but rare, because most answers are "not documented" (section 3).

## 2. By rule type

Types come from keywords in the rule text, so this is a rough grouping.

| Rule type | Lost rules | Recovered | Wrong toward eligibility | Still open |
|---|---|---|---|---|
| Age | 332 | 33.7% | 1.2% | 53.9% |
| Lab or vital value | 136 | 25.7% | 2.9% | 66.9% |
| Condition or other | 2,364 | 23.1% | 2.0% | 70.9% |
| Prior treatment | 554 | 21.8% | 2.2% | 72.0% |
| Pregnancy | 126 | 14.3% | 0.8% | 84.1% |
| Consent or compliance | 216 | 6.0% | 0.5% | 91.2% |

| Rule kind | Lost rules | Recovered | Wrong toward eligibility | Still open |
|---|---|---|---|---|
| Inclusion | 1,750 | 33.7% | 2.7% | 56.3% |
| Exclusion | 1,978 | 12.9% | 1.2% | 84.0% |

- Age is recovered most often: a single answer settles every age rule of a patient.
- Lab values carry the highest harmful rate (2.9%): a missing value is the easiest gap to fill with a
  plausible guess.
- Consent and pregnancy rules stay open: the notes rarely say anything about them, and there is
  nothing to infer from.
- Exclusion rules mostly stay open, because "no mention of X" is not evidence that the patient does
  not have X.

## 3. Questions (B7, 194 questions)

- 136 of 194 answers (70%) are "not documented"; only 51 questions settled at least one rule.
- **63 of the 194 questions (32%) were the same question: "What does the patient's record say about
  prior chemotherapy?"** Each bundled 13 rules on average, and 61 of the 63 came back "not documented".
  That single question accounts for 45% of all "not documented" answers. Without it, 75 of the
  remaining 131 questions (57%) are "not documented".
- Cause: the checker prompt shows `"prior_chemotherapy"` as its example `fact_key`, and the checker
  copies it onto unrelated rules (40% of the masked-note checks on test). The selector merges rules
  by `fact_key`, so unrelated rules end up behind one chemotherapy question. Risk control and λ̂ are
  not affected: they do not use `fact_key`. This was found after the test run.

## 4. Case studies

All five come from one patient, `sigir-201410` (a 67-year-old woman with a groin complication after
cardiac catheterization), picked as the first example of each kind in patient order.

1. **Correct inference.** Exclusion rule "Patients without (expected) distal runoff to the index site".
   The visible note says she has a cool right foot and a pulsatile groin mass with loss of distal
   pulses. The checker infers the rule's condition does not hold, matching the full-note answer.
2. **Inference wrong toward eligibility.** Inclusion rule "History of ipsilateral femoral arterial
   puncture within previous three months ...". The sentence about the catheterization is hidden. From
   the groin mass and "the point at which the right femoral artery was entered", the checker infers
   the rule is met; on the full note it said not met. The inference is plausible to a reader, which is
   what makes this kind of error hard to catch, and why permissive inferences go through risk control.
3. **Rule left open.** Exclusion rule "Known allergy to contrast media". The checker says the note
   does not mention an allergy and leaves it open, instead of treating silence as "no allergy".
4. **A question that settled rules.** "What does the patient's record say about age?" The doctor
   quotes the sentence with "67-year-old woman", and both age rules ("less than 18 years", "age 19-90")
   are settled.
5. **A wasted question.** "What does the patient's record say about prior chemotherapy?" merged 26
   rules, among them "Age > 18 years" and "willing and able to comply with follow-up", none about
   chemotherapy. The answer was "not documented", and one of three questions was spent.
