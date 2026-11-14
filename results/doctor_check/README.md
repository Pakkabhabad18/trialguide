# Check of the simulated doctor (D10), 14 Nov 2026

Code: `src/trialguide/eval/doctor_check.py`. Files: `doctor_check_sheet.csv` (the filled sheet),
`doctor_check_key.json` (the simulated doctor's answers), `doctor_check_scores.json` (scores, per item).

**What was checked.** 60 questions that the systems asked the simulated doctor (Llama-3.1-8B-Instruct) in the
60% development run, sampled half from questions the doctor answered and half from ones it called "not
documented" (seed 0). For each, the sheet showed only the question and the patient's full record with numbered
sentences, never the doctor's answer.

**Who answered (stated as it was done).** The answers were drafted by Claude (Anthropic), working only from the
records and without seeing the doctor's answers, and then reviewed and corrected by a team member (Murthy). This
is a check by an independent model from a different family than the doctor, reviewed by a person; it is **not** a
fully human blind check. Rule used: answer only from what a sentence states; "otherwise healthy"-type statements
count for disease-history questions but not for treatments or measurements; leave blank when the record does not
say.

## Scores

| | Result |
|---|---|
| Agreement on "does the record answer this?" (60 questions) | 46 / 60 (76.7%), Cohen's κ = 0.53 |
| Doctor answered, checker says the record does not say | **0** of 16 |
| Doctor said "not documented", checker found an answer | 14 of 30 |
| Both answered: same sentence cited | 26 of 30 (87%) |
| Both answered: same answer (automatic text match) | 22 of 30 (73%) |

## Reading the disagreements

We read each disagreement (`doctor_check_scores.json`, `per_item`):

- **The doctor never invents.** On none of the 16 questions the checker left blank did the doctor give an answer.
  This is what the quote check (D10) is for.
- **Of the 14 "not documented" answers the checker disagrees with, 10 hinge on absence or on hedged wording.**
  Seven are "otherwise healthy" / "none stated" for a history question (items 3, 7, 31, 41, 48, 52, 53); the
  doctor's rule does not take "otherwise healthy" as evidence of "no cancer", while the checker's rule does. Three
  more are answers the checker itself hedges ("no allergy diagnosis stated", "cause not stated", "not called a
  COPD exacerbation"; items 4, 44, 19).
- **4 are real misses**, where the record states a specific fact and the doctor did not find it: urinary function
  values (item 22), septal hypertrophy as a myocardial abnormality (33), shortness of breath preventing outpatient
  living (37), and a negative urine culture for "urinary tract infection" (47). That is 4 of the doctor's 30
  "not documented" answers in this sample (13%).
- **When both answered, 27 of 30 agree in content.** The automatic match misses rewordings (e.g. "severe aortic
  stenosis" vs "critical aortic stenosis", items 21 and 54). The 3 real disagreements: item 12 (doctor "unsure",
  checker "no other conditions"), item 32 (doctor "hiatal hernia", checker "none"; the record does show a hiatal
  hernia, so the doctor's answer is arguably the better one), item 40 (signs of sepsis: doctor "tender abdomen",
  checker "heart rate 115 and leukocytosis").

**What it means for the results.** Most of the doctor's "not documented" answers are genuine: the records do not
contain the fact, or contain only a general "otherwise healthy". Some questions (about 13% of "not documented" in
this sample) could have been answered, so the doctor is somewhat conservative, which lowers what asking can
recover. It does not invent answers, so it does not inflate the system's results. Because the sample is half
"answered" and half "not documented", the agreement figures describe these 60 questions, not the doctor's overall
answer mix.
