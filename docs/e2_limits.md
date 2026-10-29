# E2 data: what it is and its limits (T24)

E2 evaluates TrialGuide on **Indian-affiliated oncology case reports** matched to **trials recruiting in India**, with recalibration on part of the set (proposal, E2). Built on 2026-10-30 by `python -m trialguide.data.e2 --build`. The frozen lists are in `splits/e2/` and the counts and hashes are in `splits/e2/e2_info.json`.

## How it was built

| Step | Rule | Count |
|---|---|---|
| PubMed query | `India[ad] AND case reports[pt] AND neoplasms[majr]` | 16,085 articles |
| Affiliation re-check | at least one fetched author affiliation contains the word "India" ("Indian Ocean", "Indiana" do not count) | 16,041 |
| In PMC-Patients V1 | source article has a patient summary in PMC-Patients (167,034 summaries, HF revision `28d8836`) | 346 |
| Adults, one per article | 18 or older in the report; lowest-numbered adult patient of each article | **276 (pool)** |
| Sample | seed 20261030, 100 patients, split 50 calibration / 50 test | `e2_cal.txt`, `e2_test.txt` |
| Trials | ClinicalTrials.gov v2: overall status recruiting, cancer condition query, and **at least one Indian site recruiting** | 181 returned, **157 kept** (`e2_trials.txt`) |

- The 24 trials dropped at the last step have Indian sites that are only "not yet recruiting" or closed, so they cannot enrol a patient in India now.
- Trials are converted to the TrialGPT corpus format (`data/raw/e2/corpus.jsonl`), so retrieval, masking and the checker run unchanged.
- **E1 exclusion:** `e1_exclude_pmids.txt` lists all 276 pool articles. E1 teacher labels and QLoRA training (T23, T31) must skip these PMIDs, otherwise E2 would test on training data.
- **Sealed:** `load_e2_patients(Split.TEST)` raises unless `allow_test=True`, like the main test split.

## Limits (to be stated wherever E2 results appear)

1. **Affiliation is not nationality.** An Indian author affiliation does not establish the patient's nationality, ethnicity or where they were treated. Results are "on Indian-affiliated case reports", not "on Indian patients".
2. **Case reports are not routine records.** Published cases over-represent rare and unusual presentations, are written after the fact and often include the outcome. They differ from the synthetic TREC/SIGIR notes and from real referral notes.
3. **"Oncology" is MeSH Neoplasms as a major topic.** This includes some benign tumours (e.g. lymphangioma, fibromatosis) and cancers found incidentally. Titles alone are too unreliable to filter on (only 191 of 346 contain a malignancy word), so no further filter was applied.
4. **No gold labels.** There are no expert patient-trial or rule-level labels for E2. T38 must use a model-derived reference and/or a small human-labelled subset, and say which. Risk results on E2 are therefore not "risk control against expert judgement".
5. **Time mismatch.** The case reports describe patients from past years, while the trials recruit now (2026). Rules such as "newly diagnosed" or "no prior therapy" are judged against the patient's state *in the report*, not today.
6. **Different from the main benchmark.** E2 notes are much longer (median 21 sentences, against 6 to 9 for TREC/SIGIR), so prompts are longer and compute per patient is higher. The calibration halves are not exchangeable with the main calibration split, so the main risk threshold does not carry over to E2. That is why E2 is recalibrated on its own 50-patient half. With n = 50 the smallest certifiable target is 1/51, about 2%.
7. **Small trial pool, no CTRI.** 157 trials, all from ClinicalTrials.gov. The Clinical Trials Registry - India (CTRI) has no public API, so India-only trials registered only there are missing. Most trials are adult-only, which is why children were excluded (71 of 346 articles had a child patient; 13 of 157 trials accept under-18s).
8. **Snapshots.** PubMed and ClinicalTrials.gov change daily. The committed ID lists are the dataset; re-running the queries later gives different counts. Trial texts are as fetched on 2026-10-30, cached in `data/raw/e2/` (not committed, re-fetch by NCT id if needed).
9. **Licence.** PMC-Patients is CC BY-NC-SA 4.0: research use is fine, no commercial use, and any released derivative (e.g. masked E2 notes) must carry the same licence.
