# Decision log

One entry per design choice: date, who decided, options considered, what we chose, why, and
what result would prove us wrong. Written when the choice is made, not afterwards.

Murthy's entries are numbered M1, M2, ... so they do not collide with entries added by
others in parallel (the team-wide numbering, e.g. D13, is Prakash's). Append only; if a
decision changes, add a new entry that points back to the old one.

---

## M1. Pin upstream data to one commit / revision, verify with SHA-256 (T3)
- **Date / who:** 2026-10-27, Murthy
- **Options:** (a) download whatever `main` currently has; (b) pin versions but trust the download; (c) pin versions and commit a checksum for every file.
- **Chose:** (c). TrialGPT files are fetched at commit `1b3242c`, the criterion annotations at Hugging Face revision `1cfcafd`. `src/trialguide/data/checksums.json` holds the SHA-256 of every file and the downloader fails on a mismatch.
- **Why:** four people run on four Kaggle accounts and must see identical patients, trials and labels. A silent upstream edit, a truncated download or a different copy would change splits and results with no warning, and we could not explain it in the viva.
- **Proved wrong if:** upstream legitimately fixes labels we want (then we bump the pin in a reviewed PR and log it), or checksum failures turn out to be flaky (e.g. line-ending conversion on Windows; we read in binary, so none expected).

## M2. Fetch `retrieved_trials.json` and corpora, but not `trial_info.json` by default (T3)
- **Date / who:** 2026-10-27, Murthy
- **Options:** (a) download everything TrialGPT lists, including the 1.2 GB `trial_info.json`; (b) download only what the pipeline reads.
- **Chose:** (b). `trial_info.json` is behind `--with-trial-info`. Core download is about 0.4 GB.
- **Why:** Prateek's T4 baseline already reads trial text from `retrieved_trials.json` and reports that the 1.2 GB file is very slow on our cluster. Kaggle sessions have limited disk and time. Opt-in keeps it available if retrieval (T18) turns out to need it.
- **Proved wrong if:** T18 or the checker needs fields only in `trial_info.json`; then we turn the flag on in the scripts.

## M3. Downloader is stdlib-only Python behind a thin shell script (T3)
- **Date / who:** 2026-10-27, Murthy
- **Options:** (a) a bash script with `wget`/`curl`, as the TrialGPT README suggests; (b) Python module plus `scripts/download_data.sh` wrapper.
- **Chose:** (b).
- **Why:** the team works on Windows laptops, the IIIT cluster and Kaggle; `wget` is not on Windows and shell quoting differs. Python gives identical behaviour everywhere, writes to a `.part` file so an interrupted download is never mistaken for a complete one, can be unit-tested offline, and needs no extra pip package.
- **Proved wrong if:** a teammate's environment cannot run it (Python 3.10+ is already required by the package).

## M4. A patient is "usable" only if it has a note and expert trial labels (T7)
- **Date / who:** 2026-10-27, Murthy
- **Finding:** SIGIR `queries.jsonl` lists 59 patients but `sigir-201428` has no trial labels in the qrels, so **58 usable**, which matches the proposal (75 + 50 + 58 = 183). The annotation file has **1,015** rows (the Hugging Face card says 1,015; the proposal's 1,020 is a typo). It also has 18 rows for `sigir-201426`, a patient that appears in no cohort file (no note in `queries.jsonl`, no trial labels).
- **Options:** (a) keep all 59 / all annotated patients; (b) drop patients lacking a note or trial labels.
- **Chose:** (b). `sigir-201428` and `sigir-201426` are excluded from the split (T8) and from evaluation.
- **Why:** nothing can be ranked or scored for a patient without trial labels, and a patient without a cohort note cannot be masked or shown to the system. The 18 orphan rows still carry the note text in the annotation file, so they stay available for a rule-level check but not for ranking.
- **Proved wrong if:** the upstream authors publish the missing files for these two patients.

## M5. "Recoverable NEI" is defined from the annotation file as GPT-4 NEI with a definite expert label (T7)
- **Date / who:** 2026-10-27, Murthy
- **Options:** (a) reuse the paper's 30.7% (8 of 26 errors, a manual judgement); (b) define it mechanically on all 1,015 rows; (c) hand-label every NEI row.
- **Chose:** (b). A row is *recoverable* when GPT-4 answered "not enough information" and the expert gave `included`, `not included`, `excluded` or `not excluded`. Rows where the expert also says NEI are *truly unresolved*, which is where asking or abstaining belongs.
- **Result** (`results/label_audit.md`): 293 rows (28.9%) are expert NEI; **34 rows (3.3%)** are recoverable (17 inclusion, 17 exclusion), of which 21 have expert-cited evidence sentences. They are 26.4% of the 129 GPT-4/expert disagreements, close to the paper's 30.7%. GPT-4 decided where the expert says NEI on 21 rows, the over-commitment the risk control is for.
- **Why:** it is reproducible and uses all rows. It is a lower bound: GPT-4 may have resolved a rule the expert could only settle by inference, and "not applicable" is not counted as recoverable.
- **Proved wrong if:** the 200-rule hand audit (T15 and later) finds many "recoverable" rows where a careful reader cannot infer the answer from the note.
- **Consequence for the project:** only about 3% of expert-labelled rows are inference-worthy. Rule-targeted masking, not natural NEI, has to supply most of the infer/ask test cases, as the proposal already assumes.

## M6. Expert rule labels exist only for SIGIR; this limits how calibration can work (T7, raised to the team)
- **Date / who:** 2026-10-27, Murthy, **needs a team decision before T8**
- **Finding:** all rule-level labels are on SIGIR patients: 52 usable patients, 997 rows. TREC 2021 and 2022 (125 patients) have none.
- **Why it matters:** the proposal keeps expert-labelled patients in the calibration and test splits. With 52 labelled patients that is about 26 per split (roughly 500 rows each) and no labelled patient in development. Primary harmful-inference claims use expert labels (Section on calibration), so the 5% risk threshold would be fitted on SIGIR only, and TREC would rely on model-derived references.
- **Options for T8:** (a) as the proposal says, all 52 labelled patients go to cal and test, none to dev; (b) put some labelled patients in dev so prompts can be checked against expert rule labels, at the cost of fewer in cal and test.
- **Status:** resolved provisionally by M7 on 2026-10-30 (option a, the proposal's rule), so the team is not blocked. Prakash (owner of risk control) can still overrule before the split PR is merged.
- **Not used:** the `training` flag in the annotation file is a per-row boolean mixed within patients (881 true, 134 false) and undocumented, so it is ignored.

## M7. Expert-labelled patients go to calibration and test only; development has none (T8)
- **Date / who:** 2026-10-30, Murthy. Provisional until Prakash confirms (see M6).
- **Options:** (a) all 52 labelled patients in cal and test, as the proposal states; (b) put some in dev.
- **Chose:** (a), 26 in cal and 26 in test, 0 in dev.
- **Why:** it is the rule already written in the proposal (Section 6), so no deviation needs justifying, and it keeps the gold evaluation clean: nothing tuned on dev can touch an expert-labelled rule. Teammates were blocked on a frozen split, so I did not wait for the M6 discussion.
- **Cost, stated plainly:** dev has no expert rule labels, so checker accuracy on dev can only be judged against our own annotation (T15 pilot) and the full-note model reference. Cal and test each hold about 500 labelled rules, and only about 3% of rules are inference-worthy (M5), so the 5% risk target may be infeasible on that few events. The proposal already plans for this (report the smallest achievable alpha).
- **Proved wrong if:** the checker work (T12, T28) needs expert rule labels on dev to debug. Then move about 10 labelled patients to dev; this must happen before the split PR is merged, because after that the files never change.

## M8. How the split is drawn: stratified, matched pairs, one pre-chosen seed (T8)
- **Date / who:** 2026-10-30, Murthy
- **Options:** (a) one plain random shuffle; (b) try seeds until cal and test look balanced; (c) stratify by cohort and label coverage, and pair labelled patients by rule count.
- **Chose:** (c), seed 20261027 (the date; chosen before running, not searched).
- **Why:** plain random could put 30 labelled patients in cal and 22 in test, or leave one split without TREC 2022. Searching seeds is quiet cherry-picking that we could not defend in the viva. Pairing the labelled patients by number of rules and flipping a coin inside each pair balances gold evidence without any search.
- **Result** (`splits/split_info.json`): dev 37 (SIGIR 2 / TREC21 21 / TREC22 14); cal 73 (28 / 27 / 18), 26 labelled patients, 501 rule rows; test 73 (28 / 27 / 18), 26 labelled, 496 rule rows. Totals are exactly 37/73/73 over the 183 usable patients (M4).
- **Frozen:** `splits/*.txt` carry LF endings (`.gitattributes`) and their SHA-256 are in `split_info.json`; `tests/data/test_splits.py` fails if any file changes. `--write` refuses to overwrite an existing split.
- **Proved wrong if:** the leakage review finds a cohort or label imbalance we did not consider (e.g. a trial appearing in suspiciously few splits). Fixing it means a new, logged split before anything is run on cal or test.

## M9. Cached keywords and candidate lists are not used for masked runs (T8, leakage review)
- **Date / who:** 2026-10-30, Murthy
- **Finding:** `retrieved_trials.json` stores the full patient note and groups trials by expert label (0/1/2); `id2queries.json` stores keywords GPT-4 wrote from the full note. Both carry information that masking is meant to hide.
- **Chose:** for masked experiments, retrieval (T18) generates keywords from the visible note only and never reads the cached keywords or the label grouping. Reproducing TrialGPT as it was (T4, B6 on the unmasked note) may use them, because there nothing is hidden.
- **Why:** otherwise a hidden fact could still steer which trials are shown, and the masking result would be optimistic.
- **Proved wrong if:** a check shows candidate sets from visible-only keywords barely differ from the cached ones, in which case the cost of regenerating buys little. We still would keep the rule, since it is the safe default.
- **Owner to act:** Vidur (T18). Listed in `docs/leakage_checklist.md`.
