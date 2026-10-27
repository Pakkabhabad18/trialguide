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

## Open observations to resolve in T7 (not decisions yet)
- SIGIR 2016 `queries.jsonl` has **59** patients; the proposal and README say 58 (75 + 50 + 58 = 183). Check for a duplicate or extra patient before T8 fixes the 37/73/73 split.
- The criterion annotation file has **1,015** rows; the proposal says 1,020. Check against the dataset card (rows may be counted differently).
