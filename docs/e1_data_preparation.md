# E1 data preparation status

`trialguide.finetune.e1_data` provides an unlabelled candidate iterator and validation helpers.
It reads the pinned `data/raw/pmc_patients/PMC-Patients.csv` in chunks and yields each retained
record in source order; it does not collect all note text in memory. It requires `patient_uid`,
numeric `PMID`, and non-empty `patient` fields, rejects duplicate record IDs, and excludes every
PMID listed in `splits/e2/e1_exclude_pmids.txt`. A caller may pass a mutable summary dict; counts
are updated during iteration and complete after the iterator is exhausted. Each yielded note
has sentence IDs and carries its PMC V1 source, record ID, and PMID. The development loader
preserves sentence IDs when candidates are supplied, but currently fails closed before opening
any cohort query file.

## Development-note isolation blocker

The TrialGPT source has one `queries.jsonl` per cohort, with development, calibration, and test
records mixed in the same file. The existing `load_queries()` parses every complete JSON line,
including each record's `text`, before `load_patients()` applies split membership. Filtering the
returned `Patient` objects therefore reads sealed note contents transiently. These files have no
split-specific offsets or separate development-only copy available to this loader. Scanning the
mixed JSONL to find development IDs still requires reading each line containing sealed text, so
the E1 loader refuses to open those files. It validates split file hashes, the 37/73/73 counts,
and disjoint IDs first, then raises a clear error before any query file access.

To enable development loading safely, provide a steward-approved development-only source or a
trusted index that permits random access without reading calibration/test note bytes. Until that
exists, PMC iteration is independent of development-note loading, but no TrialGPT development
candidate can be materialized through this E1 path.

## Exclusion-list integrity

The repository does provide authoritative integrity metadata: `splits/e2/e2_info.json` contains
the SHA-256 recorded for `e1_exclude_pmids.txt`, and `trialguide.data.e2._hash` computes the same
newline-normalized digest used by existing E2 integrity tests. The default E1 loader verifies
the frozen list against that committed digest, then checks that it contains exactly 276 unique
numeric PMIDs. Alternate paths are used only by isolated synthetic tests and receive structural
checks; they are not treated as authoritative frozen data. The checksum validates the exact
committed bytes, not how the list was originally derived or whether the upstream pool is current.

## Evaluation overlap blocker

The frozen evaluation files expose SIGIR/TREC IDs (for example, `sigir-...` and
`trec-2021...`); PMC-Patients exposes `patient_uid` and article PMID identifiers. The
repository contains no safe mapping or precomputed fingerprint joining those namespaces.
An exact-ID intersection can be checked without note access, but a zero result does not
establish content-level non-overlap. Computing content fingerprints by reading calibration
or test notes would violate the sealed-note rule, so this preparation code does not do that.

Safe resolution proposal: the data steward could produce a signed, auditable overlap attestation
by matching candidate records against the 183 evaluation records in an approved environment,
without releasing note text or hashes to the training workspace. This attestation is not present
in the repository and no verifiable approval artifact format is defined. Until the steward
provides an approved artifact and its verification format, PMC candidates are not approved for
E1 use. The E1 candidate record's source metadata is provenance information, not proof of source
clearance.

## T23 label blocker

PMC-Patients contains no expert trial-eligibility labels. The TrialGPT criterion annotation
parquet contains the potentially relevant expert labels and GPT-4 explanations, but the
leakage checklist forbids it as model input and requires its `training` column to be ignored.
No other valid source of expert trial labels or independent second-sample agreement labels is
present in this repository. The T23 preflight consequently remains blocked: arbitrary files,
caller-set flags, and caller-constructed ready reports cannot establish approval. Therefore no
teacher labels, T23 examples, agreement filter, or training dataset are produced here. Establish
an approved reference-label source, second-sample source and procedure, and verifiable approval
artifact format before implementing those steps.
