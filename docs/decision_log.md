# Decision log

## T18: visible-note retrieval and shared corpus

### 1. DEV masking protocol and retrieval boundary

- **Date and owner:** 2026-11-14; Vidur.
- **Options considered:** Random, rule-targeted, evidence-targeted, or unmasked notes; accepting a full `Patient` or accepting only `VisibleNote`.
- **Choice:** The primary DEV evaluation uses `mask_random(patient, 0.4, base_seed=0).visible`. Retrieval accepts only `VisibleNote`. Evaluation is DEV-only.
- **Rationale:** Random 40% masking is the project's main-run setting and does not select visible facts using expert labels or patient-specific candidate buckets. The masking module keeps hidden sentence IDs separately from the visible object.
- **Trade-offs:** A deterministic random mask may not match rule-targeted B6 conditions. The 40% setting and seed define one reproducible protocol, not every possible note visibility pattern.
- **Falsifying evidence:** A project-approved T18 protocol that specifies a different primary DEV mask, or DEV-only evidence that this mask fails to represent the intended retrieval setting.

### 2. VisibleNote query formatting and truncation

- **Date and owner:** 2026-11-14; Vidur.
- **Options considered:** Keep sentence-number prefixes from `VisibleNote.text()` or remove them; query limit 64 or 256 tokens.
- **Choice:** Build one MedCPT query from visible sentence text in sentence-ID order, without sentence-number prefixes, with a 256-token limit. Compare the 64-token variant on DEV before treating the setting as final.
- **Rationale:** Only visible sentence contents belong in the query. The 256-token limit follows the pinned TrialGPT MedCPT retrieval implementation; the 64-token comparison reflects the MedCPT usage example.
- **Trade-offs:** Removing prefixes differs from the current `VisibleNote.text()` formatting. A 256-token query differs from the shorter MedCPT example and may truncate the latter part of a long note.
- **Falsifying evidence:** A DEV-only comparison showing a materially better and reproducible result for 64 tokens or another explicitly evaluated formatting. Patient IDs, hidden sentences, cached GPT-generated queries, full notes, and expert labels remain prohibited retrieval inputs.

### 3. MedCPT article fields

- **Date and owner:** 2026-11-14; Vidur.
- **Options considered:** Top-level `title` and `text`; `title` and `metadata.brief_summary`; adding eligibility criteria or other metadata to the pair.
- **Choice:** Encode each trial as `[trial["title"], trial["text"]]`.
- **Rationale:** This is the paired article input used by the pinned TrialGPT MedCPT retriever and matches MedCPT's title/body input convention. `text` and `metadata.brief_summary` are distinct corpus fields.
- **Trade-offs:** Eligibility criteria and structured metadata are excluded from the MedCPT pair even though they may contain useful retrieval signals. The current decision does not prevent the lexical retriever from using its separately defined fields.
- **Falsifying evidence:** Pinned upstream documentation or a DEV-only comparison supporting a different article mapping.

### 4. Cohort-specific shared corpus

- **Date and owner:** 2026-11-14; Vidur.
- **Options considered:** Search the matching cohort corpus, merge all cohort corpora, or use patient-specific `retrieved_trials.json` entries.
- **Choice:** Load `data/raw/trialgpt/{sigir,trec_2021,trec_2022}/corpus.jsonl` for the matching patient's cohort. Never use `retrieved_trials.json` as the shared candidate corpus.
- **Rationale:** The data manifest supplies a separate shared corpus for each cohort. Patient-specific retrieved lists are grouped by expert label and would leak candidate selection and labels.
- **Trade-offs:** Cohort-specific runs use different corpus sizes and do not directly represent one pooled search index. Cohort metadata is used only for routing, never as query text.
- **Falsifying evidence:** An approved T18 evaluation definition that explicitly requires a pooled corpus, with an accompanying protocol for comparing its results to cohort-specific search.

### 5. Corpus record validation

- **Date and owner:** 2026-11-14; Vidur.
- **Options considered:** Fail on invalid records, silently skip them, or substitute empty values for all missing fields.
- **Choice:** Fail clearly on malformed JSON, missing or invalid `_id`, duplicate IDs, and missing or invalid `title` or `text`. Optional metadata may be absent and is preserved when present.
- **Rationale:** Silent skipping or substitution can break the alignment between trial records and retrieval scores. The inspected cohort corpora share the required fields, and metadata is not required by the loader contract.
- **Trade-offs:** A malformed required record stops loading until the corpus is corrected or downloaded again. Optional metadata contents remain the responsibility of downstream methods that use them.
- **Falsifying evidence:** A pinned upstream corpus contract documenting intentional invalid required fields or a required normalization behavior.

### 6. Phase 2 lexical candidates, fusion, and tokenization

- **Date and owner:** 2026-11-14; Vidur.
- **Options considered:** Include zero-score lexical records or omit them; fuse only shared candidates or the union of method candidates; use unoffset reciprocal ranks or an offset RRF; change the lexical tokenizer or retain it.
- **Choice:** Keyword and BM25 result lists contain only positive-score matches. Fusion ranks the union of candidates returned by each method; a future dense MedCPT ranking contributes its own scored candidates. Use equal-weight RRF, `sum(1 / (20 + rank))`, with rank starting at 1. If BM25 is unavailable, keyword fallback does not contribute a second vote. Retain the current tokenizer for this phase: lowercase `[a-z][a-z0-9]+` matches and remove the existing stopword set.
- **Rationale:** Zero-score lexical records have no lexical evidence and would otherwise receive arbitrary tie-broken ranks. Candidate union lets distinct methods contribute candidates. The offset reduces the influence of a single high rank, and ID tie-breaking keeps ordering deterministic. Keeping tokenization unchanged isolates this phase's candidate and fusion changes.
- **Trade-offs:** Lexical-only retrieval can miss relevant trials with no matching tokens; those trials enter the union only if another method retrieves them. The fixed offset and equal weights are a chosen baseline, not tuned parameters. The tokenizer drops one-character terms and tokens beginning with digits, which may omit clinically meaningful abbreviations or numeric values.
- **Falsifying evidence:** DEV-only comparisons showing materially better eligible-trial recall or ranking with zero-score lexical candidates, another fusion offset/weighting, or a tokenizer change. Such comparisons must not use test patients or labels.

### 7. MedCPT models, revisions, and embedding method

- **Date and owner:** 2026-11-14; Vidur.
- **Options considered:** Floating model revisions or immutable revisions; pooled or normalized vectors; re-encode articles per query or cache corpus vectors.
- **Choice:** Use `ncbi/MedCPT-Query-Encoder` at revision `1aa262155a9128ad7d10a64fa782aeefc751f93d` and `ncbi/MedCPT-Article-Encoder` at revision `9de8685cd4789a21e222b44831b75dc78bf2e039`. Extract the final-layer CLS vectors and score with an unnormalized dot product. Encode article pairs as `[title, text]`; query limit is 256 tokens and article limit is 512 tokens.
- **Rationale:** The revisions pin the model artifacts used for reproducibility, and CLS inner-product scoring follows MedCPT's retriever convention.
- **Trade-offs:** A future model revision requires a new cache. The 256-token query is a starting DEV protocol and may truncate long visible notes. Scores are not calibrated probabilities.
- **Falsifying evidence:** Official model documentation for these revisions that specifies a different input/scoring convention, or a reproducible DEV-only comparison supporting another query limit or scoring rule.

### 8. Reusable MedCPT corpus index and runtime

- **Date and owner:** 2026-11-14; Vidur.
- **Options considered:** Recompute articles on each query, persist NumPy vectors with validation metadata, or use pickle/model-specific serialized objects; require torch at package import or load it only for inference.
- **Choice:** Provide a cohort-specific reusable index using float32 NPY vectors plus JSON metadata, with corpus fingerprint, ordered trial IDs, model revisions, tokenizer configuration, dimensions, dtype, and array checksum. Write artifacts atomically and load them without importing model libraries. Import torch/transformers lazily only when an encoder is needed; support CPU and CUDA, with fp16 only on CUDA.
- **Rationale:** Shared trial articles are stable across queries, so an ID-aligned cache avoids repeated article encoding. NPY with `allow_pickle=False` and JSON metadata provides simple validation without adding dependencies to ordinary lexical retrieval imports.
- **Trade-offs:** Each cohort and configuration needs its own cache; cache construction needs substantial memory and may require smaller batches on limited GPUs. Concurrent builders targeting one directory are not coordinated.
- **Falsifying evidence:** A cache mismatch or corruption that passes validation, or measurements showing that index reuse does not reduce article-encoding work while maintaining exact ID alignment.

### 9. Indexed scoring integration

- **Date and owner:** 2026-11-14; Vidur.
- **Options considered:** Change `retrieve_trials()` arguments/result schema, pass raw records through a dedicated indexed scorer hook, or re-encode article documents for every query.
- **Choice:** Preserve the public retrieval signature and result schema. The optional `MedCPTIndexScorer` validates corpus identity and maps vectors by trial ID; the existing scorer hook remains available. No-index default scoring keeps a bounded in-process article cache for compatibility.
- **Rationale:** This reuses the corpus index while retaining shared retrieval interfaces and deterministic ID alignment.
- **Trade-offs:** The index scorer needs the exact same corpus records and config used at build time. The legacy no-index path's bounded memory cache is process-local and does not replace the persistent index for reusable runs.
- **Falsifying evidence:** Integration tests showing changed lexical results/API behavior, incorrect candidate-to-vector alignment, or repeated article encoding while using a loaded index.

### 10. T18 DEV retrieval evaluation

- **Date and owner:** 2026-11-14; Vidur.
- **Options considered:** Evaluate all frozen splits or DEV only; expose patient-level traces or aggregate metrics; rebuild article vectors per patient or reuse one cohort index; report eligible recall against all clinical eligibility or only benchmark qrels.
- **Choice:** Load only `Split.DEV` patients and grades, apply `mask_random(patient, 0.4, base_seed=0)`, route each visible note to its cohort's shared corpus, and load/build each cohort's MedCPT index once per run. Report macro NDCG@10 and graded P@10 through `per_patient_metrics()`, plus macro Recall@10 over grade-2 qrel trial IDs for patients with at least one such qrel. Outputs are aggregate-only.
- **Rationale:** This follows the approved masking and shared-corpus contracts, reuses the existing ranking API, and makes the recall denominator explicit. Recall is a measure over benchmark-labelled grade-2 trials, not all clinically eligible trials.
- **Trade-offs:** Existing `per_patient_metrics()` ranks labelled trials only, while Recall@10 counts grade-2 qrels in the system's actual returned top ten. This distinction follows the current metric and requested recall definitions. The current `retrieve_trials()` RRF contributes keyword, BM25, and MedCPT votes; it cannot express a two-vote BM25 + MedCPT RRF without a public API change or new fusion implementation, so the runner reports the actual three-vote system as `Keyword + BM25 + MedCPT RRF`.
- **Falsifying evidence:** A reviewed ranking-metric protocol requiring non-qrel corpus candidates to occupy positions in graded P@10, or an approved retrieval interface that selects BM25 + MedCPT as a two-system RRF while preserving the current fusion behavior for existing callers.

### 11. Strict DEV-only retrieval evaluation data readers

- **Date and owner:** 2026-11-14; Vidur.
- **Options considered:** Reuse the shared split loaders, which materialize all cohort rows before filtering; or add narrow retrieval-evaluation readers that consult only `splits/dev.txt` and parse payload fields only for DEV IDs.
- **Choice:** The T18 runner uses `load_dev_patients()` and `load_dev_gold_grades()` from `trialguide.data.dev_only`. They read the DEV ID list directly, scan the combined cohort files for record IDs, and fully parse note/qrel payloads only for DEV IDs. Existing shared loaders remain unchanged. The runner has no split selection option.
- **Rationale:** The split source is one ID per whitespace-separated entry, while cohort queries and qrels are combined files. A DEV ID allowlist provides a stable selection boundary without changing other tasks' loader behavior.
- **Trade-offs:** Combined files must still be scanned sequentially, so bytes belonging to non-DEV lines are read to locate their IDs. The note reader only decodes top-level keys and `_id` for those lines; the qrel reader reads only the first TSV field. A malformed line that prevents identifying its patient ID fails closed. Absolute byte-level isolation needs upstream split-specific files or a trusted index.
- **Falsifying evidence:** An upstream source format with separate per-split note/qrel files, or synthetic malformed non-DEV payloads being parsed or materialized by the DEV readers.

### 12. Separate BM25 + MedCPT fusion and fallback semantics

- **Date and owner:** 2026-11-14; Vidur.
- **Options considered:** Change `retrieve_trials()` to select a fusion set, duplicate RRF logic in the runner, or add a separate retrieval entry point that shares the internal fusion helper.
- **Choice:** Use the section 6 equal-weight offset RRF rule, `sum(1 / (20 + rank))`, for all T18 fusion systems. `retrieve_trials()` provides Keyword + BM25 and optional MedCPT fusion; `bm25_medcpt_rrf_retrieve()` provides the BM25 + MedCPT-only system. Both use candidate union and deterministic trial-ID tie-breaking. The DEV runner also evaluates Keyword + BM25 RRF as a separate system. If `rank-bm25` is unavailable, the two-method entry point raises a clear error instead of counting keyword fallback as BM25.
- **Rationale:** Applying one fixed fusion rule makes the T18 fusion systems directly comparable and follows the pre-specified offset-20 decision in section 6. The separate two-method entry point makes the BM25 + MedCPT comparison explicit without duplicating fusion logic.
- **Trade-offs:** Applying offset-20 RRF changes the ranking scores and potentially the ordering of the existing `retrieve_trials()` fusion relative to the previous `1 / rank` implementation. This is an intentional T18 protocol change, not a backward-compatible ranking-preservation change. The two-way path cannot produce a valid independent BM25 vote without `rank-bm25` and therefore stops clearly.
- **Falsifying evidence:** Regression evidence that the shared helper changes any existing `retrieve_trials()` ordering, scores, candidate union, or method metadata; or an approved BM25 backend that avoids keyword substitution while remaining dependency-free.
