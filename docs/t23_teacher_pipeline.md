# T23 teacher pipeline status

`trialguide.finetune.t23_teacher` contains typed provenance and label records, T12-compatible
response parsing, an approval preflight, and deterministic JSONL ordering. It does not invoke a
model or generate labels. If a future inference path is approved, it must call the shared
`trialguide.llm.client`; this module must not add a separate model path.

## Intended model

The intended T23 E1 teacher is **Qwen3-8B**. T31's QLoRA student is also Qwen3-8B, while T11's
simulated doctor is Llama-3.1-8B-Instruct. Those roles are distinct. The actual model served by
the shared client and its immutable revision are unresolved because the client does not expose
verifiable model/revision metadata. No revision is asserted, and no fallback to Qwen3-32B or
Llama is allowed.

## Current state: blocked

Preflight is intentionally fail-closed. The repository does not define a verifiable approval
artifact format and contains no verified approvals for T23. Caller-supplied booleans, arbitrary
files, and constructed `ready=True` reports do not establish readiness. In particular:

- There is no approved candidate-source verifier that binds prompt text to development-only
  records without accessing sealed calibration or test note contents. Prompt construction is
  blocked until that verification path exists.
- PMC-Patients V1 has no approved expert trial-eligibility labels, and content-level overlap with
  evaluation notes has not been cleared by a data steward.
- The annotation Parquet is not a model input or source for selection. It overlaps calibration
  and test patients, and its explanations, predictions, and `training` column are forbidden.
- No approved independent second-sample label source or procedure is available. Therefore an
  agreement gate cannot produce an accepted record.
- Serialization and loading of non-empty teacher-label JSONL remain blocked. Caller-created
  agreement metadata cannot satisfy the gate.

The preflight may inspect split IDs, split metadata, file presence, and the E2 exclusion list; it
does not read calibration or test note contents. It reports the unresolved approvals, reference
source, overlap clearance, candidate verification, independent-sample evidence, and model
revision as blockers. Synthetic tests exercise rejection paths only and do not simulate approval.

## Safe engineering contracts

The intended teacher constant is `Qwen/Qwen3-8B`; `teacher_version` must remain unset until the
shared client exposes verifiable revision metadata. Parsed T12 outputs are bound to the sentence
IDs of the supplied candidate object. Evidence IDs must exist on that candidate. JSON parsing
rejects duplicate object keys, records reject duplicate identities, and canonical ordering is
stable. These checks do not constitute source approval or authorize labels.
