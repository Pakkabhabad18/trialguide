# Team Zenith / TrialGuide — Decisions, Implementation & Verification Log

**Last updated:** 29 October 2026
**Working branch:** `vidur`
**Owner of this log:** Vidur Shah
**Purpose:** Record project constraints, implementation decisions, verified results, ownership boundaries, PR coordination, known risks, and next actions. This is a local handover/work log; it does not replace the proposal, build tracker, team-wide decision log, or GitHub PR history.

---

## 1. Project direction and constraints

- **Research project:** *Beyond Abstention: Risk-Controlled Inference and Selective Clarification for Clinical Trial Matching* (TrialGuide).
- **Main model:** Qwen3-8B served with vLLM on the IIIT cluster; the planned cluster configuration uses 2 × RTX 3080 Ti. Kaggle is the backup compute option.
- **Compute and cost:** use free compute for the current workflow. Do not introduce paid API dependencies or commit credentials.
- **Simulated doctor:** Llama-3.1-8B-Instruct reads the full record and must quote supporting text; the prototype still needs its real inference path verified.
- **Repository workflow:** work on `vidur`, push to `origin/vidur`, and raise a PR with base `main`. Do not push work directly to `main`. Do not commit or push on the user's behalf.
- **Data safety:** the rule checker must use only the visible patient note. Hidden facts must not enter its prompt or output.
- **Rule semantics:** `MET` means the rule condition is true for the patient. For an inclusion rule, this means the criterion is satisfied; for an exclusion rule, it means the disqualifying condition is present.
- **Information categories:** `stated`, `inferred`, `assumption`, and `missing`. Only evidence-supported `stated`/`inferred` outcomes can be decisive. `assumption` and `missing` remain unresolved.
- **Clarification budget:** at most three questions per patient, as defined by the project design.
- **Risk target:** the intended harmful-inference target is approximately 5%, to be assessed with the approved calibration/evaluation protocol. Passing unit tests does not establish this target.
- **Patient split:** the planned patient-level split is 37 development / 73 calibration / 73 test. Do not inspect calibration/test outcomes for tuning; follow the team-approved split policy and freeze protocol.

## 2. Branch and repository state

Last verified from the user's local terminal on 29 October 2026:

- `git branch --show-current` → `vidur`
- `git log -5 --oneline` showed `HEAD -> vidur`, `origin/vidur`, and local `main` all at `fc20f7d Add README`.
- `git fetch origin` completed.
- `git status -sb` showed `## vidur...origin/vidur`; there were no committed local commits ahead of `origin/vidur`.
- `git log --oneline origin/main..vidur` returned no commits.
- `git diff --cached --check`, `git diff --cached --stat`, and `git diff --cached` returned no output because no changes were staged.
- Remote `origin` is `https://github.com/Pakkabhabad18/trialguide.git`.

At that point, the following were local working-tree changes, not yet committed:
- Modified: `README.md`
- Untracked: `src/trialguide/checker/rule_checker.py`
- Untracked: `src/trialguide/schemas.py`
- Untracked: `src/trialguide/llm/client.py`
- Untracked: `src/trialguide/data/download.py`
- Untracked: `src/trialguide/data/checksums.json`
- Untracked: `tests/test_rule_checker.py`
- Untracked: `tests/test_rule_safety.py`
- Untracked: `tests/test_rule_checker_t12_interface.py`
- Untracked duplicate-looking test files: `tests/rule_checker_t12_interface.py` and `tests/test_rule_checker_t12_interface(1).py`
- Untracked: `decisions.md`, `decisions_updated.md`, `temp.txt`, and `temp/`
- Untracked reference PDFs: `Team Zenith — Build Plan and Tracker.pdf`, `Zenith_4.pdf`

Do not stage everything with `git add .`. Review file ownership and content first. The PDFs are reference material, not T12 source files. Temporary scripts and generated smoke-test outputs should not be included in the PR unless there is an explicit reason to keep a maintained test utility in the repository.

### Test-file duplicate finding

The user's `fc /n` output showed that `tests/rule_checker_t12_interface.py` and `tests/test_rule_checker_t12_interface(1).py` were byte-identical. The other comparison showed that `tests/test_rule_checker_t12_interface.py` differs from the `(1)` copy; the latter appears to contain checker implementation code rather than just tests. This is a strong sign that the `(1)` file may be an accidental copy of `rule_checker.py` under the tests directory. **Do not delete files blindly**; inspect their contents and run tests after deciding which test file is canonical. The desired test naming convention is `tests/test_*.py`.

## 3. T12 Rule Checker v1 — implementation contract

**File:** `src/trialguide/checker/rule_checker.py`
**Status:** local functional prototype; 42 local tests pass as of 29 October 2026. Not yet proven against real model inference or calibrated risk evaluation.

### Implemented behaviour

- Accepts a visible note, a trial rule, a model identifier, and a requested sample count.
- Uses the visible note text and its sentence IDs in the prompt.
- Normalizes supported rule-status aliases, including TrialGPT-style inclusion/exclusion labels.
- Preserves the agreed `MET` semantics: inclusion `MET` means criterion satisfied; exclusion `MET` means the exclusion condition is present.
- Validates cited evidence IDs against the sentence IDs actually visible in the note; invalid IDs are not accepted as evidence.
- Parses the categories `stated`, `inferred`, `assumption`, and `missing`.
- Downgrades `assumption`/`missing` outcomes to `NOT_ENOUGH_INFO`.
- Downgrades `stated`/`inferred` outcomes with no valid visible evidence to unresolved.
- Rejects malformed or unsupported model responses. Failed samples do not become guessed decisions.
- Requests five samples by default. The prompt includes the sample index.
- Aggregates the valid samples conservatively. A tie in status or in the category for the winning status returns an unresolved result.
- Computes the reported agreement fraction as `winning status votes / requested samples`; invalid responses therefore reduce the fraction. This is an implementation choice that should be checked against the agreed evaluation protocol.
- Returns the shared `RuleCheck` fields, including `status`, `category`, `evidence_ids`, `confidence`, `n_samples`, `explanation`, and optional `fact_key`.
- Normalizes `fact_key` into a short snake_case key and uses consensus among samples supporting the winning status/category. Ambiguous/no-key outcomes return `None`; decisive stated/inferred outcomes should not request a missing-fact key.

### Confidence caveat

The `confidence` field is an **agreement fraction**, not a calibrated probability that a clinical rule is correct. Calls use temperature 0 in the current implementation, so repeated calls may produce the same output and should not be described as independent judgments. No 5% risk guarantee follows from this number alone.

### Important integration caveat

The prompt-level sample index differentiates requests at the message level. The shared client caches by backend, model, messages, and parameters in the local copy inspected earlier; because the messages contain the sample index, the cache key should differ across sample indices. This must still be rechecked against the canonical shared client once the relevant client PR is merged. Do not replace teammate-owned shared client code with an older local copy.

## 4. T19 policy compatibility

T19 policy PR was described in the project discussion as consuming the rule check's category, evidence IDs, confidence, and fact key. The local T12 checker now supplies these fields through `RuleCheck`, so the **field-level interface appears compatible**.

That does **not** yet prove end-to-end integration. Before claiming T12/T19 integration is complete, verify:
- T19 imports the same canonical schema version used by T12.
- The policy receives the actual `RuleCheck` object or the expected field representation.
- `fact_key` is available for unresolved rules and is `None` where appropriate.
- `NOT_APPLICABLE` is handled consistently by the policy and evaluation code.
- Doctor answers and re-checks follow T19's expected evidence/category rules.
- A focused integration test covers at least a decisive evidence-backed rule and an unresolved rule with a `fact_key`.

The earlier review of T19 identified a possible edge case: `decide()` may handle `NOT_APPLICABLE` before applying doctor-answer certainty restrictions. Treat this as a **review item to verify against the current PR and tests**, not as a confirmed defect without reproducing it.

## 5. Shared schemas and LLM client ownership

- `src/trialguide/schemas.py` was copied locally from `origin/prakash` for development while the shared schema PR was pending.
- `src/trialguide/llm/client.py` was copied locally from `origin/T4-trialgpt-baseline` and edited for local development.
- These local copies are not proof that the shared PRs have merged or that the copies are the canonical versions.
- Before staging these files, fetch and inspect the current target branch/merged PR state. If the shared files are already merged, use the canonical versions and keep only necessary T12-owned changes. If the client remains owned by another teammate, coordinate rather than overwriting it in the T12 PR.
- Do not include credentials, API keys, tokens, patient data, model weights, or downloaded datasets in commits.
- The project direction is Qwen3-8B on approved free compute. Import success does not prove that a real inference request succeeds.

## 6. Tests and verification performed

### Latest local verification (29 October 2026)

| Check | Result | What it establishes |
|---|---|---|
| `python -m pytest -q` | **42 passed in 0.14s** | All tests discovered in the current local checkout passed. |
| `git diff --check` | No output | No whitespace errors in tracked-file diffs checked by Git; untracked files are not covered by this command. |
| `python temp\run_checker_mock_smoke.py` with `PYTHONPATH=src` | **PASS** | `check_rule` behaved as expected with deterministic mocked JSON responses and synthetic notes/rules. |
| Mocked inclusion case | `met`, `stated`, evidence `[0]`, `unresolved=False` | Synthetic evidence-backed inclusion path passed. |
| Mocked exclusion case | `met`, `stated`, evidence `[1]`, `unresolved=False` | Synthetic exclusion semantics path passed. |
| Mocked missing-information case | `not_enough_info`, `missing`, no evidence, `unresolved=True` | Synthetic unresolved path passed. |
| `git diff --cached` | Empty | No changes were staged at the time of the command. |
| Real Qwen3-8B inference | **Not verified in this session** | No claim that the model server/client path works. |
| GPU execution | **Not verified in this session** | GPU availability/runtime integration remains outstanding. |
| Calibration / 5% harmful-inference target | **Not performed** | No calibrated-risk claim can be made. |

The mocked smoke test used synthetic fixtures and a patched `chat` function. It did not use downloaded patients, a real model, GPU execution, or calibration/test data.

### Demo readiness statement

For the 30 October progress demo, the defensible claim is:

> T12 has a locally tested functional prototype. The local suite passes 42 tests, and a mocked smoke test covers evidence-backed inclusion, evidence-backed exclusion, and unresolved missing information. Real-model/GPU inference, complete T12-to-T19 integration, split-based calibration, and validation of the 5% harmful-inference target remain outstanding.

Avoid saying that “everything except GPU works” until the non-GPU integration paths—especially the canonical shared client and policy integration—have been verified.

## 7. Data and split context

- The project plan specifies TREC Clinical Trials 2021 (75 patients), TREC Clinical Trials 2022 (50), and SIGIR 2016 (58) for 183 planned patients.
- Earlier downloader/audit notes reported 59 SIGIR query patients and 1,015 annotation rows, compared with 58 and 1,020 in proposal text. These discrepancies were marked for T7 reconciliation.
- Earlier audit notes identified an orphan annotation patient `sigir-201426` and a patient `sigir-201428` without trial labels. Use the approved data decision and audit rather than silently assuming the counts are reconciled.
- The rule-level expert labels are concentrated in SIGIR; an earlier note identified 52 usable labelled patients. The decision about how to allocate these labels across development/calibration/test must be made by the risk-control/data owners before the split is frozen.
- Do not open calibration/test labels for tuning. No actual patient-level checker evaluation is claimed by the latest mocked test output.

## 8. PR coordination and review boundaries

- **T12:** Vidur's checker implementation and tests. Intended flow: commit reviewed T12-owned files on `vidur`, push `origin/vidur`, then open a PR with base `main` and compare branch `vidur`. Prakash reviews/merges according to team workflow.
- **T19:** policy owned by Prakash. Structural compatibility should be confirmed with a focused integration test; the existence of matching field names alone is not end-to-end proof.
- **T42/shared schema PR:** earlier notes record that the schema's `MET` semantics match the checker. Confirm current live PR/merge status before relying on a local copied schema. Do not treat a prior comment review as an approval.
- **T3 downloader/data PR:** the earlier review log identified checksum fail-closed behaviour and download-failure tests as items to verify against the latest live diff. Do not assume those concerns are resolved without checking the current PR.
- **T5/shared LLM client PR:** use the canonical merged client when available. Do not commit an old copied client as a substitute for the shared component.
- The planned PR sequence previously recorded was **#3 → #4 → #1 → #2**. Re-check live PR state before assuming this order or any merge status remains current.
- Before approving any PR, inspect its current diff, checks, and unresolved review comments. Do not approve based only on a previous summary.

## 9. Git staging plan for T12

Do not run `git add .`.

1. Inspect the latest remote PR/branch status and compare local shared files against the canonical merged files.
2. Decide which duplicate test file is canonical. The `tests/test_rule_checker_t12_interface(1).py` file appears to be an accidental copy of checker implementation code and should not be included as a test.
3. Keep planning PDFs and temporary outputs out of the T12 source PR.
4. Stage only confirmed T12-owned checker code, the canonical T12 tests, and the reviewed README/decision-log edits. Include shared schemas/client/data-downloader files only if ownership and merge state make that appropriate.
5. Run `git diff --cached --check`, `git diff --cached --stat`, and inspect the full `git diff --cached`.
6. Re-run `python -m pytest -q` after cleanup.
7. Commit only after the staged diff has been reviewed. Push `vidur`, then open the PR to `main`. Never commit/push/merge/approve on Vidur's behalf.

## 10. Open issues and next actions

### Before creating the T12 PR
- [ ] Inspect current remote branches/PRs and determine whether the shared schema and client are already merged.
- [ ] Reconcile local `schemas.py` and `llm/client.py` with canonical versions; don't overwrite teammate-owned changes.
- [ ] Inspect and resolve duplicate/accidental interface-test files; keep one canonical test module.
- [ ] Confirm `decisions.md` is the intended log filename; avoid staging both `decisions.md` and `decisions_updated.md`.
- [ ] Check `temp/` and `temp.txt`; keep temporary smoke outputs out of the PR unless deliberately retained as maintained test fixtures.
- [ ] Run the full test suite after cleanup.
- [ ] Add a T12-to-T19 integration test with the actual shared schema and policy.
- [ ] Verify `NOT_APPLICABLE` semantics in policy/evaluation.
- [ ] Confirm the approved confidence denominator and sample protocol with the evaluation/risk-control owner.
- [ ] Verify the real shared-client request path and model configuration when approved GPU compute is available.

### For the 30 October demo
- [ ] Demonstrate the mocked functional prototype and its three smoke cases.
- [ ] Clearly label synthetic/mock results as such.
- [ ] Present GPU/model inference and full policy integration as pending if not verified by then.
- [ ] Do not claim calibrated risk control or the 5% target without the approved calibration evaluation.

### After T12 is merged
- [ ] Integrate with T19 using the canonical schema/client.
- [ ] Run real inference on the approved compute setup.
- [ ] Follow the frozen patient-split and calibration protocol.
- [ ] Report harmful-inference risk, coverage, abstention, and relevant ranking metrics with the agreed evaluation procedure.

---

## 11. Log update policy

- Distinguish confirmed team decisions, local implementation choices, and unresolved questions.
- Mark a task complete only when a command, test, review, or explicit team decision verifies it.
- Add dated updates when material decisions change; preserve historical context where useful.
- Never include secrets, patient-level data, model weights, or downloaded datasets in this log or a commit.
