# TrialGuide

**Beyond Abstention: Risk-Controlled Inference and Selective Clarification for Clinical Trial Matching**

Team Zenith · Language Models and Agents (CL3410), Monsoon 2026 · IIIT Hyderabad

| Member | Roll number |
| --- | --- |
| Prakash Bhabad | 2025204016 |
| Metta Venkata Ramana Murthy | 2025204034 |
| Prateek Singh | 2026701041 |
| Vidur Shah | 2026909004 |

Proposal: [`docs/proposal/Zenith.pdf`](docs/proposal/Zenith.pdf)

## The problem

LLM systems such as [TrialGPT](https://github.com/ncbi-nlp/TrialGPT) check a patient note against each eligibility rule of a clinical trial. When the note doesn't settle a rule, they either mark it "not enough information" or quietly fill the gap. The first throws away trials the patient may qualify for. The second can show an ineligible patient as eligible, which is a safety risk.

## What TrialGuide does

For every rule the note doesn't settle, TrialGuide picks one of three actions:

- **Infer.** It infers the fact only when the note gives evidence, such as a stated value or a calculation like eGFR from age, sex and creatinine. Every inference cites the sentences it used.
- **Ask.** It asks the doctor a short question. It asks at most 3 questions per patient, one at a time, and re-ranks the trials after each answer. Each question is chosen by how much its answer could change the top 10 trials.
- **Abstain.** It leaves the rule unresolved and says so.

An inference that would make a patient look eligible is accepted only if it passes **conformal risk control**. This keeps the average rate of harmful inferences at or below a target of 5%. Harmful means wrong *and* moving the patient toward eligibility.

**The doctor.** In experiments, a simulated doctor (Llama-3.1-8B-Instruct) answers each question. It must quote the sentence it used, and gives a certainty level: sure, probably, unsure or not documented. A quote that is not in the record turns the answer into "not documented", and a "sure" answer whose quote does not mention the question's topic is lowered to "probably". A "probably" answer goes through the same risk control as an inference, and an "unsure" answer can lead to one follow-up question. In the demo, a person answers in place of the simulated doctor.

## Research questions

- **RQ1.** With at most 3 questions per patient, does the infer / ask / abstain policy recover more of the full-information trial ranking than fixed-order questioning and plain abstention, without more false-eligible decisions?
- **RQ2.** How does inference coverage trade off against harmful-inference risk? Does conformal risk control do better than plain confidence thresholds?
- **RQ3.** Does QLoRA fine-tuning improve the 8B rule checker over the untuned model, without raising the harmful-inference rate?

## How it covers the course components

| Component | Where in TrialGuide |
| --- | --- |
| RAG | Trials are retrieved from the visible note (BM25 + MedCPT), and every inference is grounded in cited sentences |
| Reasoning / agent | The infer / ask / abstain policy and ranking-aware question selection over a multi-turn conversation |
| Fine-tuning | QLoRA on Qwen3-8B, trained on rule checks that include the evidence and a short reasoning |
| Tools | eGFR (CKD-EPI 2021), BMI and drug-class lookup, for values the model shouldn't guess |

## Models and data

| | |
| --- | --- |
| Main model | Qwen3-8B on the IIIT cluster (Hugging Face transformers, 8-bit on one 12 GB GPU) |
| Simulated doctor | Llama-3.1-8B-Instruct (a different model family from the main model) |
| Patients | TREC Clinical Trials 2021 (75), TREC Clinical Trials 2022 (50) and SIGIR 2016 (58): 183 expert-written synthetic patients |
| Rule-level gold labels | [TrialGPT criterion annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations) (1,015 rows, all for SIGIR patients) |
| Extension | Indian-affiliated case reports from PMC-Patients |

**Splits.** Patients are split once: 37 for development, 73 for calibration and 73 for test. The test set is opened once, after the system is frozen. Facts are hidden from notes in three ways (random, rule-targeted and evidence-targeted), and the system only ever sees the visible part.

## Evaluation

- **Baselines:**
  - B1: full-information oracle;
  - B2: inference without risk control;
  - B3: ask everything;
  - B4: fixed-order questions;
  - B5: confidence threshold;
  - B6: TrialGPT-style abstention;
  - B7: TrialGuide.
- **Ablations:** A1–A7. These remove one part at a time: inference, adaptive selection, the ranking term, evidence retrieval, tools, risk control, and one-question-per-turn asking.
- **Metrics:**
  - NDCG@10 and recovery of the full-information ranking;
  - false-eligible and false-exclusion rates;
  - harmful-inference rate, coverage and abstention;
  - questions asked per patient.
- **Statistics:** patient-level bootstrap confidence intervals and McNemar's test.
- **Checks on the simulated doctor and on retrieval:** quotes must exist in the record, answers are compared with a teammate answering blind, and cited evidence is compared with expert evidence.

## Changes since the proposal

- **Main model:** Qwen3-8B instead of Qwen3-32B. The 32B model doesn't fit in the memory of the GPUs we have (2 × 12 GB). RQ3 was redefined to match: tuned vs untuned 8B, instead of 8B vs 32B.
- **Simulated doctor:** following TA feedback, its answers must quote the record and carry a certainty level. The plan is for it to read the full record. The current code still gives it only the facts hidden from the system; the switch to the full record is in progress.
- **Checker confidence:** the model's own probability of the status it chose, from one greedy call, instead of agreement across 5 samples. With greedy decoding the 5 samples always agreed, so the vote share was always 1.0 and no threshold could act. Risk control calibrates this probability.
- **Label count:** the expert label file has 1,015 rule-level rows, not the 1,020 stated in the proposal.

## Repository layout

```
src/trialguide/
  data/        loading, patient splits, masking          (Murthy)
  eval/        metrics and statistics                    (Murthy)
  llm/         shared model client                       (Prateek, Vidur)
  retrieval/   BM25 + MedCPT trial retrieval             (Vidur)
  checker/     rule checker with evidence and confidence (Vidur)
  finetune/    QLoRA fine-tuning                         (Vidur)
  simulator/   simulated doctor                          (Prateek)
  selector/    question selection                        (Prateek)
  tools/       eGFR, BMI, drug class                     (Prateek)
  policy/      infer / ask / abstain and risk control    (Prakash)
  baselines/   B1-B7 and ablations                       (Prakash, Prateek)
  pipeline.py  one patient end to end: check, decide, ask, re-check, score
configs/       one file per experiment run
scripts/       data download and cluster job scripts
splits/        the frozen patient splits (with checksums)
results/       result files used in the reports
tests/         unit tests (run automatically on every pull request)
docs/          proposal, mid-submission report, leakage checklist, E2 limits
```

## Setup

Requires Python 3.11 or later.

```bash
git clone https://github.com/Pakkabhabad18/trialguide.git
cd trialguide
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

Patient data and model weights are not stored in this repository. They are downloaded by scripts in `scripts/`.

## Status (30 October 2026)

**Built and tested:** data download, expert-label audit, frozen patient split, masking, evaluation harness, E2 data, TrialGPT baseline, tools, rule checker, infer / ask / abstain policy, question selector, simulated doctor and the end-to-end runner.

**First runs (5 development patients, too few to compare systems):** TrialGPT reproduction, B6, B2 and B5, and a memorisation probe on all 37 development patients (no memorisation found). Numbers and caveats: [`results/mid_submission_results.txt`](results/mid_submission_results.txt).

**Next:** trial retrieval from the visible note and the RAG check, the checker calling the tools, risk-control calibration, B7 and the other baselines on a larger development sample, the ablations, QLoRA fine-tuning (E1) and E2.

Final submission: 15 November 2026.
