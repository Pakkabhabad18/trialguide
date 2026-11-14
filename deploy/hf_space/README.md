---
title: TrialGuide
emoji: 🩺
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: true
short_description: Infer, ask or abstain on clinical trial eligibility rules
---

# TrialGuide demo

When a patient note does not settle a clinical-trial eligibility rule, TrialGuide **infers** it from
evidence in the note, **asks** the doctor one targeted question at a time, or **abstains**. Inferences
that would make a patient look eligible must pass conformal risk control. In this demo you play the
doctor: answer its questions and watch the trial ranking change against plain abstention (B6).

The page replays real Qwen3-8B outputs recorded on three development patients (risk threshold
λ = 0.766), so it runs without a GPU. Code, results and the full write-up:
[github.com/Pakkabhabad18/trialguide](https://github.com/Pakkabhabad18/trialguide).

Team Zenith, Language Models and Agents, IIIT Hyderabad, Monsoon 2026.
