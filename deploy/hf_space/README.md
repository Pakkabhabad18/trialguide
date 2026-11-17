---
title: TrialGuide
emoji: 🩺
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: true
short_description: Try your own patient note and answer as the doctor
---

# TrialGuide live: try your own patient

Paste or upload a patient note, or start from a few facts. TrialGuide finds matching trials, checks
every eligibility rule, and asks you, the doctor, only about the rules it cannot settle. You can add
new information at any time and the open rules are checked again.

This Space has no GPU for the two models (Qwen3-8B first, Llama-3.1-8B second), so it uses a
word-matching stand-in and shows how the flow works, not the models' decisions. The recorded demo
with real Qwen3-8B decisions: [trialguide-demo](https://huggingface.co/spaces/Pakkabhabad18/trialguide-demo).
Code, results and the write-up: [github.com/Pakkabhabad18/trialguide](https://github.com/Pakkabhabad18/trialguide).

Team Zenith, Language Models and Agents, IIIT Hyderabad, Monsoon 2026. Research prototype, not for clinical use.
