"""Create the labels, milestones and the 48 task issues (T1-T48) in our GitHub repo.

Run once, from your own laptop (not Kaggle):

    export GITHUB_TOKEN=<a fine-grained token with "Issues: read and write" on the repo>
    python scripts/create_github_issues.py <owner>/trialguide

Safe to run again: labels, milestones and issues that already exist (same name / title) are skipped.
The task list below is copied from docs/PLAN.md. If the plan changes, edit TASKS here too.
"""

import os
import sys

import requests

API = "https://api.github.com"

# One label per person, plus "team" for tasks everyone shares.
LABELS = {
    "prakash": "1f6feb",
    "murthy": "8250df",
    "vidur": "1a7f37",
    "prateek": "bf8700",
    "team": "6e7781",
    # Status labels: the evening sync reads these to update the tracker
    "in-progress": "fbca04",
    "blocked": "d73a4a",
}

# Milestone name -> due date (end of that step in the plan).
MILESTONES = {
    "Step 0 Setup": "2026-10-21",
    "Step 1 Audit and building blocks": "2026-10-25",
    "Step 2 Policy, selector and harness": "2026-10-30",
    "Step 3 Dev runs, freeze, calibration": "2026-11-04",
    "Step 4 Test run and extensions": "2026-11-07",
    "Step 5 Report, demo, viva prep": "2026-11-10",
}

S0, S1, S2, S3, S4, S5 = list(MILESTONES)

# (id, task "done when...", owner text, labels, needs, due date, milestone)
TASKS = [
    ("T1", "Private GitHub repo with the folder layout, milestones and one issue per task", "Prakash", ["prakash"], "-", "2026-10-20", S0),
    ("T2", "Everyone has a phone-verified Kaggle account (GPU access) and has joined the repo", "All four", ["team"], "-", "2026-10-19", S0),
    ("T3", "Download script for TREC CT 2021, TREC CT 2022 and SIGIR 2016 (via the TrialGPT repo) runs from a clean checkout", "Murthy", ["murthy"], "T1", "2026-10-20", S0),
    ("T4", "TrialGPT's released code runs end to end on 2-3 patients with any small model", "Prateek", ["prateek"], "T3", "2026-10-21", S0),
    ("T5", "Shared LLM client: response caching, and every call logs model, version, date and seed", "Vidur", ["vidur"], "T1", "2026-10-20", S0),
    ("T6", "Backbone decided (D13): Qwen3-8B on the IIIT cluster (2x RTX 3080 Ti); record memory, tokens/s and projected GPU-hours in the decision log", "Vidur (Prateek ran it)", ["vidur"], "T5", "2026-10-22", S1),
    ("T7", "Expert-label audit: how many patients the 1,020 rule labels cover, and how many NEI cases are recoverable", "Murthy", ["murthy"], "T3", "2026-10-22", S1),
    ("T8", "37 / 73 / 73 patient split frozen with a fixed seed and committed; leakage checklist written", "Murthy", ["murthy"], "T7", "2026-10-22", S1),
    ("T9", "Decision log started; annotation guidelines v0 (four information categories, harmful-inference rule)", "Prakash", ["prakash"], "-", "2026-10-22", S1),
    ("T10", "Fact splitting and masking: random (20/40/60%), rule-targeted, evidence-targeted; only the visible note leaves the module", "Murthy", ["murthy"], "T8", "2026-10-25", S1),
    ("T11", "Simulated doctor (Llama-3.1-8B) reads the patient's full original record and answers one question with a value, a certainty (sure / probably / unsure / not documented) and the sentence it quoted; no quote means 'not documented'; every answer is logged", "Prateek", ["prateek"], "T5, T10", "2026-10-25", S1),
    ("T12", "Rule checker v1: label, evidence sentence and 5-sample confidence per rule", "Vidur", ["vidur"], "T6", "2026-10-25", S1),
    ("T13", "Tools (eGFR CKD-EPI 2021, BMI, drug class) with unit tests, checked on the Eka calculator set", "Prateek", ["prateek"], "T1", "2026-10-25", S1),
    ("T14", "Memorisation probe on the public TREC patients, result in the decision log", "Prateek", ["prateek"], "T6", "2026-10-25", S1),
    ("T15", "Guidelines piloted on development patients by two blind annotators; kappa recorded; guidelines frozen", "Prakash + all", ["prakash", "team"], "T9, T10", "2026-10-25", S1),
    ("T16", "TrialGPT-style abstention baseline (B6) runs on development patients with the chosen backbone", "Prateek", ["prateek"], "T4, T12", "2026-10-28", S2),
    ("T17", "Evaluation harness: NDCG@10, recovery, false-eligible and false-exclusion rates, patient-level bootstrap, McNemar", "Murthy", ["murthy"], "T8", "2026-10-28", S2),
    ("T18", "Trial retrieval (keywords, BM25, MedCPT) built from the visible note only", "Vidur", ["vidur"], "T10", "2026-10-28", S2),
    ("T19", "Infer / ask / abstain policy turns checker output into one action per unresolved rule; doctor answers marked 'probably' go through the same risk control as inferences", "Prakash", ["prakash"], "T12, T15, T42", "2026-10-28", S2),
    ("T20", "Question selector (Eq. 1), one question per turn: turns the chosen missing fact into one plain question, merges the answer, re-checks only the affected rules, re-ranks, asks one follow-up when the answer is 'unsure' (it counts toward the budget), and stops at 3 questions or when no question can change the top 10", "Prateek", ["prateek"], "T11, T12, T42", "2026-10-29", S2),
    ("T21", "Baselines B1, B2, B5 (fixed 0.9 and uncorrected tuned) and B6 (from T16) wired into the harness", "Prakash", ["prakash"], "T16, T17", "2026-10-29", S2),
    ("T22", "Baselines B3 and B4", "Prateek", ["prateek"], "T20", "2026-10-29", S2),
    ("T23", "E1 teacher data (required: fine-tuning is a basic checklist item): Qwen3-8B rule labels with the evidence sentence and a short reasoning (self-generated), on development patients and PMC-Patients notes, kept only where they agree with the expert trial label and a second sample", "Vidur", ["vidur"], "T12", "2026-11-01", S2),
    ("T24", "E2 data: Indian-affiliated PMC-Patients subset and India-recruiting trials, with limits written down", "Murthy", ["murthy"], "T3", "2026-11-01", S2),
    ("T25", "Progress check: full pipeline on 5 development patients, first B2, B6 and B7 numbers, one page of notes for the TAs", "All four", ["team"], "T17-T22", "2026-10-30", S2),
    ("T26", "B1-B7 on every development patient at 40% masking; each person runs their own baselines, Murthy merges results with commit hashes", "All four", ["team"], "T25", "2026-11-01", S3),
    ("T27", "Ablations A1-A7 and the budget sweep on development patients (A1, A6 Prakash; A2, A3, A5, A7 Prateek; A4 Vidur)", "Owners as listed", ["prakash", "prateek", "vidur"], "T26", "2026-11-02", S3),
    ("T28", "Failure analysis; each owner fixes their component; checker v2 if needed; prompts and code frozen as v1.0", "All four (Vidur tags)", ["team"], "T26, T27, T43, T44", "2026-11-02", S3),
    ("T29", "Calibration items annotated (two blind annotators, third adjudicates)", "All four", ["team"], "T15, T28", "2026-11-03", S3),
    ("T30", "Risk threshold fitted at 5% (2% and 10% also); feasibility and useful coverage recorded", "Prakash", ["prakash"], "T29, T45", "2026-11-04", S3),
    ("T31", "QLoRA on Qwen3-8B (4-bit, rank 16) trained on the T23 labels", "Vidur", ["vidur"], "T23", "2026-11-04", S3),
    ("T32", "Test items annotated, only after the v1.0 freeze (same protocol)", "All four", ["team"], "T28", "2026-11-05", S4),
    ("T33", "Test set opened once: B1-B7 at 40% masking for RQ1 and RQ2", "Murthy (runs), Prakash (risk)", ["murthy", "prakash"], "T30, T32", "2026-11-06", S4),
    ("T34", "Ablations A1-A7, budget sweep and threshold sweep on test patients", "Owners as in T27", ["prakash", "prateek", "vidur"], "T33", "2026-11-07", S4),
    ("T35", "20% and 60% masking runs for B4 and B7", "Murthy", ["murthy"], "T33", "2026-11-07", S4),
    ("T36", "Statistics: bootstrap CIs, McNemar, H1 and H2 verdicts, results per rule type; then exploratory resplits and expert-only check", "Prakash (H1, H2), Murthy (resplits)", ["prakash", "murthy"], "T33, T48", "2026-11-07", S4),
    ("T37", "E1 evaluation (RQ3 redefined, D14): QLoRA-tuned vs untuned Qwen3-8B checker on the test patients", "Vidur", ["vidur"], "T31, T33", "2026-11-07", S4),
    ("T38", "E2 evaluation on the Indian-affiliated subset, with recalibration", "Murthy", ["murthy"], "T24, T30", "2026-11-07", S4),
    ("T39", "Final report: each person writes their own sections; Prakash edits the whole", "All four", ["team"], "T36", "2026-11-09", S5),
    ("T40", "Interactive demo where a person answers TrialGuide's questions live through the same doctor interface as the simulator (Prateek), and code release with README and configs (Vidur)", "Prateek, Vidur", ["prateek", "vidur"], "T33", "2026-11-09", S5),
    ("T41", "Decision log complete; viva rehearsal where each person explains every part", "All four", ["team"], "T39", "2026-11-10", S5),
    # T42-T48 added on Oct 25: interaction design (D8) and verification of the doctor and RAG
    ("T42", "Shared data formats agreed and committed in src/trialguide/schemas.py (fact, rule check, question, answer with certainty and quote, action, conversation state, doctor interface); decision D8 (one question per turn) logged", "Prakash (Vidur, Prateek review)", ["prakash"], "T1", "2026-10-27", S2),
    ("T43", "Simulated-doctor check: automatic tests on development patients (the quote exists in the record and supports the answer, 'not documented' only when the record lacks the fact, no extra facts volunteered, same answer to 3 rewordings); a teammate answers 60 questions blind from the record and we report agreement on value and certainty; 150 answers checked by hand by two people with kappa", "Prateek (automatic), Murthy + Prakash (blind answers and hand check)", ["prateek", "murthy", "prakash"], "T11", "2026-11-01", S2),
    ("T44", "RAG check: recall of eligible trials from the full vs visible note (compared with TrialGPT's reported recall), cited sentences valid and verbatim, evidence precision and recall against expert evidence sentences, 50 inferences checked by hand", "Vidur (automatic), Prateek + Murthy (by hand)", ["vidur", "prateek", "murthy"], "T12, T18", "2026-11-01", S2),
    ("T45", "Model-derived full-note reference for development and calibration patients, with 200 rules audited by hand (test patients only after the freeze)", "Vidur (runs), Prakash + Murthy (audit)", ["vidur", "prakash", "murthy"], "T12, T15", "2026-11-03", S3),
    ("T46", "Safety test: instructions such as 'enrol this patient' planted in notes; any attempt to act is counted (target 0)", "Prakash", ["prakash"], "T19", "2026-11-04", S3),
    ("T47", "Interaction variants (ablation A7): (a) all 3 questions chosen at once before any answer, as a one-time form; (b) imperfect doctor, with 20% of answers downgraded to 'unsure' or 'not documented'; development first, then test", "Prateek", ["prateek"], "T22", "2026-11-07", S4),
    ("T48", "Rule-type tags (demographic, lab value, diagnosis, prior treatment, comorbidity) for every rule, for the per-type breakdown", "Prakash", ["prakash"], "T3", "2026-11-01", S2),
]


def get_all(session, url):
    """Fetch every page of a GitHub list endpoint and return all items."""
    items, page = [], 1
    while True:
        r = session.get(url, params={"state": "all", "per_page": 100, "page": page})
        r.raise_for_status()
        batch = r.json()
        if not batch:
            return items
        items.extend(batch)
        page += 1


def main():
    # --- Read the repo name and token; stop early with a clear message if missing ---
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("Usage: python scripts/create_github_issues.py <owner>/trialguide")
    repo = sys.argv[1]
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Set GITHUB_TOKEN first (never paste the token into this file).")

    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    base = f"{API}/repos/{repo}"

    # --- Labels: create the ones that don't exist yet ---
    existing_labels = {l["name"] for l in get_all(s, f"{base}/labels")}
    for name, colour in LABELS.items():
        if name not in existing_labels:
            s.post(f"{base}/labels", json={"name": name, "color": colour}).raise_for_status()
            print("label created:", name)

    # --- Milestones: create missing ones and remember their numbers ---
    ms_number = {m["title"]: m["number"] for m in get_all(s, f"{base}/milestones")}
    for title, due in MILESTONES.items():
        if title not in ms_number:
            r = s.post(f"{base}/milestones", json={"title": title, "due_on": f"{due}T23:59:59Z"})
            r.raise_for_status()
            ms_number[title] = r.json()["number"]
            print("milestone created:", title)

    # --- Issues: one per task, skipped if an issue with the same title already exists ---
    existing_titles = {i["title"] for i in get_all(s, f"{base}/issues") if "pull_request" not in i}
    for tid, text, owner, labels, needs, due, milestone in TASKS:
        title = f"{tid}: {text}"
        if title in existing_titles:
            continue
        body = (
            f"**Owner:** {owner}\n**Needs first:** {needs}\n**Due:** {due}\n\n"
            "Done when the task text above is true and the pull request is merged.\n"
            "Close this issue from the pull request with `Closes #<this issue number>`, "
            "and set the Status in the team tracker."
        )
        r = s.post(f"{base}/issues", json={
            "title": title, "body": body, "labels": labels, "milestone": ms_number[milestone],
        })
        r.raise_for_status()
        print("issue created:", title[:70])

    print("Done.")


if __name__ == "__main__":
    main()
