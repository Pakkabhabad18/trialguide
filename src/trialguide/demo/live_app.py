"""TrialGuide live demo: bring your own patient.

    # on a machine with two GPUs (e.g. Kaggle T4 x2): both models, a public link
    LLM_HF_DEVICES="Qwen/Qwen3-8B=0,meta-llama/Llama-3.1-8B-Instruct=1" LLM_HF_4BIT=1 \
        python -m trialguide.demo.live_app --share

    # no GPU: the page with a word-matching stand-in instead of the models (layout only)
    python -m trialguide.demo.live_app --fake

Two ways to start:
  * Paste or upload a note (.txt or .pdf). It is used as it is; nothing is hidden.
  * Answer questions: type a few facts (age, sex, main diagnosis) and let TrialGuide ask for the rest.
Either way you can add more information at any time, and you answer TrialGuide's questions as the
doctor. See live.py for what happens at each step. Owner: Prakash.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from trialguide.demo import render
from trialguide.demo.live import (
    FIRST_MODEL, SECOND_MODEL, TrialIndex, TwoModelSession, keyword_checker, note_from_facts,
    note_from_text, patient_record, read_upload, rules_of_corpus_trial,
)
from trialguide.schemas import Certainty

CERTAINTY = {"Sure": Certainty.SURE, "Probably": Certainty.PROBABLY, "Unsure": Certainty.UNSURE}
SESSIONS_DIR = Path("outputs/live_sessions")
THRESHOLD = 0.766          # lambda-hat from calibration (results/calibration), the first model's cut-off

EXAMPLE_NOTE = ("A 58-year-old woman presents with a palpable mass in the left breast. Biopsy shows invasive "
                "ductal carcinoma, estrogen receptor positive, HER2 negative. She has type 2 diabetes treated "
                "with metformin. She has not received chemotherapy. ECOG performance status is 1.")


def hero(live: bool, home: str = "") -> str:
    mode = "Live models" if live else "Stand-in checker (no GPU): decisions are word matches, not the models"
    pills = [f"First model: {FIRST_MODEL.split('/')[-1]} · λ = {THRESHOLD}",
             f"Second model: {SECOND_MODEL.split('/')[-1]}", mode, "Research prototype · not for clinical use"]
    return ('<div class="tg-hero"><h1>TrialGuide · live</h1>'
            '<p>Bring your own patient. The <b>first model</b> checks every rule of the best-matching trials. '
            'Rules it cannot settle go to a <b>second model</b>. Only what neither model can settle is '
            '<b>asked</b> of you, the doctor.</p>'
            + "".join(f'<span class="tg-pill">{p}</span>' for p in pills)
            + (f'<p><a href="{home}" style="color:#fff">Back to the recorded demo</a></p>' if home else "")
            + "</div>")


def build(raw: Path, fake: bool = False, cohorts=("sigir", "trec_2021"), home: str = ""):
    """home: link back to the recorded demo, when both are served together (deploy/web/app.py)."""
    import gradio as gr

    if fake:
        first, second = keyword_checker(), keyword_checker(settle_open=True)
    else:
        from trialguide.pipeline import checker
        first, second = checker(FIRST_MODEL, 1), checker(SECOND_MODEL, 1)
    print("building the trial index ...", flush=True)
    index = TrialIndex(raw, cohorts)
    print(f"{len(index.trials)} trials indexed", flush=True)

    def view(s, record, trial=None):
        """Every panel for the current state (s and record are None before the first check)."""
        if s is None:
            return (s, record, render.stepper(None), "", render.intro_html(), render.question_html(None),
                    render.timeline_html(None), "", gr.update(choices=[], value=None), "",
                    gr.update(interactive=False), gr.update(interactive=False))
        order = s.order(s.scores())
        trial = trial if trial in order else (order[0] if order else None)
        return (s, record, render.stepper(s), render.kpis(s, {}), render.trials_html(s, record),
                render.question_html(s), render.timeline_html(s), render.note_html(record, s, False),
                gr.update(choices=order, value=trial), render.rules_html(s, trial),
                gr.update(interactive=not s.finished), gr.update(interactive=True))

    def start(mode, text, upload, age, sex, condition, other, n_trials, budget, progress=gr.Progress()):
        progress(0.05, desc="Reading the note")
        try:
            if mode == "Paste or upload a note":
                if upload:
                    text = read_upload(upload)
                note = note_from_text(text)
            else:
                note = note_from_facts(age, sex, condition, other)
        except ValueError:
            raise gr.Error("Please paste a note, upload a file, or fill in at least the main diagnosis.")
        progress(0.15, desc="Searching the trials")
        trials = index.search(note, k=int(n_trials))
        if not trials:
            raise gr.Error("No trial matched this note. Add the diagnosis or more detail.")
        rules = [r for t in trials for r in rules_of_corpus_trial(t)]
        progress(0.3, desc=f"Both models are checking {len(rules)} rules of {len(trials)} trials (takes a minute)")
        s = TwoModelSession(note, rules, first, second, threshold=THRESHOLD, budget=int(budget))
        s.start()
        return view(s, patient_record(note, trials))

    def answer(s, record, value, typed, certainty, trial):
        if s is None or s.question is None:
            raise gr.Error("Check a patient first, or the conversation is over.")
        typed = (typed or "").strip()
        if typed:                                      # a typed answer wins, e.g. "1.4 mg/dL"
            v, c = typed, CERTAINTY[certainty]
        elif value is None:
            raise gr.Error("Choose Yes, No or Not in the record, or type the answer.")
        elif value == "Not in the record":
            v, c = "", Certainty.NOT_DOCUMENTED
        else:
            v, c = value.lower(), CERTAINTY[certainty]
        twin = s.fork()
        twin.answer(v, c)
        return (*view(twin, record, trial), None, "")

    def add(s, record, text, trial):
        if s is None:
            raise gr.Error("Check a patient first.")
        if not (text or "").strip():
            raise gr.Error("Type the new information first.")
        twin = s.fork()
        twin.add_info(text)
        return (*view(twin, record, trial), "")

    def save(s):
        if s is None:
            raise gr.Error("Nothing to save yet.")
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = SESSIONS_DIR / f"live_{stamp}.json"
        rec = {"saved": stamp, "note": s.note.sentences, "decided_by_second_model": sorted(map(list, s.by_second)),
               **s.to_record()}
        path.write_text(json.dumps(rec, indent=2, default=str) + "\n")
        return str(path)

    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="violet", radius_size="lg",
                           font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
    with gr.Blocks(title="TrialGuide live", theme=theme, css=render.CSS) as demo:
        state, record = gr.State(None), gr.State(None)
        gr.HTML(hero(not fake, home))

        # ---- input: a whole note, or a few facts and then questions ----
        mode = gr.Radio(["Paste or upload a note", "Answer questions (start with a few facts)"],
                        value="Paste or upload a note", label="How do you want to start?")
        with gr.Group(visible=True) as note_box:
            with gr.Row():
                text = gr.Textbox(EXAMPLE_NOTE, lines=6, label="Patient note", scale=3,
                                  info="Write as in a clinical note. Do not paste real identifying details.")
                upload = gr.File(label="…or upload a .txt / .pdf note", file_types=[".txt", ".md", ".pdf"],
                                 type="filepath", scale=1)
        with gr.Group(visible=False) as facts_box:
            with gr.Row():
                age = gr.Textbox(label="Age", scale=1)
                sex = gr.Dropdown(["Female", "Male"], label="Sex", scale=1)
                condition = gr.Textbox(label="Main diagnosis", scale=3, placeholder="e.g. type 2 diabetes")
            other = gr.Textbox(label="Anything else you already know (optional)", lines=2)
        with gr.Row():
            n_trials = gr.Slider(2, 5, value=3, step=1, label="Trials to check", scale=2)
            budget = gr.Slider(1, 8, value=3, step=1, label="Most questions TrialGuide may ask", scale=2)
            go = gr.Button("Find and check trials", variant="primary", size="lg", scale=2)
        mode.change(lambda m: (gr.update(visible=m.startswith("Paste")), gr.update(visible=not m.startswith("Paste")),
                               gr.update(value=3 if m.startswith("Paste") else 6)),
                    [mode], [note_box, facts_box, budget])

        # ---- output and the conversation ----
        steps = gr.HTML(render.stepper(None))
        tiles = gr.HTML("")
        with gr.Row():
            with gr.Column(scale=6):
                trials = gr.HTML(render.intro_html())
            with gr.Column(scale=4):
                question = gr.HTML(render.question_html(None))
                value = gr.Radio(["Yes", "No", "Not in the record"], value=None, label="Your answer as the doctor")
                typed = gr.Textbox(label="…or type the answer", placeholder="e.g. 1.4 mg/dL, or 'stage II'")
                certainty = gr.Radio(list(CERTAINTY), value="Sure", label="How sure are you?")
                send = gr.Button("Send answer", variant="primary", interactive=False)
                with gr.Group():
                    info = gr.Textbox(label="Add new information at any time", lines=2,
                                      placeholder="e.g. Creatinine is 1.1 mg/dL. She is not pregnant.")
                    add_btn = gr.Button("Add to the note and re-check", interactive=False)
                timeline = gr.HTML(render.timeline_html(None))
        with gr.Row():
            with gr.Column(scale=4):
                note = gr.HTML("")
            with gr.Column(scale=6):
                pick = gr.Dropdown(label="Look inside a trial", choices=[])
                rules = gr.HTML("")
        with gr.Accordion("Save this session as JSON", open=False):
            save_btn = gr.Button("Save this session")
            saved = gr.File(label="Session log (JSON)", height=90)

        outputs = [state, record, steps, tiles, trials, question, timeline, note, pick, rules, send, add_btn]
        go.click(start, [mode, text, upload, age, sex, condition, other, n_trials, budget], outputs)
        send.click(answer, [state, record, value, typed, certainty, pick], outputs + [value, typed])
        add_btn.click(add, [state, record, info, pick], outputs + [info])
        pick.change(lambda s, t: render.rules_html(s, t), [state, pick], [rules])
        save_btn.click(save, [state], [saved])
    return demo


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--fake", action="store_true", help="no GPU: word-matching stand-in instead of the models")
    ap.add_argument("--cohorts", default="sigir,trec_2021", help="trial collections to search")
    ap.add_argument("--share", action="store_true", help="public link (e.g. on Kaggle)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)
    build(args.raw, args.fake, tuple(args.cohorts.split(","))).queue().launch(
        share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
