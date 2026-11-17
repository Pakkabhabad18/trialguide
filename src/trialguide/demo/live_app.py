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

from trialguide.demo import live_view as view_
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


def build(raw: Path, fake: bool = False, cohorts=("sigir", "trec_2021"), home: str = ""):
    """The live page. home: link back to the recorded demo (set on the Hugging Face Space)."""
    import gradio as gr

    if fake:
        first, second = keyword_checker(), keyword_checker(settle_open=True)
    else:
        from trialguide.pipeline import checker
        first, second = checker(FIRST_MODEL, 1), checker(SECOND_MODEL, 1)
    print("building the trial index ...", flush=True)
    index = TrialIndex(raw, cohorts)
    print(f"{len(index.trials)} trials indexed", flush=True)

    # ---- what every step returns: the session, the record, and every panel of the results ----
    def view(s, record, trial=None):
        order = s.order(s.scores())
        trial = trial if trial in order else (order[0] if order else None)
        live = not s.finished
        return (s, record,
                gr.update(visible=False), gr.update(visible=False),        # hide the start form and the steps
                gr.update(visible=True),                                   # show the results
                view_.patient_bar(s), view_.tiles(s), view_.question(s), view_.trials(s, record),
                view_.conversation(s), render.note_html(record, s, False),
                gr.update(choices=order, value=trial), render.rules_html(s, trial),
                gr.update(visible=live))                                   # answer buttons only while asking

    def start(mode, text, upload, age, sex, condition, other, n_trials, budget, progress=gr.Progress()):
        progress(0.05, desc="Reading the note")
        try:
            if mode == "facts":
                note = note_from_facts(age, sex, condition, other)
            elif mode == "upload":
                if not upload:
                    raise ValueError
                note = note_from_text(read_upload(upload))
            else:
                note = note_from_text(text)
        except ValueError:
            raise gr.Error({"facts": "Type at least the main diagnosis.",
                            "upload": "Choose a .txt or .pdf file first."}.get(mode, "Paste a patient note first."))
        progress(0.15, desc="Searching the trials")
        found = index.search(note, k=int(n_trials))
        if not found:
            raise gr.Error("No trial matched this note. Add the diagnosis or more detail.")
        rules = [r for t in found for r in rules_of_corpus_trial(t)]
        progress(0.3, desc=f"Checking {len(rules)} rules of {len(found)} trials")
        s = TwoModelSession(note, rules, first, second, threshold=THRESHOLD,
                            budget=max(int(budget), 6) if mode == "facts" else int(budget))
        s.start()
        return view(s, patient_record(note, found))

    def reply(s, record, value, certainty, trial):
        """One answer to the current question: 'yes', 'no', 'not in the record', or typed text."""
        if s is None or s.question is None:
            raise gr.Error("There is no open question.")
        value = (value or "").strip()
        if not value:
            raise gr.Error("Type the answer first.")
        if value == "not in the record":
            v, c = "", Certainty.NOT_DOCUMENTED
        else:
            v, c = value, CERTAINTY[certainty]
        twin = s.fork()                    # a copy, so an earlier state is never changed in place
        twin.answer(v, c)
        return (*view(twin, record, trial), "")

    def add(s, record, text, trial):
        if s is None:
            raise gr.Error("Check a patient first.")
        if not (text or "").strip():
            raise gr.Error("Type the new information first.")
        twin = s.fork()
        twin.add_info(text)
        return (*view(twin, record, trial), "")

    def restart():
        """Back to the start form, with the old patient cleared."""
        return (None, None, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False))

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
    with gr.Blocks(title="TrialGuide · try your own patient", theme=theme, css=render.CSS + view_.CSS,
                   js=view_.FORCE_LIGHT) as demo:
        state, record, mode = gr.State(None), gr.State(None), gr.State("note")
        gr.HTML(view_.header(fake, home))

        # ---- 1. the start form: three ways to give the patient, as tabs ----
        with gr.Column(elem_classes="lv-panel") as form:
            gr.HTML('<div class="lv-h">Add your patient</div>'
                    '<div class="lv-sub">Choose one way. Do not enter real names or identifying details.</div>')
            with gr.Tabs():
                with gr.Tab("Paste a note") as tab_note:
                    text = gr.Textbox(lines=6, show_label=False, placeholder="e.g. A 58-year-old woman with breast "
                                      "cancer, estrogen receptor positive. She takes metformin for diabetes ...")
                    example = gr.Button("Use an example patient", size="sm", variant="secondary")
                with gr.Tab("Upload a file") as tab_upload:
                    upload = gr.File(label="A .txt or .pdf note", file_types=[".txt", ".md", ".pdf"], type="filepath")
                with gr.Tab("Enter a few facts") as tab_facts:
                    with gr.Row():
                        age = gr.Textbox(label="Age", placeholder="52", scale=1)
                        sex = gr.Dropdown(["Female", "Male"], label="Sex", scale=1)
                        condition = gr.Textbox(label="Main diagnosis", placeholder="e.g. type 2 diabetes", scale=3)
                    other = gr.Textbox(label="Anything else you know (optional)", lines=2,
                                       placeholder="e.g. HbA1c is 8.5 percent on metformin.")
                    gr.HTML('<div class="lv-sub">TrialGuide will ask you for the rest.</div>')
            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    n_trials = gr.Slider(2, 5, value=3, step=1, label="How many trials to check")
                    budget = gr.Slider(1, 8, value=3, step=1, label="Most questions TrialGuide may ask")
            go = gr.Button("Find matching trials  →", variant="primary", elem_classes="lv-big")
        steps = gr.HTML(view_.how_it_works())

        # ---- 2. the results: hidden until a patient is checked ----
        with gr.Column(visible=False) as results:
            with gr.Row(equal_height=True):
                patient = gr.HTML()
                again = gr.Button("↺  New patient", variant="secondary", scale=0, min_width=150)
            tiles = gr.HTML()
            with gr.Row():
                with gr.Column(scale=5):
                    question = gr.HTML()
                    with gr.Column(visible=True) as answer_box:
                        with gr.Row(elem_classes="lv-answer"):
                            yes = gr.Button("Yes", variant="primary")
                            no = gr.Button("No", variant="primary")
                            na = gr.Button("Not in the record", variant="secondary")
                        certainty = gr.Radio(list(CERTAINTY), value="Sure", label="How sure are you?")
                        with gr.Accordion("Answer in your own words", open=False):
                            typed = gr.Textbox(show_label=False, placeholder="e.g. 1.4 mg/dL, or stage II")
                            send = gr.Button("Send this answer", size="sm")
                    with gr.Accordion("Add new information at any time", open=False):
                        info = gr.Textbox(show_label=False, lines=2,
                                          placeholder="e.g. Creatinine is 1.1 mg/dL. She is not pregnant.")
                        add_btn = gr.Button("Add and re-check", size="sm")
                    conv = gr.HTML()
                with gr.Column(scale=6):
                    trials = gr.HTML()
            with gr.Accordion("Details: the note and every rule checked", open=False):
                with gr.Row():
                    with gr.Column(scale=4):
                        note = gr.HTML()
                    with gr.Column(scale=6):
                        pick = gr.Dropdown(label="Trial", choices=[])
                        rules = gr.HTML()
                save_btn = gr.Button("Save this session as JSON", size="sm")
                saved = gr.File(label="Session log", height=90)

        # ---- wiring ----
        tab_note.select(lambda: ("note", 3), None, [mode, budget])
        tab_upload.select(lambda: ("upload", 3), None, [mode, budget])
        tab_facts.select(lambda: ("facts", 6), None, [mode, budget])
        example.click(lambda: EXAMPLE_NOTE, None, text)
        outputs = [state, record, form, steps, results, patient, tiles, question, trials, conv, note, pick, rules,
                   answer_box]
        go.click(start, [mode, text, upload, age, sex, condition, other, n_trials, budget], outputs)
        for btn, value in ((yes, "yes"), (no, "no"), (na, "not in the record")):
            btn.click(lambda s, r, c, t, v=value: reply(s, r, v, c, t), [state, record, certainty, pick],
                      outputs + [typed])
        send.click(reply, [state, record, typed, certainty, pick], outputs + [typed])
        add_btn.click(add, [state, record, info, pick], outputs + [info])
        again.click(restart, None, [state, record, form, steps, results])
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
