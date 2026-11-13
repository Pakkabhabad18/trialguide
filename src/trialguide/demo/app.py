"""TrialGuide demo: a web page where you play the doctor.

    pip install -r requirements-demo.txt
    python -m trialguide.demo.app --bundle demo/demo_bundle.json            # laptop, no GPU
    python -m trialguide.demo.app --bundle demo/demo_bundle.json --live     # GPU: any answer path

1. Pick a development patient and press "Check this patient". The checker reads only the
   visible note; the hidden sentences are shown as redacted bars (tick the box to reveal them).
2. The candidate trials are ranked. Each card shows how its rules were settled (stated,
   inferred, answered by you, ruling the trial out, or open) and its movement against plain
   abstention (B6). The top tiles count rules, inferences held back by the risk threshold,
   questions left, and NDCG@10 of both rankings against the expert labels.
3. TrialGuide asks one question at a time (at most three) and says why it chose it. Answer
   as the doctor, with a certainty; the answered rules are re-checked and the page updates.
4. "Save this session" writes the whole conversation and both rankings to a JSON file.
Owner: Prakash.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from trialguide.demo import render
from trialguide.demo.record import (
    MissingRecording, RecordingChecker, ReplayChecker, load_bundle, patient_note, patient_rules,
)
from trialguide.demo.session import DemoSession
from trialguide.schemas import Certainty

CERTAINTY = {"Sure": Certainty.SURE, "Probably": Certainty.PROBABLY, "Unsure": Certainty.UNSURE}
SESSIONS_DIR = Path("outputs/demo_sessions")


def build(bundle_path: Path, live: bool = False, model: str = "Qwen/Qwen3-8B"):
    import gradio as gr

    patients, store, meta = load_bundle(bundle_path)
    if live:                                           # call the model for anything not recorded
        from trialguide.pipeline import checker
        check = RecordingChecker(checker(model, 1), store)
    else:                                              # replay recorded checks only
        check = ReplayChecker(store)
    threshold = float(meta.get("threshold", 0.9))
    p_answer = None                                    # the same answerability prior the bundle was recorded with
    if meta.get("prior") and Path(meta["prior"]).exists():
        from trialguide.selector.answerability import load
        p_answer = load(Path(meta["prior"]))
    ids = sorted(patients)

    def gold_of(pid):
        return {t["trial_id"]: int(t.get("gold", 0)) for t in patients[pid]["trials"]}

    def view(s, pid, reveal, trial=None):
        """Every panel of the page for the current state."""
        p = patients[pid]
        if s is None:
            empty = '<div class="tg-card" style="color:#64748b">The ranking appears here after the check.</div>'
            return (s, render.stepper(None), "", empty, render.question_html(None), render.timeline_html(None),
                    render.note_html(p, None, reveal), gr.update(choices=[], value=None), "",
                    gr.update(interactive=False))
        order = s.order(s.scores())
        trial = trial if trial in order else (order[0] if order else None)
        return (s, render.stepper(s), render.kpis(s, gold_of(pid)), render.trials_html(s, p),
                render.question_html(s), render.timeline_html(s), render.note_html(p, s, reveal),
                gr.update(choices=order, value=trial), render.rules_html(s, trial),
                gr.update(interactive=not s.finished))

    def start(pid, reveal):
        p = patients[pid]
        s = DemoSession(patient_note(p, pid), patient_rules(p), check, threshold=threshold, p_answer=p_answer)
        try:
            s.start()
        except MissingRecording as e:
            raise gr.Error(str(e))
        return view(s, pid, reveal)

    def answer(s, pid, value, certainty, reveal, trial):
        if s is None or s.question is None:
            raise gr.Error("Press 'Check this patient' first, or the conversation is over.")
        if value == "Not documented":
            v, c = "", Certainty.NOT_DOCUMENTED
        else:
            v, c = value.lower(), CERTAINTY[certainty]
        twin = s.fork()                                # the old state stays if this path was not recorded
        try:
            twin.answer(v, c)
        except MissingRecording as e:
            raise gr.Error(str(e))
        return view(twin, pid, reveal, trial)

    def save(s, pid):
        if s is None:
            raise gr.Error("Nothing to save yet.")
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = SESSIONS_DIR / f"{pid}_{stamp}.json"
        record = {"saved": stamp, "bundle": str(bundle_path), "meta": meta, **s.to_record(gold_of(pid))}
        path.write_text(json.dumps(record, indent=2) + "\n")
        return str(path)

    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="violet", radius_size="lg",
                           font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
    with gr.Blocks(title="TrialGuide demo", theme=theme, css=render.CSS) as demo:
        state = gr.State(None)
        gr.HTML(render.hero(meta, live))
        with gr.Row(equal_height=True):
            pid = gr.Dropdown(ids, value=ids[0], label="Development patient", scale=3)
            reveal = gr.Checkbox(False, label="Reveal what only the doctor knows", scale=2)
            go = gr.Button("Check this patient", variant="primary", scale=2, size="lg")
        steps = gr.HTML(render.stepper(None))
        tiles = gr.HTML("")
        with gr.Row():
            with gr.Column(scale=6):
                trials = gr.HTML('<div class="tg-card" style="color:#64748b">The ranking appears here after the check.</div>')
            with gr.Column(scale=4):
                question = gr.HTML(render.question_html(None))
                value = gr.Radio(["Yes", "No", "Not documented"], value="Yes", label="Your answer as the doctor")
                certainty = gr.Radio(list(CERTAINTY), value="Sure", label="How sure are you?",
                                     info="Probably → treated like an inference; Unsure → one follow-up question")
                send = gr.Button("Send answer", variant="primary", interactive=False)
                timeline = gr.HTML(render.timeline_html(None))
        with gr.Row():
            with gr.Column(scale=4):
                note = gr.HTML(render.note_html(patients[ids[0]], None, False))
            with gr.Column(scale=6):
                pick = gr.Dropdown(label="Look inside a trial", choices=[])
                rules = gr.HTML("")
        with gr.Row():
            save_btn = gr.Button("Save this session")
            saved = gr.File(label="Session log (JSON)", height=90)

        outputs = [state, steps, tiles, trials, question, timeline, note, pick, rules, send]
        go.click(start, [pid, reveal], outputs)
        send.click(answer, [state, pid, value, certainty, reveal, pick], outputs)
        pid.change(lambda p, r: view(None, p, r), [pid, reveal], outputs)
        reveal.change(lambda s, p, r: render.note_html(patients[p], s, r), [state, pid, reveal], [note])
        pick.change(lambda s, t: render.rules_html(s, t), [state, pick], [rules])
        save_btn.click(save, [state, pid], [saved])
    return demo


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", type=Path, default=Path("demo/demo_bundle.json"))
    ap.add_argument("--live", action="store_true", help="call the model for paths not in the bundle (GPU)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--share", action="store_true", help="public link (e.g. when running on Kaggle)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)
    build(args.bundle, args.live, args.model).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
