"""Export the recorded demo as one JSON file for the static web app (deploy/web/).

    python -m trialguide.demo.export_web --bundle demo/demo_bundle.json --out deploy/web/data.json

The web app has no server: it replays TrialGuide from this file. For each recorded patient we walk every
path a doctor can take (7 possible answers per question, up to the 3-question budget) with the same
DemoSession the Gradio demo uses, and store each state of the conversation. To keep the file small, a
state stores only what changed from the state before it: the rules whose status or decision changed,
the doctor's sentence it added, and the turn that led to it. The app applies the changes along the path. Nothing is computed in the browser except
looking states up, so the web app shows exactly what the Python system decided.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trialguide.demo.record import MissingRecording, ReplayChecker, load_bundle, patient_note, patient_rules
from trialguide.demo.session import DemoSession
from trialguide.pipeline import ACCEPTED
from trialguide.schemas import ActionType, Certainty, RuleKind, RuleStatus

# answer code -> (value, certainty); the codes form the path keys, e.g. "ys/nd"
ANSWERS = {"ys": ("yes", Certainty.SURE), "yp": ("yes", Certainty.PROBABLY), "yu": ("yes", Certainty.UNSURE),
           "ns": ("no", Certainty.SURE), "np": ("no", Certainty.PROBABLY), "nu": ("no", Certainty.UNSURE),
           "nd": ("", Certainty.NOT_DOCUMENTED)}
FIRST_PATIENT = "sigir-201529"     # clean questions; the others open with the copied "prior chemotherapy" key


def rule_class(session: DemoSession, ref, action) -> str:
    """How a rule ended up, as the app colours it: stated / inferred / answered / against / open."""
    c = session.checks[ref]
    if action.action not in ACCEPTED:
        return "open"
    against = (c.kind == RuleKind.INCLUSION and c.status == RuleStatus.NOT_MET) or \
              (c.kind == RuleKind.EXCLUSION and c.status == RuleStatus.MET)
    if against:
        return "against"
    if any(i in session.doctor_certainty for i in c.evidence_ids):
        return "answered"
    return "inferred" if action.action == ActionType.ACCEPT_INFERENCE else "stated"


def rule_states(session: DemoSession) -> dict:
    """'trial|rule' -> [status, category, decision, confidence, evidence ids, class]."""
    acts = session.action_of()
    out = {}
    for ref in session.rules:
        c, a = session.checks[ref], acts[ref]
        out[f"{ref[0]}|{ref[1]}"] = [c.status.value, c.category.value, a.action.value, round(c.confidence, 3),
                                    list(c.evidence_ids), rule_class(session, ref, a)]
    return out


def question(session: DemoSession):
    q = session.question
    if q is None or session.finished:
        return None
    return {"text": q.text, "fact": q.text.replace("What does the patient's record say about ", "").rstrip("?"),
            "turn": q.turn, "followup": q.is_followup, "rules": len(q.rule_refs),
            "trials": sorted({t for t, _ in q.rule_refs}), "score": round(q.score or 0.0, 3)}


def node(session: DemoSession, parent_rules: dict, gold: dict, known_ids: set) -> dict:
    """One state of the conversation, storing only the rules that differ from the state before it."""
    rules = rule_states(session)
    counts = {t: session.counts(t) for t in session.trial_ids()}
    ours, _ = session.ndcg(gold)
    added = {i: s for i, s in session.note.sentences.items() if i not in known_ids}
    last = session.turns[-1] if session.turns else None
    return {
        "q": question(session),
        "done": bool(session.finished or session.question is None),
        "scores": {t: round(v, 4) for t, v in session.scores().items()},
        "counts": {t: [c["stated"], c["inferred"], c["answered"], c["open"]] for t, c in counts.items()},
        "ndcg": round(ours, 4),
        "held": session.inference_counts().get("held_back", 0),
        "rules": {k: v for k, v in rules.items() if parent_rules is None or parent_rules.get(k) != v},
        "added": {str(i): s for i, s in added.items()},
        "turn": None if last is None else {"question": last.question, "answer": last.answer,
                                           "certainty": last.certainty, "settled": last.settled},
    }


def export_patient(pid: str, p: dict, store: dict, meta: dict, p_answer) -> dict:
    gold = {t["trial_id"]: int(t.get("gold", 0)) for t in p["trials"]}
    threshold = float(meta.get("threshold", 0.9))
    root = DemoSession(patient_note(p, pid), patient_rules(p), ReplayChecker(store), threshold=threshold,
                       p_answer=p_answer)
    root.start()
    known = set(root.note.sentences)
    nodes, missing = {"": node(root, None, gold, known)}, 0     # the root keeps every rule

    def walk(session: DemoSession, key: str, seen: set):
        nonlocal missing
        if session.finished or session.question is None:
            return
        for code, (value, certainty) in ANSWERS.items():
            child = session.fork()
            try:
                child.answer(value, certainty)
            except MissingRecording:
                missing += 1
                continue
            child_key = f"{key}/{code}" if key else code
            child_known = seen | set(session.note.sentences)
            nodes[child_key] = node(child, rule_states(session), gold, child_known)
            walk(child, child_key, child_known | set(child.note.sentences))

    walk(root, "", known)
    _, b6 = root.ndcg(gold)
    return {
        "id": pid, "cohort": p["cohort"],
        "visible": p["visible"], "hidden": p["hidden"],
        "trials": [{"id": t["trial_id"], "title": t.get("title", ""), "gold": int(t.get("gold", 0)),
                    "rules": [{"id": r["rule_id"], "kind": r["kind"], "text": r["text"]} for r in t["rules"]]}
                   for t in p["trials"]],
        "b6": {"scores": {t: round(v, 4) for t, v in root.b6_scores().items()}, "ndcg": round(b6, 4)},
        "nodes": nodes, "missing_paths": missing,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", type=Path, default=Path("demo/demo_bundle.json"))
    ap.add_argument("--out", type=Path, default=Path("deploy/web/data.json"))
    args = ap.parse_args(argv)
    patients, store, meta = load_bundle(args.bundle)
    p_answer = None
    if meta.get("prior") and Path(meta["prior"]).exists():
        from trialguide.selector.answerability import load
        p_answer = load(Path(meta["prior"]))
    order = sorted(patients, key=lambda p: (p != FIRST_PATIENT, p))
    out = {"meta": {"model": meta.get("model", "Qwen/Qwen3-8B"), "threshold": meta.get("threshold"),
                    "mask": meta.get("mask"), "budget": 3},
           "patients": [export_patient(pid, patients[pid], store, meta, p_answer) for pid in order]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    for p in out["patients"]:
        print(f'{p["id"]}: {len(p["nodes"])} states, {p["missing_paths"]} unrecorded answers')
    print(f"{args.out}: {args.out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
