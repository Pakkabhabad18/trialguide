"""Record a demo bundle on a GPU, so the demo can be replayed on any laptop.

    python -m trialguide.demo.build_bundle --manifest results/dev_run/manifest.json \
        --patients 3 --threshold 0.766 --out demo/demo_bundle.json

For each chosen development patient it masks the note (rule-targeted, 40%, as in the pilot),
runs the checker on every rule, and then walks every answer a person can give in the demo
(yes / no with each certainty, and "not documented") for up to three questions, recording
every checker call on the way. The number of new calls stays small because only the rules
a question can settle are re-checked. Development patients only. Owner: Prakash.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trialguide.demo.record import RecordingChecker, save_bundle
from trialguide.demo.session import DemoSession
from trialguide.schemas import Certainty

# Every answer the demo offers (value, certainty)
ANSWERS = [("yes", Certainty.SURE), ("no", Certainty.SURE), ("yes", Certainty.PROBABLY),
           ("no", Certainty.PROBABLY), ("yes", Certainty.UNSURE), ("no", Certainty.UNSURE),
           ("", Certainty.NOT_DOCUMENTED)]


def explore(session: DemoSession, max_paths: int = 400) -> int:
    """Walk every answer path from this session; returns how many paths were walked."""
    walked, stack = 0, [session]
    while stack and walked < max_paths:
        s = stack.pop()
        if s.finished:
            walked += 1
            continue
        for value, certainty in ANSWERS:
            twin = s.fork()
            twin.answer(value, certainty)
            stack.append(twin)
    return walked


def main(argv=None) -> None:
    from trialguide.data.masking import candidate_rules, mask_rule, rule_demand
    from trialguide.data.patients import load_patients
    from trialguide.pipeline import checker, rules_of_trial
    from trialguide.schemas import Split

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--patients", type=int, default=3)
    ap.add_argument("--fraction", type=float, default=0.4)
    ap.add_argument("--threshold", type=float, default=0.9, help="lambda used when recording (use lambda-hat)")
    ap.add_argument("--prior", type=Path, default=Path("results/answerability_prior.json"),
                    help="answerability prior used by B7 (frozen from dev turns)")
    ap.add_argument("--out", type=Path, default=Path("demo/demo_bundle.json"))
    args = ap.parse_args(argv)

    m = json.loads(args.manifest.read_text())
    patients = load_patients(args.raw, Split.DEV)                 # development patients only
    rec = RecordingChecker(checker(args.model, 1))                # tools on, as in B7
    p_answer = None
    if args.prior.exists():                                       # same selector as the final B7
        from trialguide.selector.answerability import load
        p_answer = load(args.prior)
    bundle = {}
    for pid in sorted(m["patients"])[: args.patients]:
        info, pt = m["patients"][pid], patients[pid]
        entry = {e["patient_id"]: e for e in json.load(
            open(args.raw / "trialgpt" / info["cohort"] / "retrieved_trials.json"))}[pid]
        trials = [(t, int(lab)) for lab in ("0", "1", "2") for t in entry.get(lab, [])
                  if t["NCTID"] in info["trials"][lab]]
        masked = mask_rule(pt, rule_demand(pt.sentences, candidate_rules(args.raw, pt.cohort, pid)), args.fraction)
        rules = [r for t, _ in trials for r in rules_of_trial(t)]
        session = DemoSession(masked.visible, rules, rec, threshold=args.threshold, p_answer=p_answer)
        session.start()
        paths = explore(session)
        bundle[pid] = {
            "cohort": pt.cohort, "split": "dev",
            "visible": {str(i): s for i, s in masked.visible.sentences.items()},
            "hidden": {str(i): pt.sentences[i] for i in masked.hidden_ids},
            "trials": [{"trial_id": t["NCTID"], "title": t.get("brief_title") or t.get("title", ""),
                        "gold": gold, "rules": [{"rule_id": r.rule_id, "kind": r.kind.value, "text": r.text}
                                                for r in rules_of_trial(t)]} for t, gold in trials],
        }
        print(f"{pid}: {len(rules)} rules, {paths} answer paths, {len(rec.store)} checks recorded so far", flush=True)
    save_bundle(args.out, bundle, rec.store, {"model": args.model, "threshold": args.threshold,
                                              "mask": f"rule{round(args.fraction * 100)}",
                                              "prior": str(args.prior) if p_answer else None})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
