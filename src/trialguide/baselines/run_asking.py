"""Run the systems that ask the doctor (B7, B3, B4, A1, A2, A3, A5, A6) on one GPU, in rounds.

The checker (Qwen) and the simulated doctor (Llama) do not fit on one 12 GB GPU together, so
each round loads ONE of them (LLM_ONLY_MODEL) and replays every patient's conversation from the
start. Every model call is cached and decoding is greedy, so a replay repeats the earlier steps
exactly and continues until it needs the other model; that patient then waits for the next
round. Rounds alternate until every conversation is finished (run_asking.sh drives them).

    LLM_ONLY_MODEL=Qwen/Qwen3-8B python -m trialguide.baselines.run_asking --manifest M --out D

Each call prints how many conversations finished and writes {out}/status.json; when all are
finished it writes {out}/summary.json (NDCG@10, questions, answers, per patient) and the turns.
Uses the same patients, trials and rule-targeted masks as the B6 runs; only masked notes,
since on the full note nothing is hidden. The doctor reads the full note.
Owner of this file: Prateek (T22, T27).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHECKER_MODEL = "Qwen/Qwen3-8B"
# system -> (policy name, question strategy, question budget, checker uses the tools,
#            Eq. 1 weighted by the chance of an answer)
SYSTEMS = {
    "B7": ("B7", "ranking", 3, True, True),        # TrialGuide: answerability-weighted Eq. 1
    "B7-eq1": ("B7", "ranking", 3, True, False),   # B7 as first run on dev: plain Eq. 1
    "A1": ("A1", "ranking", 3, True, True),        # no inference: ask or abstain only
    "A2": ("A2", "fixed", 3, True, False),         # no adaptive selection: fact of the best-ranked trial first
    "A3": ("A3", "uncertainty", 3, True, False),   # no ranking term: least confident fact first
    "A5": ("A5", "ranking", 3, False, True),       # no tools in the checker
    "A6": ("A6", "ranking", 3, True, True),        # no risk control: every inference accepted
    "B4": ("B4", "fixed", 3, True, False),         # fixed-order questions (same strategy as A2 here)
    "B3": ("B3", "all", 50, True, False),          # ask every candidate fact
}
LABELS = ("0", "1", "2")


def to_json(obj):
    if dataclasses.is_dataclass(obj):
        return {k: to_json(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    return getattr(obj, "value", obj)


def main():
    from trialguide.data.masking import candidate_rules, mask_rule, rule_demand
    from trialguide.data.patients import load_patients
    from trialguide.eval.ranking import per_patient_metrics
    from trialguide.llm.client import ONLY_MODEL, ModelNotLoaded
    from trialguide.pipeline import checker, rules_of_trial, run_patient
    from trialguide.policy.policy import system_config
    from trialguide.schemas import Split
    from trialguide.simulator.doctor import SimulatedDoctor

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--allow-test", action="store_true", help="needed for a test-split manifest")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--systems", nargs="+", default=["B7", "A2", "A3"])
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="lambda for the risk check; a stand-in until calibration (T30)")
    ap.add_argument("--prior", type=Path, default=Path("results/answerability_prior.json"),
                    help="P(answer | fact), estimated on dev turns (selector/answerability.py)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("outputs/asking"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    m = json.loads(args.manifest.read_text())
    split = Split(m.get("split", "dev"))
    if split == Split.TEST and not args.allow_test:
        raise SystemExit("the test split is opened once, after the system is frozen: pass --allow-test")
    patients = load_patients(args.raw, split, allow_test=args.allow_test)
    checks = {True: checker(CHECKER_MODEL, n_samples=1, tools=True),
              False: checker(CHECKER_MODEL, n_samples=1, tools=False)}
    tool_settings = sorted({SYSTEMS[s][3] for s in args.systems})

    # Same masked notes as the B6 runs. The doctor reads the whole record (TA feedback): it can
    # answer anything the full note says, and "not documented" means the note never said it.
    notes, record, rules, gold = {}, {}, {}, {}
    for pid, info in m["patients"].items():
        pt = patients[pid]
        masked = mask_rule(pt, rule_demand(pt.sentences, candidate_rules(args.raw, pt.cohort, pid)),
                           m["fraction"])
        notes[pid] = masked.visible
        record[pid] = [pt.sentences[i] for i in sorted(pt.sentences)]
        entry = {e["patient_id"]: e for e in json.load(
            open(args.raw / "trialgpt" / info["cohort"] / "retrieved_trials.json"))}[pid]
        trials = [t for lab in LABELS for t in entry.get(lab, []) if t["NCTID"] in info["trials"][lab]]
        gold[pid] = {t["NCTID"]: int(lab) for lab in LABELS for t in entry.get(lab, [])
                     if t["NCTID"] in info["trials"][lab]}
        rules[pid] = [r for t in trials for r in rules_of_trial(t)]
    p_answer = None
    if any(SYSTEMS[s][4] for s in args.systems):
        from trialguide.selector.answerability import load
        p_answer = load(args.prior)
    doctor = SimulatedDoctor(record, log_path=None)          # turns are saved below instead

    # First check of every rule, all at once so the GPU batches them (cache hits in later rounds)
    if ONLY_MODEL in ("", CHECKER_MODEL):
        jobs = [(t, pid, r) for t in tool_settings for pid in notes for r in rules[pid]]
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(lambda job: checks[job[0]](notes[job[1]], job[2]), jobs))

    def run(job):
        system, pid = job
        policy, strategy, budget, tools, weighted = SYSTEMS[system]
        config = system_config(policy, args.threshold)
        return run_patient(notes[pid], rules[pid], config, checks[tools], doctor=doctor,
                           strategy=strategy, budget=budget, p_answer=p_answer if weighted else None)

    jobs = [(s, pid) for s in args.systems for pid in notes]
    results, waiting = {}, []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {job: pool.submit(run, job) for job in jobs}
        for job, future in futures.items():
            try:
                results[job] = future.result()
            except ModelNotLoaded as needed:
                waiting.append({"system": job[0], "patient": job[1], "needs": str(needed)})

    status = {"model_this_round": ONLY_MODEL or "both", "finished": len(results),
              "total": len(jobs), "waiting": waiting}
    (args.out / "status.json").write_text(json.dumps(status, indent=1))
    print(f"round with {status['model_this_round']}: {len(results)}/{len(jobs)} conversations finished",
          flush=True)
    if waiting:
        return

    summary = {"checker": CHECKER_MODEL, "doctor": doctor.model, "threshold": args.threshold,
               "threshold_note": "stand-in until calibration (T30)", "mask": "rule",
               "fraction": m["fraction"], "manifest": str(args.manifest), "systems": {}}
    with open(args.out / "turns.jsonl", "w") as f:
        for system in args.systems:
            rankings = {pid: results[(system, pid)].scores for pid in notes}
            metrics = per_patient_metrics(rankings, gold)
            turns = {pid: results[(system, pid)].turns for pid in notes}
            answers = [a.certainty.value for t in turns.values() for _, a in t]
            summary["systems"][system] = {
                "policy": SYSTEMS[system][0], "strategy": SYSTEMS[system][1], "budget": SYSTEMS[system][2],
                "tools": SYSTEMS[system][3], "answerability": SYSTEMS[system][4],
                "patients": len(metrics),
                "NDCG@10": sum(v["ndcg"] for v in metrics.values()) / len(metrics) if metrics else None,
                "P@10": sum(v["p10"] for v in metrics.values()) / len(metrics) if metrics else None,
                "questions_per_patient": sum(len(t) for t in turns.values()) / len(turns),
                "answers": {c: answers.count(c) for c in sorted(set(answers))},
                "per_patient": metrics,
            }
            (args.out / f"rankings_{system}.json").write_text(json.dumps(rankings, indent=1))
            for pid, t in turns.items():
                f.write(json.dumps({"system": system, "patient_id": pid,
                                    "turns": [{"question": to_json(q), "answer": to_json(a)} for q, a in t]}) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({s: {k: v[k] for k in ("NDCG@10", "questions_per_patient", "answers")}
                      for s, v in summary["systems"].items()}, indent=2))


if __name__ == "__main__":
    main()
