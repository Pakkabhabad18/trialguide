"""Blind check of the simulated doctor (D10, TA feedback of Oct 10): does a person reading the same
record give the same answer?

    # 1. make the sheet: 60 questions the systems asked, with the patient's full record
    python -m trialguide.eval.doctor_check sheet --turns results/dev60_run/turns_asking.jsonl \
        --out results/doctor_check

    # 2. a teammate fills in the two empty columns of doctor_check_sheet.csv WITHOUT looking at the
    #    key file: `answer` (the answer as written in the record, or empty) and `sentence_id`
    #    (the record sentence that answers it, or empty when the record does not say)

    # 3. score it
    python -m trialguide.eval.doctor_check score --out results/doctor_check

Why: the simulated doctor (Llama-3.1-8B) answers from the full record and must quote it, else its
answer becomes "not documented". Most of its answers are "not documented", so the check that
matters most is whether the record really does not say, or whether the doctor misses facts that are
there. The person answers blind: the sheet holds only the question and the record, never the
doctor's answer.

Sampling: unique (patient, question) pairs from the development runs, half that the doctor answered
and half it called "not documented" (seeded), because a plain random draw would be ~70% "not
documented" and say little about the answered ones. Results are reported per stratum.

Scores:
- documented agreement: both say the record answers it, or both say it does not; Cohen's kappa;
- for the questions both answered: same sentence cited, and same answer (normalised text, or the
  same number for numeric answers).
Writes results/doctor_check/doctor_check_scores.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

SHEET = "doctor_check_sheet.csv"
KEY = "doctor_check_key.json"            # the doctor's answers: the person filling the sheet must not open it
SCORES = "doctor_check_scores.json"


def unique_questions(turns_path: Path) -> list:
    """Every (patient, question text) the systems asked, once, with the doctor's answer."""
    seen, out = set(), []
    with open(turns_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for t in row["turns"]:
                q, a = t["question"], t["answer"]
                key = (row["patient_id"], q["text"])
                if key in seen:
                    continue
                seen.add(key)
                out.append({"patient_id": row["patient_id"], "question": q["text"],
                            "doctor_certainty": a["certainty"], "doctor_answer": a.get("value") or "",
                            "doctor_quote": a.get("quote")})
    return out


def sample(questions: list, n: int, seed: int) -> list:
    """Half answered, half "not documented" (or as many as exist), shuffled together."""
    rng = random.Random(seed)
    answered = [q for q in questions if q["doctor_certainty"] != "not_documented"]
    missing = [q for q in questions if q["doctor_certainty"] == "not_documented"]
    rng.shuffle(answered)
    rng.shuffle(missing)
    k = min(n // 2, len(answered))
    picked = answered[:k] + missing[:n - k]
    rng.shuffle(picked)
    return picked


def quote_sentence(quote, sentences: dict):
    """The record sentence the doctor's quote comes from (the one sharing most words), or None."""
    if not quote:
        return None
    words = set(re.findall(r"\w+", quote.lower()))
    best = max(sentences, key=lambda i: len(words & set(re.findall(r"\w+", sentences[i].lower()))))
    return best if words & set(re.findall(r"\w+", sentences[best].lower())) else None


def make_sheet(args) -> None:
    from trialguide.data.patients import load_patients
    from trialguide.schemas import Split

    patients = load_patients(args.raw, Split.DEV)
    picked = sample(unique_questions(args.turns), args.n, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    key = []
    with open(args.out / SHEET, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "patient_id", "question", "record", "answer", "sentence_id"])
        for i, q in enumerate(picked, 1):
            sents = patients[q["patient_id"]].sentences
            record = "\n".join(f"{sid}. {s}" for sid, s in sorted(sents.items()))
            w.writerow([i, q["patient_id"], q["question"], record, "", ""])
            key.append({"item": i, **q, "doctor_sentence_id": quote_sentence(q["doctor_quote"], sents)})
    (args.out / KEY).write_text(json.dumps(key, indent=1) + "\n")
    print(f"{len(picked)} questions written to {args.out / SHEET} "
          f"({sum(q['doctor_certainty'] != 'not_documented' for q in picked)} the doctor answered)")


def same_answer(a: str, b: str) -> bool:
    """Same answer after lower-casing and dropping punctuation, or the same leading number."""
    norm = lambda s: " ".join(re.findall(r"[a-z0-9.]+", (s or "").lower()))
    num = lambda s: (re.findall(r"\d+(?:\.\d+)?", s or "") or [None])[0]
    a_n, b_n = norm(a), norm(b)
    if a_n and b_n and (a_n in b_n or b_n in a_n):
        return True
    return num(a) is not None and num(a) == num(b)


def kappa(x: list, y: list) -> float:
    """Cohen's kappa for two raters' yes/no labels."""
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def score(args) -> dict:
    key = {k["item"]: k for k in json.loads((args.out / KEY).read_text())}
    with open(args.out / SHEET, encoding="utf-8") as f:
        filled = {int(r["item"]): r for r in csv.DictReader(f)}
    rows = []
    for item, k in sorted(key.items()):
        r = filled[item]
        person_says = bool(r["answer"].strip() or r["sentence_id"].strip())
        doctor_says = k["doctor_certainty"] != "not_documented"
        sid = r["sentence_id"].strip()
        rows.append({"item": item, "stratum": "answered" if doctor_says else "not_documented",
                     "person_documented": person_says, "doctor_documented": doctor_says,
                     "same_sentence": (doctor_says and person_says and sid.isdigit()
                                       and int(sid) == k["doctor_sentence_id"]),
                     "same_answer": doctor_says and person_says and same_answer(r["answer"], k["doctor_answer"]),
                     "question": k["question"], "doctor_answer": k["doctor_answer"], "person_answer": r["answer"]})
    both = [r for r in rows if r["person_documented"] and r["doctor_documented"]]
    out = {"items": len(rows),
           "documented_agreement": sum(r["person_documented"] == r["doctor_documented"] for r in rows) / len(rows),
           "documented_kappa": kappa([r["person_documented"] for r in rows], [r["doctor_documented"] for r in rows]),
           "doctor_not_documented_but_record_says": sum(r["person_documented"] and not r["doctor_documented"] for r in rows),
           "doctor_answered_but_record_does_not_say": sum(r["doctor_documented"] and not r["person_documented"] for r in rows),
           "both_answered": len(both),
           "same_sentence_when_both_answered": sum(r["same_sentence"] for r in both) / len(both) if both else None,
           "same_answer_when_both_answered": sum(r["same_answer"] for r in both) / len(both) if both else None,
           "per_item": rows}
    for stratum in ("answered", "not_documented"):
        s = [r for r in rows if r["stratum"] == stratum]
        out[f"agreement_{stratum}"] = (sum(r["person_documented"] == r["doctor_documented"] for r in s) / len(s)
                                       if s else None)
    (args.out / SCORES).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_item"}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet", help="sample the questions and write the blank sheet")
    s.add_argument("--turns", type=Path, default=Path("results/dev60_run/turns_asking.jsonl"))
    s.add_argument("--raw", type=Path, default=Path("data/raw"))
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=Path("results/doctor_check"))
    c = sub.add_parser("score", help="score the filled sheet against the doctor's answers")
    c.add_argument("--out", type=Path, default=Path("results/doctor_check"))
    args = ap.parse_args(argv)
    make_sheet(args) if args.cmd == "sheet" else score(args)


if __name__ == "__main__":
    main()
