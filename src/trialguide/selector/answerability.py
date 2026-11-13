"""How likely the doctor can answer a question about a fact (the one change after the dev run).

Eq. 1 scores a fact by the expected change in the top 10 IF it is answered. On the 37 development
patients most answers were "not documented": the selector kept asking about facts many trials
need (prior chemotherapy was asked for every patient and answered twice) that the notes rarely
record. B7 now weighs Eq. 1 by the chance of an answer:

    Score'(f) = P(answer | f) * Score(f)

P(answer | f) is the share of development questions about the same (canonical) fact that the
doctor answered with anything but "not documented", smoothed toward the overall rate:

    P(answer | f) = (answered_f + M * overall) / (asked_f + M)

so a fact never asked on dev gets the overall rate, and only facts asked several times move far
from it. The table is estimated once from the development turns and frozen
(results/answerability_prior.json); calibration and test patients never enter it.

Owner: Prateek (T20).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from trialguide.selector.selector import canonical_fact_key

M = 2.0                      # prior strength, in questions
PAnswer = Callable[[str], float]


def estimate(turn_files: Iterable[Path]) -> dict:
    """The prior from turns.jsonl files of development runs. Each (patient, fact) counts once."""
    answered = {}
    for path in turn_files:
        for line in open(path):
            row = json.loads(line)
            for turn in row["turns"]:
                key = (row["patient_id"], canonical_fact_key(turn["question"]["fact_key"]))
                answered[key] = turn["answer"]["certainty"] != "not_documented"
    facts = defaultdict(lambda: [0, 0])
    for (_, fact), ok in answered.items():
        facts[fact][0] += 1
        facts[fact][1] += int(ok)
    overall = sum(answered.values()) / len(answered) if answered else 1.0
    return {"overall": overall, "m": M, "pairs": len(answered),
            "facts": {f: {"asked": n, "answered": a} for f, (n, a) in sorted(facts.items())}}


def p_answer(prior: dict) -> PAnswer:
    """P(answer | fact) from an estimated prior."""
    overall, m = prior["overall"], prior["m"]
    facts = prior["facts"]

    def p(fact_key: str) -> float:
        f = facts.get(canonical_fact_key(fact_key))
        if f is None:
            return overall
        return (f["answered"] + m * overall) / (f["asked"] + m)
    return p


def load(path: Path) -> PAnswer:
    return p_answer(json.loads(Path(path).read_text()))
