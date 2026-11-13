"""One patient through TrialGuide, one step at a time, for the interactive demo.

This is the same loop as pipeline.run_patient, split into steps so a person can answer each
question in the browser:

    start()  -> checker on every rule, policy decides, the selector picks the first question
    answer() -> the answer is added to the note, only the affected rules are re-checked,
                the policy decides again, the next question is picked (or the loop stops)

The checker is passed in, so the demo can run on a live model or replay recorded checks.
Owner: Prakash.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Callable, Optional

from trialguide.pipeline import ACCEPTED, effective_checks
from trialguide.policy.policy import decide_all, finalize, needs_follow_up, question_candidates, system_config
from trialguide.schemas import (
    ActionType, Certainty, ConversationState, DoctorAnswer, InfoCategory, Question, RuleCheck,
    RuleKind, RuleStatus, TrialRule, VisibleNote,
)
from trialguide.selector.selector import (
    follow_up_question, merge_answer, merge_candidates, select_question, trial_score,
)

CheckFn = Callable[[VisibleNote, TrialRule], RuleCheck]


def answer_sentence(question: Question, value: str) -> str:
    """The doctor's answer as one sentence, e.g. 'Prior chemotherapy: no.' (also the quote)."""
    label = question.text.replace("What does the patient's record say about ", "").split("?")[0]
    return f"{label[:1].upper()}{label[1:]}: {value.strip().rstrip('.')}."


@dataclass
class Turn:
    question: str
    answer: str
    certainty: str
    settled: list          # (trial_id, rule_id, old status, new status) for rules the answer changed


class DemoSession:
    def __init__(self, note: VisibleNote, rules: list[TrialRule], check: CheckFn, threshold: float = 0.9,
                 strategy: str = "ranking", budget: int = 3, p_answer: Optional[Callable[[str], float]] = None):
        self.check = check
        self.note = note
        self.rules = {(r.trial_id, r.rule_id): r for r in rules}
        self.config = system_config("B7", threshold=threshold)
        self.strategy = strategy
        self.p_answer = p_answer        # B7: Eq. 1 weighted by the chance the doctor can answer (dev prior)
        self.state = ConversationState(note.patient_id, budget=budget)
        self.doctor_certainty: dict[int, Certainty] = {}
        self.checks: dict = {}
        self.initial_checks: dict = {}
        self.actions: list = []
        self.question: Optional[Question] = None
        self.turns: list[Turn] = []
        self.finished = False

    # ---- the loop ----

    def start(self) -> None:
        """Check every rule once, decide, and pick the first question."""
        self.checks = {ref: self.check(self.note, r) for ref, r in self.rules.items()}
        self.initial_checks = dict(self.checks)
        self._decide()
        self.question = self._pick(None)
        self.finished = self.question is None

    def answer(self, value: str, certainty: Certainty) -> None:
        """Take the doctor's answer to the current question and move on."""
        if self.question is None:
            raise RuntimeError("there is no open question")
        q = self.question
        if certainty == Certainty.NOT_DOCUMENTED:
            reply = DoctorAnswer(q.question_id, "", certainty)
        else:
            reply = DoctorAnswer(q.question_id, value, certainty, quote=answer_sentence(q, value))
        self.state.add_turn(q, reply)
        self.note, sid = merge_answer(self.note, q, reply)
        settled = []
        if sid is not None:
            self.doctor_certainty[sid] = certainty
            for ref in q.rule_refs:                    # re-check only the rules this answer can settle
                if ref in self.rules:
                    old = self.checks[ref].status
                    self.checks[ref] = self.check(self.note, self.rules[ref])
                    if self.checks[ref].status != old:
                        settled.append((*ref, old.value, self.checks[ref].status.value))
        self.turns.append(Turn(q.text, value if sid is not None else "not documented",
                               certainty.value, settled))
        self._decide()
        self.question = self._pick(q)
        self.finished = self.question is None

    def _decide(self) -> None:
        self.actions = decide_all(self.checks.values(), self.config, self.doctor_certainty,
                                  self.state.asked_fact_keys())

    def _pick(self, previous: Optional[Question]) -> Optional[Question]:
        if previous is not None and self.state.turns and needs_follow_up(self.state.turns[-1].answer, previous):
            return follow_up_question(previous, self.state)
        candidates = merge_candidates(question_candidates(self.actions, self.checks.values()))
        return select_question(candidates, effective_checks(list(self.checks.values()), self.actions),
                               self.state, strategy=self.strategy, p_answer=self.p_answer)

    def fork(self) -> "DemoSession":
        """An independent copy that shares the checker (used to pre-record answer paths)."""
        check, self.check = self.check, None
        twin = copy.deepcopy(self)
        self.check = twin.check = check
        return twin

    # ---- what the page shows ----

    def final_actions(self) -> list:
        return finalize(self.actions) if self.finished else self.actions

    def scores(self) -> dict:
        """TrialGuide's trial scores now (unasked questions count as abstentions)."""
        final = effective_checks(list(self.checks.values()), self.final_actions())
        return {t: trial_score(c for c in final if c.trial_id == t) for t in self.trial_ids()}

    def b6_scores(self) -> dict:
        """The same patient under plain abstention (B6): no inference, no questions."""
        b6 = system_config("B6")
        acts = finalize(decide_all(self.initial_checks.values(), b6))
        final = effective_checks(list(self.initial_checks.values()), acts)
        return {t: trial_score(c for c in final if c.trial_id == t) for t in self.trial_ids()}

    @staticmethod
    def order(scores: dict) -> list:
        """Trial ids best first; ties broken by id so the order is repeatable."""
        return sorted(scores, key=lambda t: (-scores[t], t))

    def ndcg(self, gold: dict, k: int = 10) -> tuple:
        """NDCG@k of TrialGuide's and B6's current ranking against the gold labels (0/1/2).
        None when no candidate trial has a positive label."""
        from trialguide.eval.ranking import ndcg_at_k
        ours = ndcg_at_k([gold.get(t, 0) for t in self.order(self.scores())], k)
        b6 = ndcg_at_k([gold.get(t, 0) for t in self.order(self.b6_scores())], k)
        return ours, b6

    def inference_counts(self) -> dict:
        """Inferences toward eligibility accepted vs held back by the risk threshold."""
        acts = self.action_of()
        out = {"accepted": 0, "held_back": 0}
        for ref, c in self.checks.items():
            if c.category == InfoCategory.INFERRED and c.status in (RuleStatus.MET, RuleStatus.NOT_MET):
                if acts[ref].action == ActionType.ACCEPT_INFERENCE:
                    out["accepted"] += 1
                else:
                    out["held_back"] += 1
        return out

    def to_record(self, gold: Optional[dict] = None) -> dict:
        """Everything about this conversation, for the saved session log."""
        ours = self.scores()
        rec = {"patient_id": self.note.patient_id, "threshold": self.config.threshold,
               "strategy": self.strategy, "budget": self.state.budget, "finished": self.finished,
               "turns": [{"question": t.question, "answer": t.answer, "certainty": t.certainty,
                          "rules_changed": [list(x) for x in t.settled]} for t in self.turns],
               "ranking": self.order(ours), "scores": ours,
               "b6_ranking": self.order(self.b6_scores()), "b6_scores": self.b6_scores(),
               "counts": {t: self.counts(t) for t in self.trial_ids()},
               "inferences": self.inference_counts(),
               "rules": [{"trial_id": t, "rule_id": r, "kind": c.kind.value, "status": c.status.value,
                          "category": c.category.value, "confidence": c.confidence,
                          "evidence_ids": c.evidence_ids, "decision": a.action.value, "reason": a.reason}
                         for (t, r), c in sorted(self.checks.items())
                         for a in [self.action_of()[(t, r)]]]}
        if gold:
            rec["ndcg@10"], rec["b6_ndcg@10"] = self.ndcg(gold)
        return rec

    def trial_ids(self) -> list:
        return sorted({t for t, _ in self.rules})

    def action_of(self) -> dict:
        return {(a.trial_id, a.rule_id): a for a in self.final_actions()}

    def counts(self, trial_id: str) -> dict:
        """How this trial's rules were handled: stated, inferred, answered by the doctor, open."""
        acts = self.action_of()
        out = {"stated": 0, "inferred": 0, "answered": 0, "open": 0}
        for (t, r), c in self.checks.items():
            if t != trial_id:
                continue
            a = acts[(t, r)].action
            if a not in ACCEPTED:
                out["open"] += 1
            elif any(i in self.doctor_certainty for i in c.evidence_ids):
                out["answered"] += 1
            elif a == ActionType.ACCEPT_INFERENCE:
                out["inferred"] += 1
            else:
                out["stated"] += 1
        return out

    def rule_rows(self, trial_id: str) -> list[dict]:
        """One row per rule of a trial, for the rules table."""
        acts = self.action_of()
        rows = []
        for (t, r), rule in sorted(self.rules.items()):
            if t != trial_id:
                continue
            c, a = self.checks[(t, r)], acts[(t, r)]
            evidence = " ".join(f"[{i}] {self.note.sentences.get(i, '')}" for i in c.evidence_ids)
            rows.append({"rule": rule.rule_id, "kind": rule.kind.value, "text": rule.text,
                         "status": c.status.value, "category": c.category.value,
                         "decision": a.action.value, "confidence": round(c.confidence, 2),
                         "evidence": evidence})
        return rows
