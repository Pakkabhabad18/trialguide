"""TrialGuide end to end for one patient: check rules, decide, ask, re-check, score trials.

    visible note --checker (T12)--> RuleChecks --policy (T19)--> actions
        if the system may ask: selector (T20) picks one question -> doctor (T11) answers
        -> answer appended to the note (fact merging) -> affected rules re-checked -> repeat
    when no question is left: unasked rules abstain; rules the policy did not accept count as
    "not enough information"; each trial gets TrialGPT's linear score (selector.trial_score).

Every system is this one loop with different settings (proposal Section 7):
    B6  policy B6 (no inference, no asking)            B2  policy B2 (inference, no risk control)
    B5  policy B5 (confidence threshold)               B7  policy B7 (lambda-hat), strategy "ranking"
    B3  B7 policy, strategy "all" with a large budget  B4  B7 policy, strategy "fixed"
The checker and the doctor are passed in, so tests run without a model.
Owner of this file: Prateek (T16, T22; shared runner for the baselines).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from trialguide.policy.policy import (
    PolicyConfig, decide_all, finalize, needs_follow_up, question_candidates,
)
from trialguide.schemas import (
    ActionType, Certainty, ConversationState, Doctor, InfoCategory, RuleCheck, RuleKind,
    RuleStatus, TrialRule, VisibleNote,
)
from trialguide.selector.selector import (
    follow_up_question, merge_answer, merge_candidates, select_question, trial_score,
)

CheckFn = Callable[[VisibleNote, TrialRule], RuleCheck]
ACCEPTED = (ActionType.USE_STATED, ActionType.ACCEPT_INFERENCE)


def rules_of_trial(trial: dict) -> list[TrialRule]:
    """A TrialGPT corpus trial -> one TrialRule per criterion, split as the masking code does (T10)."""
    from trialguide.data.masking import split_criteria
    tid = trial["NCTID"]
    inc = split_criteria(trial.get("inclusion_criteria", ""))
    exc = split_criteria(trial.get("exclusion_criteria", ""))
    return ([TrialRule(tid, f"inc_{i}", RuleKind.INCLUSION, t) for i, t in enumerate(inc)]
            + [TrialRule(tid, f"exc_{i}", RuleKind.EXCLUSION, t) for i, t in enumerate(exc)])


def effective_checks(checks: list[RuleCheck], actions) -> list[RuleCheck]:
    """Rules the policy accepted keep their status; every other rule counts as not enough information."""
    accepted = {(a.trial_id, a.rule_id) for a in actions if a.action in ACCEPTED}
    return [c if (c.trial_id, c.rule_id) in accepted else
            replace(c, status=RuleStatus.NOT_ENOUGH_INFO, category=InfoCategory.MISSING,
                    evidence_ids=[], tool_calls=[])
            for c in checks]


@dataclass
class PatientResult:
    patient_id: str
    scores: dict                       # trial_id -> score
    checks: list                       # final RuleChecks as the checker returned them
    actions: list                      # final policy actions
    turns: list = field(default_factory=list)   # (question, answer) pairs
    n_checks: int = 0                  # checker calls, re-checks included


def run_patient(note: VisibleNote, rules: list[TrialRule], config: PolicyConfig, check: CheckFn,
                doctor: Optional[Doctor] = None, strategy: str = "ranking", budget: int = 3) -> PatientResult:
    """One patient through one system. `rules` are all rules of the patient's candidate trials."""
    by_ref = {(r.trial_id, r.rule_id): r for r in rules}
    checks = {ref: check(note, r) for ref, r in by_ref.items()}
    n_checks = len(checks)
    state = ConversationState(note.patient_id, budget=budget)
    doctor_certainty: dict[int, Certainty] = {}

    def decide():
        return decide_all(checks.values(), config, doctor_certainty, state.asked_fact_keys())

    actions = decide()
    question = None
    while config.allow_asking and doctor is not None and state.can_ask():
        if question is not None and needs_follow_up(state.turns[-1].answer, question):
            question = follow_up_question(question, state)
        else:
            candidates = merge_candidates(question_candidates(actions, checks.values()))
            question = select_question(candidates, effective_checks(list(checks.values()), actions),
                                       state, strategy=strategy)
        if question is None:
            break
        answer = doctor.answer(note.patient_id, question)
        state.add_turn(question, answer)
        note, sentence_id = merge_answer(note, question, answer)
        if sentence_id is not None:
            doctor_certainty[sentence_id] = answer.certainty
            for ref in question.rule_refs:          # re-check only the rules this answer can settle
                if ref in by_ref:
                    checks[ref] = check(note, by_ref[ref])
                    n_checks += 1
        actions = decide()

    actions = finalize(actions)
    final = effective_checks(list(checks.values()), actions)
    trials = sorted({r.trial_id for r in rules})
    scores = {t: trial_score(c for c in final if c.trial_id == t) for t in trials}
    return PatientResult(note.patient_id, scores, list(checks.values()), actions,
                         [(t.question, t.answer) for t in state.turns], n_checks)


def checker(model: str, n_samples: int = 1) -> CheckFn:
    """Vidur's rule checker (T12) with a fixed model. With one sample (greedy) the confidence is the
    model's probability of its status; with more, the vote share of the samples."""
    from trialguide.checker.rule_checker import check_rule
    mode = "logprob" if n_samples == 1 else "votes"
    return lambda note, rule: check_rule(note, rule, model, n_samples=n_samples, confidence=mode)
