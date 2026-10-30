"""Question selection under a budget (task T20, proposal Section 5).

The policy (T19) turns unresolved rules into ASK actions and groups them by the
fact they need (policy.question_candidates: fact_key -> [(trial_id, rule_id)]).
This module decides which of those facts, if any, is worth one of the few
questions:

    Score(f) = sum_a p(a) * sum_t |s_t^{f=a} - s_t| / log2(1 + r_t)        (Eq. 1)

s_t is trial t's current score, r_t its current rank, s_t^{f=a} its score if the
answer to f were a. Changes near the top weigh more, and a fact shared by many
trials scores higher.

Choices the proposal leaves open (logged in the decision log):
- s_t is TrialGPT's linear matching score computed from rule statuses
  (% inclusion met, -1 if any inclusion is not met, -1 if any exclusion is met).
  TrialGPT's LLM aggregation score cannot be recomputed for every hypothetical
  answer without an LLM call each time.
- One fact can touch rules worded in opposite directions ("no prior
  chemotherapy" vs "prior chemotherapy") and the checker does not record that
  polarity, so each affected rule is modelled as becoming met or not met, with
  probability p_met (default 0.5). The rules of one trial are enumerated jointly.
- Facts are merged across trials by a canonical fact key (canonical_fact_key),
  because the checker names each rule's missing fact independently.

Stopping rule: ask until the budget is used or no candidate can change the
top 10. Other strategies give the comparison systems: "fixed" (B4, A2),
"uncertainty" (A3) and "all" (B3, no budget).
"""

from __future__ import annotations

import itertools
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Optional

from trialguide.schemas import (
    Certainty, ConversationState, DoctorAnswer, Question, RuleCheck, RuleKind,
    RuleStatus, TrialScore, VisibleNote,
)

TOP_K = 10
MAX_JOINT_RULES = 8  # rules of one trial enumerated jointly (2^8 = 256 outcomes)
EPS = 1e-9

# ---------------------------------------------------------------------------
# Trial scores and ranking
# ---------------------------------------------------------------------------

def trial_score(checks: Iterable[RuleCheck]) -> float:
    """TrialGPT's linear matching score from rule statuses (TrialGPT Eqs. 4-11, 14).

    Same formula as rank_results.get_matching_score in the TrialGPT baseline:
    share of inclusion rules met (not-applicable rules left out), minus 1 if any
    inclusion rule is not met, minus 1 if any exclusion rule is met.
    """
    met = not_met = no_info = 0
    excluded = False
    for c in checks:
        if c.kind == RuleKind.INCLUSION:
            if c.status == RuleStatus.MET:
                met += 1
            elif c.status == RuleStatus.NOT_MET:
                not_met += 1
            elif c.status == RuleStatus.NOT_ENOUGH_INFO:
                no_info += 1
        elif c.status == RuleStatus.MET:
            excluded = True
    score = met / (met + not_met + no_info) if met + not_met + no_info else 0.0
    if not_met:
        score -= 1
    if excluded:
        score -= 1
    return score


def rank(scores: dict[str, float]) -> list[TrialScore]:
    """Trials sorted by score, best first; ties broken by trial id so ranks are stable."""
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [TrialScore(trial_id=t, score=s, rank=i + 1) for i, (t, s) in enumerate(ordered)]


def _by_trial(checks: Iterable[RuleCheck]) -> dict[str, dict[str, RuleCheck]]:
    trials: dict[str, dict[str, RuleCheck]] = defaultdict(dict)
    for c in checks:
        trials[c.trial_id][c.rule_id] = c
    return trials


# ---------------------------------------------------------------------------
# Fact merging across trials
# ---------------------------------------------------------------------------

# Spelling variants the checker produces for the same fact; extend as dev runs show more
_SYNONYMS = {
    "chemo": "chemotherapy",
    "rt": "radiotherapy",
    "radiation": "radiotherapy",
    "radiation_therapy": "radiotherapy",
    "hep": "hepatitis",
    "hbv": "hepatitis_b",
    "hcv": "hepatitis_c",
    "hiv_status": "hiv",
    "pregnant": "pregnancy",
    "mi": "myocardial_infarction",
    "heart_attack": "myocardial_infarction",
    "egfr": "kidney_function",
    "renal_function": "kidney_function",
    "creatinine": "kidney_function",
    "ecog": "performance_status",
    "ecog_performance_status": "performance_status",
}
_DROP = {"history", "of", "the", "patient", "status", "any", "current", "known",
         "prior", "previous", "past"}


def canonical_fact_key(key: str) -> str:
    """One name per fact, so rules from different trials merge into one question.

    Lower-cases, maps known synonyms, drops filler words and sorts the rest, so
    'History of prior chemo', 'prior_chemotherapy' and 'chemotherapy_history'
    all become 'chemotherapy'. The result is for matching only; questions use
    the checker's own wording (Candidate.label).
    """
    key = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
    key = _SYNONYMS.get(key, key)
    words = []
    for w in key.split("_"):
        w = _SYNONYMS.get(w, w)
        if w and w not in _DROP:
            words.extend(w.split("_"))
    return "_".join(sorted(set(words))) or key


@dataclass
class Candidate:
    """One fact the system could ask about, with every rule its answer would settle."""
    fact_key: str
    rule_refs: list[tuple[str, str]]                      # (trial_id, rule_id)
    merged_keys: list[str] = field(default_factory=list)  # checker keys merged into this one

    @property
    def label(self) -> str:
        """Readable name for the question: the first checker key merged into this fact."""
        return (self.merged_keys[0] if self.merged_keys else self.fact_key).replace("_", " ")


def merge_candidates(groups: dict[str, list[tuple[str, str]]]) -> list[Candidate]:
    """Merge policy.question_candidates output by canonical fact key."""
    merged: dict[str, Candidate] = {}
    for key, refs in groups.items():
        canon = canonical_fact_key(key)
        cand = merged.setdefault(canon, Candidate(canon, []))
        cand.merged_keys.append(key)
        for ref in refs:
            if ref not in cand.rule_refs:
                cand.rule_refs.append(ref)
    return sorted(merged.values(), key=lambda c: c.fact_key)


# ---------------------------------------------------------------------------
# Eq. 1
# ---------------------------------------------------------------------------

PMet = Callable[[str, str], float]  # (trial_id, rule_id) -> probability the rule turns out met


def _outcomes(trial_checks: dict[str, RuleCheck], rule_ids: list[str], p_met: PMet, trial_id: str):
    """Yield (probability, score) for every joint met/not-met outcome of the rules."""
    rule_ids = rule_ids[:MAX_JOINT_RULES]
    for statuses in itertools.product((RuleStatus.MET, RuleStatus.NOT_MET), repeat=len(rule_ids)):
        prob = 1.0
        changed = dict(trial_checks)
        for rule_id, status in zip(rule_ids, statuses):
            p = p_met(trial_id, rule_id)
            prob *= p if status == RuleStatus.MET else 1 - p
            changed[rule_id] = replace(trial_checks[rule_id], status=status)
        yield prob, trial_score(changed.values())


def _affected(candidate: Candidate) -> dict[str, list[str]]:
    by_trial: dict[str, list[str]] = defaultdict(list)
    for trial_id, rule_id in candidate.rule_refs:
        by_trial[trial_id].append(rule_id)
    return by_trial


def score_candidate(candidate: Candidate, checks: Iterable[RuleCheck],
                    p_met: Optional[PMet] = None) -> float:
    """Eq. 1: expected rank-discounted change in trial scores if this fact were answered."""
    p_met = p_met or (lambda trial_id, rule_id: 0.5)
    trials = _by_trial(checks)
    current = {t: trial_score(rules.values()) for t, rules in trials.items()}
    ranks = {ts.trial_id: ts.rank for ts in rank(current)}

    total = 0.0
    for trial_id, rule_ids in _affected(candidate).items():
        if trial_id not in trials:
            continue
        rule_ids = [r for r in rule_ids if r in trials[trial_id]]
        expected_change = sum(
            prob * abs(score - current[trial_id])
            for prob, score in _outcomes(trials[trial_id], rule_ids, p_met, trial_id)
        )
        total += expected_change / math.log2(1 + ranks[trial_id])
    return total


def can_change_top_k(candidate: Candidate, checks: Iterable[RuleCheck], k: int = TOP_K) -> bool:
    """True if some answer could change which trials are in the top k, or their order.

    Each affected trial is varied on its own, the others kept at their current
    scores (stopping rule, proposal Section 5 step 5).
    """
    trials = _by_trial(checks)
    current = {t: trial_score(rules.values()) for t, rules in trials.items()}
    top = [ts.trial_id for ts in rank(current)[:k]]
    for trial_id, rule_ids in _affected(candidate).items():
        if trial_id not in trials:
            continue
        rule_ids = [r for r in rule_ids if r in trials[trial_id]]
        for _, score in _outcomes(trials[trial_id], rule_ids, lambda t, r: 0.5, trial_id):
            if abs(score - current[trial_id]) < EPS:
                continue
            new_top = [ts.trial_id for ts in rank({**current, trial_id: score})[:k]]
            if new_top != top:
                return True
    return False


# ---------------------------------------------------------------------------
# Choosing the next question
# ---------------------------------------------------------------------------

STRATEGIES = ("ranking", "fixed", "uncertainty", "all")


def question_text(label: str) -> str:
    """A plain, non-leading question about one fact (D8): it names the fact, not an answer."""
    return f"What does the patient's record say about {label}?"


def select_question(candidates: list[Candidate], checks: list[RuleCheck], state: ConversationState,
                    strategy: str = "ranking", p_met: Optional[PMet] = None,
                    k: int = TOP_K) -> Optional[Question]:
    """The next question to ask, or None to stop.

    strategy:
      ranking     - Eq. 1 (B7, our system); stops when no candidate can change the top k
      fixed       - fixed order: the fact of the best-ranked affected trial first (B4, A2)
      uncertainty - the fact whose rules the checker is least sure about (A3)
      all         - every fact in fixed order, until the budget runs out (B3: give it
                    a budget as large as the number of candidates)
    Facts already asked are never asked again.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    if not state.can_ask():
        return None

    asked = {canonical_fact_key(k_) for k_ in state.asked_fact_keys()}
    pool = [c for c in candidates if c.fact_key not in asked and c.rule_refs]
    if not pool:
        return None

    if strategy == "ranking":
        pool = [c for c in pool if can_change_top_k(c, checks, k)]
        if not pool:
            return None
        # highest score first; ties go to the alphabetically first fact, so runs are repeatable
        scored = sorted(((score_candidate(c, checks, p_met), c) for c in pool),
                        key=lambda item: (-item[0], item[1].fact_key))
        score, best = scored[0]
    elif strategy in ("fixed", "all"):
        current = {t: trial_score(r.values()) for t, r in _by_trial(checks).items()}
        ranks = {ts.trial_id: ts.rank for ts in rank(current)}
        best = min(pool, key=lambda c: (min(ranks.get(t, 10**9) for t, _ in c.rule_refs), c.fact_key))
        score = 0.0
    else:  # uncertainty
        conf = {(c.trial_id, c.rule_id): c.confidence for c in checks}
        def mean_conf(c: Candidate) -> float:
            values = [conf[r] for r in c.rule_refs if r in conf]
            return sum(values) / len(values) if values else 1.0
        best = min(pool, key=lambda c: (mean_conf(c), c.fact_key))
        score = 1.0 - mean_conf(best)

    turn = len(state.turns) + 1
    return Question(
        question_id=f"{state.patient_id}-q{turn}",
        patient_id=state.patient_id,
        fact_key=best.fact_key,
        text=question_text(best.label),
        turn=turn,
        rule_refs=list(best.rule_refs),
        score=score,
    )


def follow_up_question(previous: Question, state: ConversationState) -> Optional[Question]:
    """The one follow-up allowed after an 'unsure' answer (D11); it uses a budget slot."""
    if previous.is_followup or not state.can_ask():
        return None
    turn = len(state.turns) + 1
    return replace(previous, question_id=f"{state.patient_id}-q{turn}", turn=turn, is_followup=True,
                   text=f"{previous.text} Is there anything more definite in the record?")


# ---------------------------------------------------------------------------
# Merging an answer back into the note
# ---------------------------------------------------------------------------

def merge_answer(note: VisibleNote, question: Question, answer: DoctorAnswer
                 ) -> tuple[VisibleNote, Optional[int]]:
    """Append a documented answer to the visible note as a new numbered sentence.

    The checker then re-checks only the rules in question.rule_refs, and the
    policy reads the sentence's certainty through the returned sentence id.
    A 'not documented' answer adds nothing (the fact is never asked again).
    """
    if answer.certainty == Certainty.NOT_DOCUMENTED:
        return note, None
    sentence_id = max(note.sentences, default=-1) + 1
    text = f"Doctor's answer about {question.fact_key.replace('_', ' ')}: {answer.quote}"
    return replace(note, sentences={**note.sentences, sentence_id: text}), sentence_id
