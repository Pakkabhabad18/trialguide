"""Infer / ask / abstain policy (task T19).

For every rule the checker has looked at, the policy picks exactly one action:

- USE_STATED: the note says it directly (or the rule does not apply). Use it.
- ACCEPT_INFERENCE: the checker inferred it from evidence, and it is safe to use.
- ASK: the rule is unresolved; its missing fact becomes a question candidate.
  The question selector (T20) decides which candidate is actually asked.
- ABSTAIN: leave the rule unresolved and say so.

The one rule that protects patients (proposal Sections 3 and 5): an inference that
pushes a patient TOWARD eligibility is accepted only if its confidence is at least
the risk threshold lambda. A wrong inference of that kind is a harmful inference,
which risk control (T30) keeps at or below 5% on average.
An inference that pushes the patient away from eligibility cannot be harmful by our
definition (decision D15), so it is accepted. False exclusions are still counted
in the evaluation.

Doctor answers (decision D11) reach the policy as new numbered sentences appended
to the visible note by fact merging (T20). The checker then re-checks the affected
rules. If a re-checked rule cites a doctor sentence, its certainty matters:
- sure: treated like any note sentence;
- probably: the rule is treated as an inference, so the threshold applies;
- unsure: the sentence is not used as evidence (one follow-up question is allowed);
- not documented: nothing is appended, and the fact is never asked again.

One PolicyConfig with three switches gives the baselines and ablations that differ
only in policy (see SYSTEMS at the bottom), so they all share the same code path.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, Optional

from trialguide.schemas import (
    Action, ActionType, Certainty, DoctorAnswer, InfoCategory, Question,
    RuleCheck, RuleStatus, favours_eligibility,
)


@dataclass(frozen=True)
class PolicyConfig:
    """The three switches that separate our systems.

    allow_inference: may the system use inferred facts at all?
    allow_asking:    may unresolved rules become questions to the doctor?
    threshold:       minimum confidence for an inference that favours eligibility.
                     0.0 accepts every such inference (no risk control).
                     For B7 it is lambda-hat from calibration (T30).
    """
    name: str
    allow_inference: bool
    allow_asking: bool
    threshold: float = 0.0

    def __post_init__(self) -> None:
        # A threshold outside [0, 1] can't be compared with a confidence
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")


# ---------------------------------------------------------------------------
# Deciding one rule
# ---------------------------------------------------------------------------

def _effective_category(check: RuleCheck, doctor_certainty: dict[int, Certainty]) -> InfoCategory:
    """The category the policy should act on, after taking doctor certainty into account.

    doctor_certainty maps the sentence id of each appended doctor answer to its certainty.
    A rule that leans on an 'unsure' answer is not settled; one that leans on a
    'probably' answer is only an inference.
    """
    cited = [doctor_certainty[i] for i in check.evidence_ids if i in doctor_certainty]
    if Certainty.UNSURE in cited:
        return InfoCategory.MISSING
    if Certainty.PROBABLY in cited and check.category == InfoCategory.STATED:
        return InfoCategory.INFERRED
    return check.category


def _unresolved(check: RuleCheck, config: PolicyConfig, already_asked: set[str], reason: str) -> Action:
    """Ask about the missing fact if allowed and not asked before; otherwise abstain."""
    can_ask = config.allow_asking and check.fact_key and check.fact_key not in already_asked
    action = ActionType.ASK if can_ask else ActionType.ABSTAIN
    if not can_ask and config.allow_asking and check.fact_key in already_asked:
        reason += "; fact already asked"
    elif not can_ask and config.allow_asking and not check.fact_key:
        reason += "; no fact to ask about"
    return Action(check.patient_id, check.trial_id, check.rule_id, action,
                  reason=reason, confidence=check.confidence, threshold=config.threshold)


def decide(check: RuleCheck, config: PolicyConfig,
           doctor_certainty: Optional[dict[int, Certainty]] = None,
           already_asked: Optional[set[str]] = None) -> Action:
    """Choose one action for one rule. Every branch records a short reason for the logs."""
    doctor_certainty = doctor_certainty or {}
    already_asked = already_asked or set()

    def act(action_type: ActionType, reason: str) -> Action:
        # Small helper so each branch below stays one line
        return Action(check.patient_id, check.trial_id, check.rule_id, action_type,
                      reason=reason, confidence=check.confidence, threshold=config.threshold)

    # 1. A rule that cannot apply to this patient is settled (e.g. pregnancy rule, male patient)
    if check.status == RuleStatus.NOT_APPLICABLE:
        return act(ActionType.USE_STATED, "not applicable")

    category = _effective_category(check, doctor_certainty)
    decided = check.status in (RuleStatus.MET, RuleStatus.NOT_MET)

    # 2. Stated in the note (or by a sure doctor answer): use it
    if category == InfoCategory.STATED and decided:
        return act(ActionType.USE_STATED, "stated in the note")

    # 3. Inferred from evidence
    if category == InfoCategory.INFERRED and decided:
        if not config.allow_inference:
            return _unresolved(check, config, already_asked, "inference not allowed in this system")
        if not favours_eligibility(check.kind, check.status):
            # Pushes the patient out of the trial: cannot be a harmful inference (D15)
            return act(ActionType.ACCEPT_INFERENCE, "inference toward exclusion")
        if check.confidence >= config.threshold:
            return act(ActionType.ACCEPT_INFERENCE,
                       f"inference toward eligibility, confidence {check.confidence:.2f} >= {config.threshold:.2f}")
        return _unresolved(check, config, already_asked,
                           f"inference toward eligibility below threshold "
                           f"({check.confidence:.2f} < {config.threshold:.2f})")

    # 4. Assumption, missing, or the checker said "not enough information": never used as a fact
    return _unresolved(check, config, already_asked, f"unresolved ({category.value})")


def decide_all(checks: Iterable[RuleCheck], config: PolicyConfig,
               doctor_certainty: Optional[dict[int, Certainty]] = None,
               already_asked: Optional[set[str]] = None) -> list[Action]:
    """Apply decide() to every rule of a patient."""
    return [decide(c, config, doctor_certainty, already_asked) for c in checks]


# ---------------------------------------------------------------------------
# Question candidates and the end of the conversation
# ---------------------------------------------------------------------------

def question_candidates(actions: Iterable[Action], checks: Iterable[RuleCheck]) -> dict[str, list[tuple[str, str]]]:
    """Group ASK actions by the fact they need (proposal Section 5, step 4).

    Returns fact_key -> list of (trial_id, rule_id) it would settle. Rules in different
    trials that need the same fact become ONE candidate, so one question can settle
    several rules; the selector (T20) scores these candidates with Eq. 1.
    """
    fact_of = {(c.trial_id, c.rule_id): c.fact_key for c in checks}
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for a in actions:
        if a.action == ActionType.ASK:
            key = fact_of.get((a.trial_id, a.rule_id))
            if key:
                groups[key].append((a.trial_id, a.rule_id))
    return dict(groups)


def needs_follow_up(answer: DoctorAnswer, question: Question) -> bool:
    """An 'unsure' answer allows one follow-up question about the same fact (D11).

    A follow-up of a follow-up is not allowed, so the doctor is never pressed twice.
    The follow-up still uses one question from the budget (ConversationState enforces it).
    """
    return answer.certainty == Certainty.UNSURE and not question.is_followup


def finalize(actions: Iterable[Action]) -> list[Action]:
    """When the conversation ends, any question candidate not asked becomes an abstention."""
    out = []
    for a in actions:
        if a.action == ActionType.ASK:
            a = replace(a, action=ActionType.ABSTAIN, reason=a.reason + "; not asked (budget or no gain)")
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# Systems that differ only in policy (proposal Section 7, baselines and ablations)
# ---------------------------------------------------------------------------

def system_config(name: str, threshold: Optional[float] = None) -> PolicyConfig:
    """Return the policy settings for a baseline or ablation by its name.

    B5 and B7 need a threshold: B5 uses 0.9 unless a tuned value is given;
    B7 must be given lambda-hat from calibration (T30), so it fails loudly without one.
    Systems that differ elsewhere (B3, B4, A2, A3 in question selection; A4, A5 in
    evidence and tools; A7 in asking all questions at once) use B7's policy.
    """
    if name == "B6":   # TrialGPT-style abstention: never infers, never asks
        return PolicyConfig("B6", allow_inference=False, allow_asking=False)
    if name == "B2":   # infers with raw confidence, no risk control, no questions
        return PolicyConfig("B2", allow_inference=True, allow_asking=False, threshold=0.0)
    if name == "B5":   # plain confidence threshold, no questions
        return PolicyConfig("B5", allow_inference=True, allow_asking=False,
                            threshold=0.9 if threshold is None else threshold)
    if name == "A1":   # ablation: no inference, ask or abstain only
        return PolicyConfig("A1", allow_inference=False, allow_asking=True)
    if name == "A6":   # ablation: our system without risk control
        return PolicyConfig("A6", allow_inference=True, allow_asking=True, threshold=0.0)
    if name in ("B7", "B3", "B4", "A2", "A3", "A4", "A5", "A7"):
        if threshold is None:
            raise ValueError(f"{name} needs the calibrated threshold lambda-hat (T30)")
        return PolicyConfig(name, allow_inference=True, allow_asking=True, threshold=threshold)
    raise ValueError(f"unknown system: {name}")
