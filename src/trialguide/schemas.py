"""Shared data formats used by every TrialGuide component (task T42).

Every part of the system (data, checker, tools, doctor, selector, policy, eval)
passes these objects to each other, so all components agree on one shape.
Change this file only through a reviewed pull request and a decision-log entry,
because a change here can break everyone's code.

Plain dataclasses and string enums are used on purpose:
- no extra dependency;
- `dataclasses.asdict(obj)` turns any object into a dict that `json.dumps` can write,
  which is how runs are logged.

Terms follow the proposal:
- Section 3 for the four information categories and the harmful-inference rule;
- Section 5 for the decision policy and the question loop;
- decisions D8, D10 and D11 in docs/decision_log.md for the conversation and the doctor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

# Default number of doctor questions per patient (proposal RQ1, decision D3).
QUESTION_BUDGET = 3


# ---------------------------------------------------------------------------
# Enums: fixed vocabularies. They inherit from str so they print and serialise
# as plain text ("met", "sure", ...) in logs and JSON files.
# ---------------------------------------------------------------------------

class Split(str, Enum):
    """Which patient split a record belongs to (decision D1)."""
    DEV = "dev"
    CAL = "cal"
    TEST = "test"


class MaskType(str, Enum):
    """How facts were hidden from the note (proposal Section 6)."""
    NONE = "none"          # full note, used only for B1 (oracle) and the full-note reference
    RANDOM = "random"
    RULE = "rule"          # facts some candidate trial rule asks about
    EVIDENCE = "evidence"  # exactly the expert-cited evidence sentences


class RuleKind(str, Enum):
    """Whether a trial rule is an inclusion or an exclusion criterion."""
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"


class RuleStatus(str, Enum):
    """Whether the condition written in the rule holds for this patient.

    MET means the condition in the rule text is true for the patient:
    - for an inclusion rule, MET = TrialGPT "included";
    - for an exclusion rule, MET = TrialGPT "excluded" (the patient has the excluded condition).
    """
    MET = "met"
    NOT_MET = "not_met"
    NOT_ENOUGH_INFO = "not_enough_info"
    NOT_APPLICABLE = "not_applicable"


class InfoCategory(str, Enum):
    """The four information categories (proposal Section 3, Table 2).

    Only STATED and INFERRED may ever be used as facts.
    """
    STATED = "stated"          # the note says it directly
    INFERRED = "inferred"      # follows from stated facts by a rule or calculation a clinician would accept
    ASSUMPTION = "assumption"  # plausible, but the note does not establish it; never used as a fact
    MISSING = "missing"        # nothing in the note bears on it


class Certainty(str, Enum):
    """How sure the doctor is about an answer (decision D11).

    How the policy treats each level:
    - SURE: used as a stated fact.
    - PROBABLY: treated like an inference and goes through risk control.
    - UNSURE: may trigger one follow-up question (it counts toward the budget).
    - NOT_DOCUMENTED: the record does not contain the fact; the rule stays unresolved.
    """
    SURE = "sure"
    PROBABLY = "probably"
    UNSURE = "unsure"
    NOT_DOCUMENTED = "not_documented"


class ActionType(str, Enum):
    """What the policy decides for one rule (proposal Section 5)."""
    USE_STATED = "use_stated"              # the note (or a sure doctor answer) settles it
    ACCEPT_INFERENCE = "accept_inference"  # inference accepted, passed risk control
    ASK = "ask"                            # becomes a candidate question for the selector
    ABSTAIN = "abstain"                    # left unresolved and reported as such


class FactSource(str, Enum):
    """Where a fact came from, so every fact can be traced in the logs."""
    NOTE = "note"      # a sentence of the visible note
    DOCTOR = "doctor"  # an answer from the (simulated or human) doctor
    TOOL = "tool"      # computed by a tool, e.g. eGFR


# ---------------------------------------------------------------------------
# Patient side
# ---------------------------------------------------------------------------

@dataclass
class VisibleNote:
    """The part of a patient note the system is allowed to see.

    Only this object may leave the data/masking module. Hidden sentences are kept
    by the masking code and are never put in here (leakage checklist).
    """
    patient_id: str
    split: Split
    sentences: dict[int, str]       # sentence id -> sentence text, ids as in the original note
    mask_type: MaskType = MaskType.NONE
    mask_fraction: float = 0.0      # share of facts hidden (0.2 / 0.4 / 0.6 for random masks)
    mask_seed: Optional[int] = None

    def text(self) -> str:
        """Join the visible sentences in their original order, each with its id, as TrialGPT does."""
        return "\n".join(f"{i}. {s}" for i, s in sorted(self.sentences.items()))


@dataclass
class Fact:
    """One piece of patient information the system may use."""
    fact_id: str
    patient_id: str
    text: str
    source: FactSource
    sentence_id: Optional[int] = None      # set when source is NOTE
    certainty: Optional[Certainty] = None  # set when source is DOCTOR
    question_id: Optional[str] = None      # set when source is DOCTOR


# ---------------------------------------------------------------------------
# Trial side and rule checking
# ---------------------------------------------------------------------------

@dataclass
class TrialRule:
    """One eligibility criterion of one trial."""
    trial_id: str
    rule_id: str          # unique within the trial, e.g. "inc_3" or "exc_1"
    kind: RuleKind
    text: str
    rule_type: Optional[str] = None  # demographic / lab value / diagnosis / prior treatment / comorbidity (T48)


@dataclass
class ToolCall:
    """A tool use inside a rule check, kept so every computed value can be traced."""
    name: str              # e.g. "egfr_ckd_epi_2021", "bmi", "rxclass"
    inputs: dict           # e.g. {"age": 62, "sex": "female", "creatinine_mg_dl": 1.9}
    output: Optional[str]  # None if an input was missing; the value is then missing, not guessed
    unit: Optional[str] = None


@dataclass
class RuleCheck:
    """The checker's output for one (patient, trial, rule) triple (task T12)."""
    patient_id: str
    trial_id: str
    rule_id: str
    kind: RuleKind
    status: RuleStatus
    category: InfoCategory
    evidence_ids: list[int] = field(default_factory=list)  # visible-note sentence ids it cites
    confidence: float = 1.0       # share of the 5 samples that agree on the status, 0..1
    n_samples: int = 5
    explanation: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    fact_key: Optional[str] = None  # short name of the missing fact, e.g. "prior_chemotherapy";
                                    # rules sharing a fact_key across trials merge into one question

    def __post_init__(self) -> None:
        # Basic sanity checks so a malformed checker output fails early, not in the results
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.category in (InfoCategory.STATED, InfoCategory.INFERRED) and not (self.evidence_ids or self.tool_calls):
            raise ValueError("a stated or inferred rule must cite evidence sentences or a tool call")

    @property
    def unresolved(self) -> bool:
        """True when the note alone does not settle this rule."""
        return self.category in (InfoCategory.ASSUMPTION, InfoCategory.MISSING) or \
            self.status == RuleStatus.NOT_ENOUGH_INFO


# ---------------------------------------------------------------------------
# Harmful inference (proposal Section 3)
# ---------------------------------------------------------------------------

def favours_eligibility(kind: RuleKind, status: RuleStatus) -> bool:
    """True if this status pushes the patient toward being eligible.

    An inclusion rule that is MET, or an exclusion rule that is NOT_MET, makes the patient look eligible.
    """
    return (kind == RuleKind.INCLUSION and status == RuleStatus.MET) or \
           (kind == RuleKind.EXCLUSION and status == RuleStatus.NOT_MET)


def is_harmful(kind: RuleKind, predicted: RuleStatus, reference: RuleStatus) -> bool:
    """An accepted inference is harmful if it is wrong AND it moves the patient toward eligibility.

    Examples:
    - inclusion rule wrongly marked MET (reference NOT_MET or not established) -> harmful;
    - exclusion rule wrongly marked NOT_MET -> harmful;
    - a wrong status that pushes toward exclusion is an error, but not a harmful one
      (it is counted separately as a possible false exclusion).
    """
    return predicted != reference and favours_eligibility(kind, predicted)


# ---------------------------------------------------------------------------
# Conversation with the doctor (decisions D8, D10, D11)
# ---------------------------------------------------------------------------

@dataclass
class Question:
    """One question TrialGuide asks the doctor. One question per turn."""
    question_id: str
    patient_id: str
    fact_key: str                 # the missing fact this question asks about
    text: str                     # plain question, about one fact, not leading
    turn: int                     # 1, 2 or 3
    rule_refs: list[tuple[str, str]] = field(default_factory=list)  # (trial_id, rule_id) pairs it can settle
    score: float = 0.0            # Eq. 1 score at the time it was chosen
    is_followup: bool = False     # True when it follows up an "unsure" answer


@dataclass
class DoctorAnswer:
    """The doctor's reply to one question (simulated in tests, a person in the demo)."""
    question_id: str
    value: str                    # e.g. "yes", "no", "1.9 mg/dL", or "" when not documented
    certainty: Certainty
    quote: Optional[str] = None   # sentence from the record that supports the answer (D10)
    raw_text: str = ""            # the doctor's full reply, kept for the hand checks

    def __post_init__(self) -> None:
        # D10: no quote means "not documented", and "not documented" carries no quote
        if self.certainty == Certainty.NOT_DOCUMENTED and self.quote:
            raise ValueError("a 'not documented' answer must not carry a quote")
        if self.certainty != Certainty.NOT_DOCUMENTED and not self.quote:
            raise ValueError("an answer without a supporting quote must be 'not documented'")


class Doctor(Protocol):
    """Anything that can answer TrialGuide's questions.

    The simulated doctor (Llama-3.1-8B, T11) and the human answering in the demo (T40)
    both implement this one method, so the rest of the system doesn't care which it is.
    The doctor is stateless: TrialGuide keeps the conversation in ConversationState.
    """

    def answer(self, patient_id: str, question: Question) -> DoctorAnswer:
        """Answer a single question about one patient, using that patient's record."""
        ...


@dataclass
class Turn:
    """One question and its answer."""
    question: Question
    answer: DoctorAnswer


@dataclass
class ConversationState:
    """Everything TrialGuide remembers during one patient's conversation."""
    patient_id: str
    budget: int = QUESTION_BUDGET
    turns: list[Turn] = field(default_factory=list)
    learned_facts: list[Fact] = field(default_factory=list)

    def questions_left(self) -> int:
        """How many questions can still be asked (follow-ups count too)."""
        return self.budget - len(self.turns)

    def can_ask(self) -> bool:
        """True if the budget allows another question."""
        return self.questions_left() > 0

    def asked_fact_keys(self) -> set[str]:
        """Facts already asked about, so the selector never repeats a question
        (a follow-up on an 'unsure' answer is the one allowed exception)."""
        return {t.question.fact_key for t in self.turns}

    def add_turn(self, question: Question, answer: DoctorAnswer) -> None:
        """Record a question and its answer, refusing anything over the budget."""
        if not self.can_ask():
            raise RuntimeError(f"question budget of {self.budget} already used for {self.patient_id}")
        if question.patient_id != self.patient_id:
            raise ValueError("question belongs to a different patient")
        self.turns.append(Turn(question, answer))


# ---------------------------------------------------------------------------
# Policy output and ranking
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """The policy's decision for one rule (task T19)."""
    patient_id: str
    trial_id: str
    rule_id: str
    action: ActionType
    reason: str = ""              # short, human-readable reason, kept for the case studies
    confidence: Optional[float] = None
    threshold: Optional[float] = None  # the risk-control threshold used (lambda-hat), if any


@dataclass
class TrialScore:
    """One trial's place in a patient's ranking."""
    trial_id: str
    score: float
    rank: int  # 1 = top
