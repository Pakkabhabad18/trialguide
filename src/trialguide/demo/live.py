"""Live demo for a new patient: any note, processed by the real models while you watch.

Unlike the recorded demo (record.py), nothing here is prepared in advance:

    1. A note comes in, either pasted / uploaded or built from a few facts typed by the user.
       It is cut into numbered sentences, as for the benchmark patients.
    2. BM25 searches the TrialGPT trial collections with the note and keeps the top trials.
       Each trial's eligibility text is split into rules, as in the experiments.
    3. The first model (Qwen3-8B, the checker of the experiments) checks every rule.
       The policy accepts what is stated and the inferences that pass the risk threshold.
    4. NEW in the live demo: a second model (Llama-3.1-8B) checks every rule the first model
       could not settle. If it settles a rule from the note, that is the decision, and the rule
       is tagged "decided by the second model". Our 5% risk result was measured for the first
       model with its threshold only, so these rules are shown separately.
    5. Only rules that neither model can settle become questions to the doctor (at most a few,
       one at a time, chosen by Eq. 1). Each answer is added to the note and the affected rules
       are re-checked by both models. The user can also add new information at any time.

Owner: Prakash.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

from trialguide.demo.session import DemoSession, Turn
from trialguide.pipeline import ACCEPTED
from trialguide.policy.policy import decide_all
from trialguide.schemas import (
    ActionType, InfoCategory, RuleKind, RuleStatus, Split, TrialRule, VisibleNote,
)

CheckFn = Callable[[VisibleNote, TrialRule], object]

FIRST_MODEL = "Qwen/Qwen3-8B"
SECOND_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
WORKERS = 8          # parallel checker calls; the HF backend batches them on the GPU


# ---------------------------------------------------------------------------
# 1. The patient's note
# ---------------------------------------------------------------------------

def sentences_of(text: str) -> list[str]:
    """Cut free text into sentences (the same NLTK splitter the benchmark notes use)."""
    from trialguide.data.patients import _ensure_punkt
    from nltk.tokenize import sent_tokenize
    _ensure_punkt()
    text = " ".join((text or "").split())
    return [s for s in sent_tokenize(text) if s.strip()]


def note_from_text(text: str, patient_id: str = "new-patient") -> VisibleNote:
    """A whole pasted or uploaded note. Nothing is hidden: the system sees all of it."""
    sents = sentences_of(text)
    if not sents:
        raise ValueError("the note is empty")
    return VisibleNote(patient_id, Split.DEV, dict(enumerate(sents)))


def note_from_facts(age: str, sex: str, condition: str, other: str = "",
                    patient_id: str = "new-patient") -> VisibleNote:
    """Q&A mode: start from a few facts; TrialGuide asks for the rest."""
    parts = []
    if str(age).strip() or str(sex).strip():
        who = " ".join(x for x in [f"{str(age).strip()}-year-old" if str(age).strip() else "", str(sex).strip().lower()] if x)
        parts.append(f"The patient is a {who or 'adult'}.")
    if condition.strip():
        parts.append(f"Main diagnosis: {condition.strip().rstrip('.')}.")
    if other.strip():
        parts.append(other.strip())
    return note_from_text(" ".join(parts), patient_id)


def read_upload(path: str) -> str:
    """Text of an uploaded .txt / .md or .pdf file."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(p) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            from pypdf import PdfReader
            return "\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages)
    return p.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# 2. Candidate trials
# ---------------------------------------------------------------------------

def rules_of_corpus_trial(trial: dict) -> list[TrialRule]:
    """A corpus.jsonl trial -> one rule per criterion (split as in the experiments)."""
    from trialguide.data.masking import split_criteria
    md = trial.get("metadata", {})
    inc = split_criteria(md.get("inclusion_criteria", ""))
    exc = split_criteria(md.get("exclusion_criteria", ""))
    tid = trial["_id"]
    return ([TrialRule(tid, f"inc_{i}", RuleKind.INCLUSION, t) for i, t in enumerate(inc)]
            + [TrialRule(tid, f"exc_{i}", RuleKind.EXCLUSION, t) for i, t in enumerate(exc)])


class TrialIndex:
    """BM25 over the trial collections, built once when the app starts."""

    def __init__(self, raw: Path, cohorts=("sigir", "trec_2021")):
        from rank_bm25 import BM25Okapi
        from trialguide.retrieval import _tokens, _trial_text
        from trialguide.retrieval.corpus import load_trial_corpus
        seen, self.trials = set(), []
        for cohort in cohorts:
            for t in load_trial_corpus(raw, cohort):
                if t["_id"] not in seen and rules_of_corpus_trial(t):
                    seen.add(t["_id"])
                    self.trials.append(t)
        self._tokens = _tokens
        self.bm25 = BM25Okapi([_tokens(_trial_text(t)) for t in self.trials])

    def search(self, note: VisibleNote, k: int = 3, max_rules: int = 30) -> list[dict]:
        """The k best trials for this note that have at most max_rules criteria (keeps the wait short)."""
        query = self._tokens(" ".join(note.sentences.values()))
        scores = self.bm25.get_scores(query)
        order = sorted(range(len(self.trials)), key=lambda i: -scores[i])
        out = []
        for i in order:
            if scores[i] <= 0 or len(out) == k:
                break
            if len(rules_of_corpus_trial(self.trials[i])) <= max_rules:
                out.append(self.trials[i])
        return out


def patient_record(note: VisibleNote, trials: list[dict]) -> dict:
    """The patient in the shape render.py expects (no hidden sentences, no expert labels)."""
    return {"visible": {str(i): s for i, s in note.sentences.items()}, "hidden": {}, "split": "dev",
            "trials": [{"trial_id": t["_id"], "title": t.get("title", ""), "gold": None,
                        "rules": [{"rule_id": r.rule_id, "kind": r.kind.value, "text": r.text}
                                  for r in rules_of_corpus_trial(t)]} for t in trials]}


# ---------------------------------------------------------------------------
# 3-5. The conversation with two models
# ---------------------------------------------------------------------------

def check_many(check: CheckFn, note: VisibleNote, rules: dict) -> dict:
    """Check many rules at once (threads, so the GPU batches them). rules: ref -> TrialRule."""
    refs = list(rules)
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(lambda ref: check(note, rules[ref]), refs))
    return dict(zip(refs, results))


def settles(check) -> bool:
    """Does a check settle the rule from the note (a decided status backed by cited sentences)?"""
    return (check.status in (RuleStatus.MET, RuleStatus.NOT_MET, RuleStatus.NOT_APPLICABLE)
            and check.category in (InfoCategory.STATED, InfoCategory.INFERRED))


class TwoModelSession(DemoSession):
    """DemoSession plus a second model that looks at every rule the first model left open."""

    def __init__(self, note, rules, check, second_check, second_name: str = SECOND_MODEL, **kw):
        super().__init__(note, rules, check, **kw)
        self.second_check = second_check
        self.second_name = second_name
        self.first: dict = {}        # ref -> the first model's latest check
        self.second: dict = {}       # ref -> the second model's check of the same rule
        self.by_second: set = set()  # rules whose decision came from the second model

    def start(self) -> None:
        self.checks = check_many(self.check, self.note, self.rules)
        self.initial_checks = dict(self.checks)       # B6 comparison uses the first model only
        self._decide()
        self.question = self._pick(None)
        self.finished = self.question is None

    def _decide(self) -> None:
        # Any check in self.checks that is not the second model's is a fresh first-model result
        # (start, or a re-check after an answer); the second model must then look again.
        for ref, c in self.checks.items():
            if c is not self.second.get(ref) and self.first.get(ref) is not c:
                self.first[ref] = c
                self.second.pop(ref, None)
        refs = list(self.rules)
        actions = decide_all([self.first[r] for r in refs], self.config, self.doctor_certainty,
                             self.state.asked_fact_keys())
        # the second model looks only at rules the first model could not settle
        todo = {r: self.rules[r] for r, a in zip(refs, actions)
                if a.action not in ACCEPTED and r not in self.second}
        if todo:
            self.second.update(check_many(self.second_check, self.note, todo))
        self.by_second, final = set(), []
        for ref, a in zip(refs, actions):
            s = self.second.get(ref)
            if a.action not in ACCEPTED and s is not None and settles(s):
                self.by_second.add(ref)
                self.checks[ref] = s
                a = replace(a, action=ActionType.ACCEPT_INFERENCE, confidence=s.confidence,
                            reason=f"decided by the second model ({self.second_name.split('/')[-1]}): "
                                   f"{s.status.value.replace('_', ' ')}")
            else:
                self.checks[ref] = self.first[ref]
            final.append(a)
        self.actions = final

    def add_info(self, text: str) -> None:
        """New facts from the user at any time: added to the note, open rules checked again."""
        new = sentences_of(text)
        if not new:
            return
        start = max(self.note.sentences, default=-1) + 1
        self.note = replace(self.note, sentences={**self.note.sentences,
                                                  **{start + i: s for i, s in enumerate(new)}})
        acts = self.action_of()
        open_refs = {r: self.rules[r] for r in self.rules
                     if acts[r].action not in ACCEPTED}
        before = {r: self.checks[r].status for r in open_refs}
        fresh = check_many(self.check, self.note, open_refs)
        self.checks.update(fresh)
        self._decide()
        settled = [(*r, before[r].value, self.checks[r].status.value)
                   for r in open_refs if self.checks[r].status != before[r]]
        self.turns.append(Turn("You added information", " ".join(new), "sure", settled))
        self.question = None if self.state.questions_left() <= 0 else self._pick(None)
        self.finished = self.question is None

    def fork(self) -> "TwoModelSession":
        second, self.second_check = self.second_check, None
        twin = super().fork()
        self.second_check = twin.second_check = second
        return twin

    # ---- what the page shows ----

    def counts(self, trial_id: str) -> dict:
        """As in the recorded demo, plus the rules the second model decided, counted on their own."""
        acts = self.action_of()
        out = {"stated": 0, "inferred": 0, "answered": 0, "second": 0, "open": 0}
        for (t, r), c in self.checks.items():
            if t != trial_id:
                continue
            a = acts[(t, r)].action
            if a not in ACCEPTED:
                out["open"] += 1
            elif (t, r) in self.by_second:
                out["second"] += 1
            elif any(i in self.doctor_certainty for i in c.evidence_ids):
                out["answered"] += 1
            elif a == ActionType.ACCEPT_INFERENCE:
                out["inferred"] += 1
            else:
                out["stated"] += 1
        return out

    def rule_rows(self, trial_id: str) -> list[dict]:
        rows = super().rule_rows(trial_id)
        for row in rows:
            row["decided_by"] = ("second model" if (trial_id, row["rule"]) in self.by_second
                                 else "first model")
        return rows


# ---------------------------------------------------------------------------
# A stand-in checker for trying the page without a GPU (--fake). Not a model:
# it only matches words, so its decisions mean nothing. Tests use it too.
# ---------------------------------------------------------------------------

_STOP = {"the", "and", "with", "for", "who", "are", "has", "have", "any", "not", "patients", "patient",
         "must", "be", "of", "or", "to", "in", "a", "an", "is", "on", "at", "by", "as", "than", "from"}


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z0-9-]{2,}", text.lower()) if w not in _STOP}


_LEAD = re.compile(r"^(patients?|subjects?|participants?)?\s*(who|with|that|must|should|have|has|had|are|is|be|"
                   r"a|an|the|any|known|documented|presence of|evidence of|diagnosis of|history of|prior)\b\s*", re.I)


_TAIL = {"or", "and", "of", "a", "an", "the", "with", "to", "in", "on", "for", "by", "at", "as", "be", "is", "are"}


def _fact_label(text: str) -> Optional[str]:
    """A short readable name for the fact a rule needs, used as the question (stand-in only).
    The real checker writes its own name for the fact; this keeps the stand-in's questions readable."""
    t = re.sub(r"\s+", " ", (text or "")).strip(" .;:-*").lower()
    if not t or "empty criterion" in t:
        return None
    if re.search(r"\bage\b|years? (of age|old)", t):
        return "age"                                  # one word is fine here
    for _ in range(4):                               # drop leading filler such as "patients with a history of"
        t2 = _LEAD.sub("", t).strip()
        if t2 == t:
            break
        t = t2
    t = re.sub(r"\([^)]*\)?", " ", t)                 # drop bracketed asides, even unclosed ones
    t = re.split(r"[,;:]| (?:who|with|that|which|or within|defined as|such as|including) ", t)[0]
    if t.startswith("for "):                          # a heading such as "For COPD patients", not a rule
        return None
    words = t.replace("/", " ").split()[:9]
    while words and words[-1] in _TAIL:              # no dangling "or", "of", "a" at the end
        words.pop()
    return "_".join(words) if len(words) >= 2 else None


_AGE = re.compile(r"(\d{1,3})[- ]year[- ]old|aged? (\d{1,3})", re.I)


def _age_check(note: VisibleNote, rule: TrialRule):
    """(status, sentence id) for an age rule when the note states the age, else None (stand-in only).
    MET means the rule's text is true for the patient, as for every check (also for exclusion rules)."""
    text = rule.text.lower()
    if not re.search(r"\bage\b|years? (of age|old)|\byears\b", text):
        return None
    nums = [int(n) for n in re.findall(r"\d{1,3}", text) if 0 < int(n) < 120]
    if not nums:
        return None
    for sid, sent in note.sentences.items():
        m = _AGE.search(sent)
        if not m:
            continue
        age = int(m.group(1) or m.group(2))
        if len(nums) >= 2 and re.search(r"\d\s*(-|to|and)\s*\d", text):
            ok = min(nums[:2]) <= age <= max(nums[:2])
        elif re.search(r"<|≤|younger|under|less|below|up to", text):
            ok = age <= nums[0] if re.search(r"≤|<=|or younger|or less|up to", text) else age < nums[0]
        else:
            ok = age >= nums[0] if re.search(r"≥|>=|or older|or over|at least|or more", text) else age >= nums[0]
        return (RuleStatus.MET if ok else RuleStatus.NOT_MET), sid
    return None


def keyword_checker(settle_open: bool = False) -> CheckFn:
    """settle_open=False: the 'first model' (stated if the rule's words appear in a sentence).
    settle_open=True: the 'second model' (also settles consent / willingness rules by inference)."""
    from trialguide.schemas import RuleCheck

    def check(note: VisibleNote, rule: TrialRule):
        words = _words(rule.text)
        best = max(note.sentences.items(), key=lambda kv: len(words & _words(kv[1])), default=None)
        hit = best is not None and len(words & _words(best[1])) >= 2
        base = dict(patient_id=note.patient_id, trial_id=rule.trial_id, rule_id=rule.rule_id,
                    kind=rule.kind, n_samples=1)
        age = _age_check(note, rule)
        if age is not None:
            return RuleCheck(**base, status=age[0], category=InfoCategory.STATED,
                             evidence_ids=[age[1]], confidence=0.95, explanation="age stated in the note")
        if hit:
            return RuleCheck(**base, status=RuleStatus.MET, category=InfoCategory.STATED,
                             evidence_ids=[best[0]], confidence=0.95, explanation="words match")
        if settle_open and re.search(r"consent|willing|able to comply", rule.text, re.I) and note.sentences:
            return RuleCheck(**base, status=RuleStatus.MET, category=InfoCategory.INFERRED,
                             evidence_ids=[min(note.sentences)], confidence=0.8, explanation="assumed willing")
        key = _fact_label(rule.text)
        return RuleCheck(**base, status=RuleStatus.NOT_ENOUGH_INFO, category=InfoCategory.MISSING,
                         confidence=0.5, explanation="not in the note", fact_key=key)

    return check
