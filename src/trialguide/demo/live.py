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
    """Check many rules at once (threads, so the GPU batches them). rules: ref -> TrialRule.
    A checker with a .many method (the remote checker) gets all rules in one call instead."""
    if hasattr(check, "many"):
        return check.many(note, rules)
    refs = list(rules)
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(lambda ref: check(note, rules[ref]), refs))
    return dict(zip(refs, results))


def safe(check: CheckFn) -> CheckFn:
    """A checker that never stops the page: if one rule check fails (a model or network error),
    that rule is left open, as an unresolved check, and the error is printed for the log."""
    import traceback
    from trialguide.schemas import RuleCheck

    def run(note: VisibleNote, rule: TrialRule):
        try:
            return check(note, rule)
        except Exception as e:                       # noqa: BLE001  (any failure leaves the rule open)
            traceback.print_exc()
            return RuleCheck(note.patient_id, rule.trial_id, rule.rule_id, rule.kind, RuleStatus.NOT_ENOUGH_INFO,
                             InfoCategory.MISSING, confidence=0.0, n_samples=1,
                             explanation=f"check failed: {type(e).__name__}")
    return run


# ---------------------------------------------------------------------------
# Remote models: the free website sends rule checks to the GPU notebook on Kaggle when it is
# running, and uses the word-matching stand-in when it is not.
# ---------------------------------------------------------------------------

def check_to_json(c) -> dict:
    """A RuleCheck as plain JSON (enums as their values)."""
    from dataclasses import asdict
    d = asdict(c)
    for k in ("kind", "status", "category"):
        d[k] = getattr(c, k).value
    return d


def note_to_json(note: VisibleNote) -> dict:
    return {"patient_id": note.patient_id, "sentences": {str(i): t for i, t in note.sentences.items()}}


def note_from_json(d: dict) -> VisibleNote:
    return VisibleNote(d["patient_id"], Split.DEV, {int(i): t for i, t in d["sentences"].items()})


def rule_to_json(r: TrialRule) -> dict:
    return {"trial_id": r.trial_id, "rule_id": r.rule_id, "kind": r.kind.value, "text": r.text}


def rule_from_json(d: dict) -> TrialRule:
    return TrialRule(d["trial_id"], d["rule_id"], RuleKind(d["kind"]), d["text"])


def serve_checks(payload: str, checkers: dict) -> str:
    """GPU side: one request = one model, one note, many rules. Returns the checks as JSON."""
    import json
    req = json.loads(payload)
    check = checkers[req["model"]]
    note = note_from_json(req["note"])
    rules = {i: rule_from_json(r) for i, r in enumerate(req["rules"])}
    out = check_many(check, note, rules)
    return json.dumps([check_to_json(out[i]) for i in range(len(rules))])


_clients: dict = {}


class RemoteChecker:
    """Website side: checks rules with a model running on the Kaggle GPU (url = its gradio.live link).
    If the GPU stops answering, the rules are checked by the fallback (the stand-in) instead and
    self.failed is set, so the page can say so."""

    def __init__(self, url: str, model: str, key: str, fallback: CheckFn):
        self.url, self.model, self.key, self.fallback = url, model, key, fallback
        self.failed = False

    def _client(self):
        from gradio_client import Client
        if self.url not in _clients:
            _clients[self.url] = Client(self.url, verbose=False)
        return _clients[self.url]

    def many(self, note: VisibleNote, rules: dict) -> dict:
        import json
        refs = list(rules)
        payload = json.dumps({"model": self.model, "note": note_to_json(note),
                              "rules": [rule_to_json(rules[r]) for r in refs]})
        if not self.failed:
            try:
                job = self._client().submit(payload, self.key, api_name="/check_rules")
                from trialguide.policy.calibrate import check_from_json
                got = [check_from_json(d) for d in json.loads(job.result(timeout=900))]
                return dict(zip(refs, got))
            except Exception:                        # noqa: BLE001  (GPU gone: use the stand-in)
                import traceback
                traceback.print_exc()
                self.failed = True
                _clients.pop(self.url, None)
        return {r: self.fallback(note, rules[r]) for r in refs}

    def __call__(self, note: VisibleNote, rule: TrialRule):
        return self.many(note, {0: rule})[0]


# ---------------------------------------------------------------------------
# Demo safeguard (live page only, not part of the evaluated system): a model's conclusion counts only
# if a sentence it cites is about the rule. In the first runs on new notes the models often "settled"
# rules such as "written informed consent" or "active infection" by citing an unrelated sentence
# ("He has not received docetaxel."). Such a result is turned back into an open rule, so the doctor
# is asked instead of the model guessing.
# ---------------------------------------------------------------------------

_GENERIC = {"cancer", "disease", "treatment", "therapy", "history", "previous", "prior", "within", "least",
            "months", "years", "year", "weeks", "days", "study", "trial", "including", "other", "than",
            "more", "less", "must", "should", "will", "been", "were", "this", "that", "such", "following"}


def _topic_words(text: str) -> set:
    return {w for w in _words(text) if len(w) >= 4 and w not in _GENERIC}


def grounded(check, rule: TrialRule, note: VisibleNote) -> bool:
    """True when a cited sentence (or a tool result) shares a topic word with the rule."""
    if check.tool_calls:
        return True
    words = _topic_words(rule.text)
    if not words:                                      # nothing to compare (e.g. an empty criterion)
        return True
    return any(words & _topic_words(note.sentences.get(i, "")) for i in check.evidence_ids)


class Guarded:
    """Wraps a checker with the demo safeguard. strict=False (first model): only inferences must be
    grounded. strict=True (second model): every conclusion must be, since it has no risk threshold."""

    def __init__(self, inner: CheckFn, strict: bool = False):
        self.inner, self.strict = inner, strict

    @property
    def failed(self) -> bool:                          # the remote GPU stopped (see RemoteChecker)
        return getattr(self.inner, "failed", False)

    def _guard(self, c, rule: TrialRule, note: VisibleNote):
        settled = c.status in (RuleStatus.MET, RuleStatus.NOT_MET, RuleStatus.NOT_APPLICABLE)
        check_it = c.category == InfoCategory.INFERRED or (self.strict and c.category == InfoCategory.STATED)
        if settled and check_it and not grounded(c, rule, note):
            return replace(c, status=RuleStatus.NOT_ENOUGH_INFO, category=InfoCategory.MISSING, evidence_ids=[],
                           tool_calls=[], fact_key=c.fact_key or _fact_label(rule.text),
                           explanation="left open: the cited sentence is not about this rule")
        return c

    def many(self, note: VisibleNote, rules: dict) -> dict:
        out = check_many(self.inner, note, rules)
        return {r: self._guard(c, rules[r], note) for r, c in out.items()}

    def __call__(self, note: VisibleNote, rule: TrialRule):
        return self._guard(self.inner(note, rule), rule, note)


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
