"""Simulated doctor (task T11, proposal Section 6, decisions D8, D10, D11).

The doctor answers TrialGuide's questions about one patient using ONLY the facts
masking hid from the system. It never sees the trials, the rules, the visible
note or the patient id, only the question and the hidden facts. The prompt is
adapted from the MediQ patient simulator (Li et al., NeurIPS 2024).

Faithfulness is enforced in code, not just asked for in the prompt (D10): an
answer must quote a hidden fact. If the quoted text is not found in the hidden
facts, the answer becomes "not documented", so an invented answer can never
reach the system. Every question and answer is appended to a JSONL log; the
"not documented" replies feed the failure analysis and the 150-answer
faithfulness check.

The backbone is a different model family from the system (Llama-3.1-8B-Instruct
vs Qwen3), so the doctor and the system do not share blind spots.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from difflib import SequenceMatcher
from typing import Callable, Optional

from trialguide.schemas import Certainty, DoctorAnswer, Question

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LOG_PATH = os.getenv("DOCTOR_LOG", "outputs/doctor_log.jsonl")
MIN_QUOTE_MATCH = 0.85  # share of the quote that must match a hidden fact (allows small edits)

SYSTEM_PROMPT = """You are a doctor answering questions about your patient, using ONLY the patient's record below.

Rules:
- Answer only from the record. Never use medical knowledge to guess facts that are not written there.
- Quote the sentence from the record that answers the question, copied exactly.
- If the record does not answer the question, say so: certainty "not_documented", empty answer, quote null.
- Answer only the question asked. Do not volunteer other facts.

Certainty:
- "sure": the record states the answer directly.
- "probably": the record strongly suggests the answer but does not state it.
- "unsure": the record mentions the topic but leaves the answer unclear.
- "not_documented": the record does not mention it.

Reply with a JSON object only:
{"answer": "short answer, e.g. yes / no / 1.9 mg/dL", "certainty": "sure | probably | unsure | not_documented", "quote": "exact sentence from the record, or null"}"""

ChatFn = Callable[[list, str], str]  # (messages, model) -> reply text


def _default_chat(messages: list, model: str) -> str:
    # Imported here so this module (and its tests) work before the shared client is on main
    from trialguide.llm.client import chat
    return chat(messages, model, temperature=0)


def _parse(text: str) -> Optional[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip().strip("`")
    if text.startswith("json"):
        text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip(" .")


def find_quote(quote: str, facts: list[str]) -> Optional[str]:
    """The hidden fact the quote comes from, or None if it is not in the record.

    Accepts an exact substring, or a near-copy where at least MIN_QUOTE_MATCH of
    the quote matches one fact (models drop a word or change punctuation).
    """
    q = _norm(quote)
    if len(q) < 3:
        return None
    for fact in facts:
        if q in _norm(fact):
            return fact
    for fact in facts:
        matched = sum(b.size for b in SequenceMatcher(None, q, _norm(fact)).get_matching_blocks())
        if matched / len(q) >= MIN_QUOTE_MATCH:
            return fact
    return None


class SimulatedDoctor:
    """Implements schemas.Doctor: answer(patient_id, question) -> DoctorAnswer."""

    def __init__(self, hidden_facts: dict[str, list[str]], model: str = MODEL,
                 chat_fn: Optional[ChatFn] = None, log_path: Optional[str] = LOG_PATH):
        """hidden_facts: patient_id -> the facts masking hid (from the masking module, T10)."""
        self.hidden_facts = hidden_facts
        self.model = model
        self.chat_fn = chat_fn or _default_chat
        self.log_path = log_path

    def answer(self, patient_id: str, question: Question) -> DoctorAnswer:
        facts = self.hidden_facts.get(patient_id, [])
        if not facts:
            return self._log(patient_id, question, self._not_documented(question), "no hidden facts", None)

        record = "\n".join(f"- {fact}" for fact in facts)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Patient record:\n{record}\n\nQuestion: {question.text}\n\nJSON reply:"},
        ]
        raw = self.chat_fn(messages, self.model)
        answer, reason = self._validate(question, raw, facts)
        return self._log(patient_id, question, answer, reason, raw)

    def _validate(self, question: Question, raw: str, facts: list[str]) -> tuple[DoctorAnswer, str]:
        """The answer to use, and why it is 'not documented' when it is."""
        data = _parse(raw)
        if data is None:
            return self._not_documented(question, raw), "unparsable reply"
        try:
            certainty = Certainty(str(data.get("certainty", "")).strip().lower().replace(" ", "_"))
        except ValueError:
            return self._not_documented(question, raw), "unknown certainty"
        if certainty == Certainty.NOT_DOCUMENTED:
            return self._not_documented(question, raw), "doctor said not documented"

        quote = data.get("quote")
        source = find_quote(quote, facts) if isinstance(quote, str) else None
        if source is None:
            # D10: an answer the record does not support is not used
            return self._not_documented(question, raw), "quote not found in the record"
        answer = DoctorAnswer(question_id=question.question_id, value=str(data.get("answer", "")).strip(),
                              certainty=certainty, quote=source, raw_text=raw)
        return answer, ""

    @staticmethod
    def _not_documented(question: Question, raw: str = "") -> DoctorAnswer:
        return DoctorAnswer(question_id=question.question_id, value="",
                            certainty=Certainty.NOT_DOCUMENTED, raw_text=raw)

    def _log(self, patient_id: str, question: Question, answer: DoctorAnswer, reason: str,
             raw: Optional[str]) -> DoctorAnswer:
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps({
                    "date": datetime.datetime.now().isoformat(),
                    "model": self.model,
                    "patient_id": patient_id,
                    "question_id": question.question_id,
                    "fact_key": question.fact_key,
                    "question": question.text,
                    "certainty": answer.certainty.value,
                    "answer": answer.value,
                    "quote": answer.quote,
                    "reason": reason,
                    "raw": raw,
                }) + "\n")
        return answer
