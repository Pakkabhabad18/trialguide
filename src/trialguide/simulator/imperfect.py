"""The imperfect doctor for the interaction variant of ablation A7 (T47, part b).

A wrapper around any doctor: a fixed 20% of its answers are made worse, half to "unsure" (the value and
quote are kept, but the policy no longer treats the answer as settled) and half to "not documented" (the
answer is dropped). Which answers are hit depends only on the patient and the question text, through a
hash, so every system and every replay round sees the same downgrades. Answers that are already
"not documented" stay as they are, so in practice fewer than 20% of answers change.
Owner: Prakash.
"""

from __future__ import annotations

import hashlib

from trialguide.schemas import Certainty, DoctorAnswer, Question


class ImperfectDoctor:
    def __init__(self, doctor, rate: float = 0.2, salt: str = "a7-imperfect"):
        self.doctor, self.rate, self.salt = doctor, rate, salt
        self.model = getattr(doctor, "model", "unknown")

    def draw(self, patient_id: str, question: Question) -> float:
        """A number in [0, 1) fixed by the patient and the question text."""
        digest = hashlib.sha256(f"{self.salt}|{patient_id}|{question.text}".encode()).hexdigest()
        return int(digest[:12], 16) / 16 ** 12

    def answer(self, patient_id: str, question: Question) -> DoctorAnswer:
        answer = self.doctor.answer(patient_id, question)
        u = self.draw(patient_id, question)
        if u >= self.rate or answer.certainty == Certainty.NOT_DOCUMENTED:
            return answer
        note = f"[imperfect doctor: downgraded from {answer.certainty.value}] "
        if u < self.rate / 2:
            return DoctorAnswer(answer.question_id, answer.value, Certainty.UNSURE, answer.quote,
                                note + answer.raw_text)
        return DoctorAnswer(answer.question_id, "", Certainty.NOT_DOCUMENTED, None, note + answer.raw_text)
