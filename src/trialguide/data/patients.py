"""Patients and facts (T10): load a split's notes and cut each note into sentence-level facts.

A fact is one sentence of the note. Sentence ids follow TrialGPT exactly (nltk sentence
splitting, plus its closing "informed consent" sentence for SIGIR), so the expert evidence ids
in the annotation file point at the same sentences as ours (checked for all 52 labelled patients).

`Patient` holds the FULL note. It is for masking, the simulated doctor and scoring only. Anything
that is under test receives a `VisibleNote` from `trialguide.data.masking`, never a `Patient`.
"""

from dataclasses import dataclass
from pathlib import Path

from trialguide.data.audit import cohort_of, load_queries
from trialguide.data.splits import load_split
from trialguide.schemas import Fact, FactSource, Split

# TrialGPT appends this sentence to every SIGIR patient before splitting (run_matching.py)
SIGIR_CONSENT = ("The patient will provide informed consent, and will comply with the "
                 "trial protocol without any practical issues.")


@dataclass
class Patient:
    patient_id: str
    cohort: str                 # sigir | trec_2021 | trec_2022
    split: Split
    sentences: dict             # sentence id -> text, the full note


def _ensure_punkt() -> None:
    """Fetch nltk's sentence model on first use (needs internet once)."""
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def split_sentences(text: str, cohort: str) -> dict:
    """Cut a raw note into {sentence id: sentence}, numbered from 0 as in TrialGPT."""
    from nltk.tokenize import sent_tokenize
    _ensure_punkt()
    if cohort == "sigir":
        text = text.rstrip() + " " + SIGIR_CONSENT
    return dict(enumerate(sent_tokenize(text)))


def load_patients(raw: Path, split: Split, allow_test: bool = False) -> dict:
    """{patient_id: Patient} for one frozen split.

    The test split is sealed until T33: asking for it without allow_test=True raises, so no
    script can read the 73 test notes by accident.
    """
    if split == Split.TEST and not allow_test:
        raise PermissionError("The test split is sealed until T33 (opened once, after the v1.0 freeze).")
    notes = {}
    for cohort, patients in load_queries(raw).items():
        notes.update({p: (cohort, text) for p, text in patients.items()})
    return {p: Patient(p, notes[p][0], split, split_sentences(notes[p][1], notes[p][0]))
            for p in load_split()[split]}


def facts_of(patient: Patient) -> list:
    """The patient's note as schema Facts, one per sentence."""
    return [Fact(fact_id=f"{patient.patient_id}#s{i}", patient_id=patient.patient_id,
                 text=s, source=FactSource.NOTE, sentence_id=i)
            for i, s in patient.sentences.items()]


__all__ = ["Patient", "split_sentences", "load_patients", "facts_of", "cohort_of", "SIGIR_CONSENT"]
