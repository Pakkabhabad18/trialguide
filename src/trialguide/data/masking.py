"""Masking (T10): hide facts from a note in three ways and hand out only the visible part.

    random    hide a fraction (20 / 40 / 60%) of sentences, chosen by a seeded draw
    rule      hide the sentences that the most candidate-trial rules ask about
    evidence  hide exactly the sentences the experts cited as evidence (labelled patients only)

`MaskedNote.visible` is the only thing a system may read. `hidden_ids` is for scoring and logs;
a component under test must never read it (docs/leakage_checklist.md).

Seeds depend on (base seed, split, patient, mask type, fraction), so calibration and test masks
are drawn independently of development and of each other, and every mask is reproducible.
"""

import hashlib
import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from trialguide.data.patients import Patient
from trialguide.schemas import MaskType, VisibleNote

DEFINITE = {"included", "not included", "excluded", "not excluded"}


@dataclass(frozen=True)
class MaskedNote:
    visible: VisibleNote
    hidden_ids: tuple           # scoring and logs only; never passed to a system


def derive_seed(base_seed: int, split: str, patient_id: str, mask_type: str, key: str = "") -> int:
    """A stable 63-bit seed that differs per split, patient, mask type and fraction/key."""
    h = hashlib.sha256(f"{base_seed}|{split}|{patient_id}|{mask_type}|{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def n_to_hide(n_sentences: int, fraction: float) -> int:
    """How many sentences a fraction means: rounded half up, at least 1, never the whole note."""
    if fraction <= 0 or n_sentences < 2:
        return 0
    return min(max(int(fraction * n_sentences + 0.5), 1), n_sentences - 1)


def _masked(patient: Patient, hidden: set, mask_type: MaskType, fraction: float, seed) -> MaskedNote:
    visible = {i: s for i, s in patient.sentences.items() if i not in hidden}
    note = VisibleNote(patient_id=patient.patient_id, split=patient.split, sentences=visible,
                       mask_type=mask_type, mask_fraction=fraction, mask_seed=seed)
    return MaskedNote(note, tuple(sorted(hidden)))


def mask_none(patient: Patient) -> MaskedNote:
    """The full note, for the oracle baseline B1 and full-note references only."""
    return _masked(patient, set(), MaskType.NONE, 0.0, None)


def mask_random(patient: Patient, fraction: float, base_seed: int = 0) -> MaskedNote:
    seed = derive_seed(base_seed, patient.split.value, patient.patient_id, "random", f"{fraction:.2f}")
    ids = sorted(patient.sentences)
    hidden = set(random.Random(seed).sample(ids, n_to_hide(len(ids), fraction)))
    return _masked(patient, hidden, MaskType.RANDOM, fraction, seed)


# ---- rule-targeted ----

def split_criteria(text: str) -> list:
    """One string per eligibility criterion. Criteria are separated by blank lines; a header
    such as "inclusion criteria:" may share a chunk with the first criterion, so only the
    header words are removed."""
    out = []
    for chunk in re.split(r"\n\s*\n", text or ""):
        chunk = re.sub(r"^\s*(inclusion|exclusion)\s+criteria\s*:?\s*", "", chunk.strip(), flags=re.I)
        if chunk.strip():
            out.append(" ".join(chunk.split()))
    return out


@lru_cache(maxsize=3)
def _retrieved(raw: str, cohort: str) -> dict:
    with open(Path(raw) / "trialgpt" / cohort / "retrieved_trials.json", encoding="utf-8") as f:
        return {e["patient_id"]: e for e in json.load(f)}


def candidate_rules(raw: Path, cohort: str, patient_id: str) -> list:
    """For each candidate trial of the patient (the benchmark's expert-labelled set), its criteria.
    Benchmark construction only: a system never sees this."""
    entry = _retrieved(str(raw), cohort)[patient_id]
    rules = []
    for bucket in ("0", "1", "2"):
        for trial in entry.get(bucket, []):
            rules.append(split_criteria(trial.get("inclusion_criteria", ""))
                         + split_criteria(trial.get("exclusion_criteria", "")))
    return [r for r in rules if r]


def rule_demand(sentences: dict, trial_rules: list) -> dict:
    """How strongly candidate-trial rules ask about each sentence.

    demand(sentence) = mean over trials of the best TF-IDF cosine between the sentence and any
    rule of that trial. Lexical, so it is a proxy for "some rule asks about this fact"; it needs
    no threshold and does not depend on the model under test.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    ids = sorted(sentences)
    flat = [r for rules in trial_rules for r in rules]
    if not flat:
        return {i: 0.0 for i in ids}
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(flat + [sentences[i] for i in ids])
    s_mat, r_mat = vec.transform([sentences[i] for i in ids]), vec.transform(flat)
    sims = (s_mat @ r_mat.T).toarray()                      # sentences x rules
    demand, start = [0.0] * len(ids), 0
    for rules in trial_rules:
        block = sims[:, start:start + len(rules)].max(axis=1)
        demand = [d + b for d, b in zip(demand, block)]
        start += len(rules)
    return {i: d / len(trial_rules) for i, d in zip(ids, demand)}


def mask_rule(patient: Patient, demand: dict, fraction: float, base_seed: int = 0) -> MaskedNote:
    """Hide the sentences with the highest rule demand; ties are broken by a seeded draw."""
    seed = derive_seed(base_seed, patient.split.value, patient.patient_id, "rule", f"{fraction:.2f}")
    rng = random.Random(seed)
    tie = {i: rng.random() for i in sorted(patient.sentences)}
    order = sorted(patient.sentences, key=lambda i: (-round(demand.get(i, 0.0), 9), tie[i]))
    hidden = set(order[:n_to_hide(len(order), fraction)])
    return _masked(patient, hidden, MaskType.RULE, fraction, seed)


# ---- evidence-targeted ----

def evidence_sets(raw: Path, scope: str = "pair") -> dict:
    """Expert evidence sentence ids from the annotation file, for rules the experts settled.

    scope "pair": {(patient, trial): ids}, "rule": {(patient, trial, annotation_id): ids},
    "patient": {patient: ids}. Only rows with a definite label count; evidence cited for
    "not enough information" rows is not what an answer depends on.
    """
    ann = pd.read_parquet(Path(raw) / "annotations" / "criterion_annotations.parquet")
    ann = ann[ann.expert_eligibility.isin(DEFINITE) & ann.expert_sentences.notna()]
    keyers = {"pair": lambda r: (r.patient_id, r.trial_id),
              "rule": lambda r: (r.patient_id, r.trial_id, int(r.annotation_id)),
              "patient": lambda r: r.patient_id}
    out = {}
    for r in ann.itertuples():
        ids = set(json.loads(r.expert_sentences))
        if ids:
            out.setdefault(keyers[scope](r), set()).update(ids)
    return {k: frozenset(v) for k, v in out.items()}


def mask_evidence(patient: Patient, evidence_ids) -> MaskedNote:
    """Hide exactly the cited sentences. No randomness; the note can end up with little or nothing
    visible, which is the point of this test (the right answer is then to ask or abstain)."""
    hidden = {i for i in evidence_ids if i in patient.sentences}
    return _masked(patient, hidden, MaskType.EVIDENCE, len(hidden) / max(len(patient.sentences), 1), None)
