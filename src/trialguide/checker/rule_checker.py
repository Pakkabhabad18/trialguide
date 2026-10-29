"""Conservative rule checker for TrialGuide (T12).
Uses only the visible patient note. Unsupported assumptions never become facts.
"""

from collections import Counter
import json
import re

from trialguide.llm.client import chat, parse_json
from trialguide.schemas import (
    InfoCategory,
    RuleCheck,
    RuleKind,
    RuleStatus,
)


_VALID_CATEGORIES = {item.value for item in InfoCategory}
_VALID_STATUSES = {item.value for item in RuleStatus}


def _normalise_fact_key(value):
    """Normalize a missing clinical fact into a safe snake_case key."""
    if not isinstance(value, str):
        return None

    key = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")

    if not key or len(key) > 64 or key in {
        "none", "null", "unknown", "missing",
        "not_applicable", "not_known",
    }:
        return None

    return key


def _consensus_fact_key(samples):
    """Return the most common fact key only when its lead is unambiguous."""
    counts = Counter(
        sample["fact_key"]
        for sample in samples
        if sample.get("fact_key") is not None
    )

    if not counts:
        return None

    key, votes = counts.most_common(1)[0]

    if list(counts.values()).count(votes) > 1:
        return None

    return key


def _normalise_status(value, kind):
    """Accept the shared schema vocabulary and TrialGPT's equivalent labels."""
    value = str(value).strip().lower().replace(" ", "_")

    aliases = {
        "not_enough_information": RuleStatus.NOT_ENOUGH_INFO.value,
        "not_enough_info": RuleStatus.NOT_ENOUGH_INFO.value,
        "not_applicable": RuleStatus.NOT_APPLICABLE.value,
    }

    if kind == RuleKind.INCLUSION:
        aliases.update({
            "included": RuleStatus.MET.value,
            "not_included": RuleStatus.NOT_MET.value,
        })
    else:
        aliases.update({
            "excluded": RuleStatus.MET.value,
            "not_excluded": RuleStatus.NOT_MET.value,
        })

    value = aliases.get(value, value)

    if value not in _VALID_STATUSES:
        raise ValueError(f"Invalid rule status: {value}")

    return value


def _parse_sample(raw, visible_ids, kind):
    """Validate one model response; return None for malformed responses."""
    try:
        data = parse_json(raw)

        if not isinstance(data, dict):
            return None

        status = _normalise_status(data["status"], kind)
        category = str(data["category"]).strip().lower()

        if category not in _VALID_CATEGORIES:
            return None

        raw_ids = data.get("evidence_ids", [])

        if not isinstance(raw_ids, list):
            return None

        # Only integer IDs that exist in the visible note are admissible.
        evidence_ids = sorted({
            item for item in raw_ids
            if isinstance(item, int)
            and not isinstance(item, bool)
            and item in visible_ids
        })

        explanation = str(data.get("explanation", "")).strip()
        fact_key = _normalise_fact_key(data.get("fact_key"))

        # Assumptions and missing information can never establish a rule.
        if category in (
            InfoCategory.ASSUMPTION.value,
            InfoCategory.MISSING.value,
        ):
            status = RuleStatus.NOT_ENOUGH_INFO.value
            evidence_ids = []

        # A stated/inferred conclusion must have valid supporting evidence.
        if category in (
            InfoCategory.STATED.value,
            InfoCategory.INFERRED.value,
        ) and not evidence_ids:
            category = InfoCategory.MISSING.value
            status = RuleStatus.NOT_ENOUGH_INFO.value
            evidence_ids = []
            explanation = (
                "Unresolved: the response supplied no valid visible-note "
                "evidence IDs."
            )

        return {
            "status": status,
            "category": category,
            "evidence_ids": evidence_ids,
            "explanation": explanation,
            "fact_key": fact_key,
        }

    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def check_rule(note, rule, model, n_samples=5):
    """Check a trial rule against only the visible note.

    Confidence is the modal-status vote fraction over requested samples;
    it is an agreement measure, not a calibrated probability of correctness.
    Invalid responses count against the fraction because the denominator is
    the number of requested samples.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    if rule.kind not in (RuleKind.INCLUSION, RuleKind.EXCLUSION):
        raise ValueError(f"Unsupported rule kind: {rule.kind}")

    visible_ids = set(note.sentences)
    note_text = note.text()
    samples = []

    system_prompt = """
You are a conservative clinical-trial rule checker.
Use ONLY the supplied visible patient note. Never invent facts or assume
an undocumented fact is false.

Interpret status as whether the rule's condition is true for this patient:
- inclusion rule: met means the patient satisfies the criterion;
- exclusion rule: met means the patient has the disqualifying condition.

Use category:
- stated: directly supported by a visible sentence;
- inferred: follows from visible evidence through a clear, defensible inference;
- assumption: plausible but not established by the note;
- missing: the note contains no relevant evidence.

Only stated or inferred results may have a decisive status, and they must
cite valid visible-note sentence IDs. Assumption or missing means
not_enough_info. Do not treat absence of a statement as proof of absence.

Return ONLY a JSON object with these keys:
{
  "status": "met | not_met | not_enough_info | not_applicable",
  "category": "stated | inferred | assumption | missing",
  "evidence_ids": [0],
  "explanation": "brief explanation",
  "fact_key": "prior_chemotherapy or null"
}
The evidence_ids must be integer IDs copied from the visible note.

For unresolved results (assumption, missing, or not_enough_info), provide
fact_key as a concise snake_case name for the single missing clinical fact
when one is identifiable. Use null when no specific fact can be named.
Use the same key for the same clinical fact across rules/trials. Do not use
patient, trial, or rule IDs, and do not use generic keys like "unknown".
For decisive stated/inferred results, fact_key must be null.
"""

    for sample_index in range(n_samples):
        user_prompt = f"""
Independent assessment sample: {sample_index + 1} of {n_samples}.
Assess this rule independently from the other samples.

Patient ID: {note.patient_id}
Trial ID: {rule.trial_id}
Rule ID: {rule.rule_id}
Rule kind: {rule.kind.value}
Rule text: {rule.text}

Visible patient note:
{note_text}

Return the required JSON object only.
"""
        try:
            raw = chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model,
                temperature=0,
            )

            parsed = _parse_sample(raw, visible_ids, rule.kind)

            if parsed is not None:
                samples.append(parsed)

        except Exception:
            # A failed sample must not turn into a guessed result.
            continue

    if not samples:
        return RuleCheck(
            patient_id=note.patient_id,
            trial_id=rule.trial_id,
            rule_id=rule.rule_id,
            kind=rule.kind,
            status=RuleStatus.NOT_ENOUGH_INFO,
            category=InfoCategory.MISSING,
            evidence_ids=[],
            confidence=0.0,
            n_samples=n_samples,
            explanation="Unresolved: no valid model samples were available.",
            fact_key=None,
        )

    status_counts = Counter(sample["status"] for sample in samples)
    winning_status, winning_votes = status_counts.most_common(1)[0]
    confidence = winning_votes / n_samples

    # A status tie must never produce an arbitrary decision.
    if list(status_counts.values()).count(winning_votes) > 1:
        return RuleCheck(
            patient_id=note.patient_id,
            trial_id=rule.trial_id,
            rule_id=rule.rule_id,
            kind=rule.kind,
            status=RuleStatus.NOT_ENOUGH_INFO,
            category=InfoCategory.MISSING,
            evidence_ids=[],
            confidence=confidence,
            n_samples=n_samples,
            explanation="Unresolved: assessment samples tied on rule status.",
            fact_key=None,
        )

    winning_samples = [
        sample for sample in samples
        if sample["status"] == winning_status
    ]

    category_counts = Counter(
        sample["category"] for sample in winning_samples
    )
    winning_category, category_votes = category_counts.most_common(1)[0]

    # A category tie is also ambiguous: do not combine evidence categories.
    if list(category_counts.values()).count(category_votes) > 1:
        return RuleCheck(
            patient_id=note.patient_id,
            trial_id=rule.trial_id,
            rule_id=rule.rule_id,
            kind=rule.kind,
            status=RuleStatus.NOT_ENOUGH_INFO,
            category=InfoCategory.MISSING,
            evidence_ids=[],
            confidence=confidence,
            n_samples=n_samples,
            explanation=(
                "Unresolved: assessment samples tied on evidence category."
            ),
            fact_key=None,
        )

    evidence_ids = sorted({
        sentence_id
        for sample in winning_samples
        if sample["category"] == winning_category
        for sentence_id in sample["evidence_ids"]
    })

    # Assumptions/missing information cannot establish a decisive status.
    if winning_category in (
        InfoCategory.ASSUMPTION.value,
        InfoCategory.MISSING.value,
    ):
        winning_status = RuleStatus.NOT_ENOUGH_INFO.value
        evidence_ids = []

    elif not evidence_ids:
        winning_category = InfoCategory.MISSING.value
        winning_status = RuleStatus.NOT_ENOUGH_INFO.value

    unresolved = (
        winning_status == RuleStatus.NOT_ENOUGH_INFO.value
        or winning_category in (
            InfoCategory.ASSUMPTION.value,
            InfoCategory.MISSING.value,
        )
    )

    # Only use keys from samples agreeing on the winning status/category.
    key_samples = [
        sample for sample in winning_samples
        if sample["category"] == winning_category
    ]
    fact_key = _consensus_fact_key(key_samples) if unresolved else None

    explanations = [
        sample["explanation"]
        for sample in winning_samples
        if sample["explanation"]
    ]

    explanation = explanations[0] if explanations else (
        f"{winning_votes} of {n_samples} requested samples agreed on "
        f"status {winning_status}; {len(samples)} samples were valid."
    )

    return RuleCheck(
        patient_id=note.patient_id,
        trial_id=rule.trial_id,
        rule_id=rule.rule_id,
        kind=rule.kind,
        status=RuleStatus(winning_status),
        category=InfoCategory(winning_category),
        evidence_ids=evidence_ids,
        confidence=confidence,
        n_samples=n_samples,
        explanation=explanation,
        fact_key=fact_key,
    )