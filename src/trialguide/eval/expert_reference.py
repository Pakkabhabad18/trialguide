"""Expert rule labels as a reference for our rule ids (task T30 needs it; also rule-level accuracy).

The TrialGPT criterion annotations give, for 53 SIGIR patients, an expert label for single
criteria of their trials: patient_id, trial_id, criterion_type, criterion_text, expert_eligibility.
Our checker works on rules named "inc_3", "exc_0", ... made by pipeline.rules_of_trial from the
trial text. This module links the two: each annotation row is matched to the rule of the same
trial and kind whose text is the same criterion.

Matching: texts are compared after lower-casing and collapsing white space and punctuation.
If no rule matches exactly, the closest rule is taken only if it is clearly the closest
(similarity >= 0.9 and no other rule within 0.05); otherwise the row is left unmatched and
reported, never guessed. Owner: Prakash.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from trialguide.schemas import RuleKind, RuleStatus

# The six expert labels, per kind of rule (MET = the rule's condition holds, decision D15)
EXPERT_TO_STATUS = {
    (RuleKind.INCLUSION, "included"): RuleStatus.MET,
    (RuleKind.INCLUSION, "not included"): RuleStatus.NOT_MET,
    (RuleKind.EXCLUSION, "excluded"): RuleStatus.MET,
    (RuleKind.EXCLUSION, "not excluded"): RuleStatus.NOT_MET,
}
SHARED = {"not enough information": RuleStatus.NOT_ENOUGH_INFO,
          "not applicable": RuleStatus.NOT_APPLICABLE}

MIN_SIMILARITY = 0.9
MIN_MARGIN = 0.05


def expert_status(kind: RuleKind, label: str) -> RuleStatus:
    """'included' / 'not excluded' / ... -> RuleStatus for this kind of rule."""
    label = label.strip().lower()
    if label in SHARED:
        return SHARED[label]
    if (kind, label) not in EXPERT_TO_STATUS:
        raise ValueError(f"label {label!r} does not fit a {kind.value} rule")
    return EXPERT_TO_STATUS[(kind, label)]


def normalise(text: str) -> str:
    """Lower case, punctuation to spaces, single spaces: what two copies of a criterion share."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def match_rule(text: str, rules: list):
    """The rule (from one trial, one kind) that this criterion text is, or None.
    Returns (rule, how) with how = "exact" or "fuzzy"."""
    target = normalise(text)
    if not target:
        return None, "empty"
    exact = [r for r in rules if normalise(r.text) == target]
    if len(exact) == 1:
        return exact[0], "exact"
    scored = sorted(((SequenceMatcher(None, target, normalise(r.text)).ratio(), r.rule_id, r) for r in rules),
                    reverse=True)
    if not scored:
        return None, "no rules"
    best = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best[0] >= MIN_SIMILARITY and best[0] - second >= MIN_MARGIN:
        return best[2], "fuzzy"
    return None, "unmatched"


def build_reference(annotations, rules_by_trial: dict) -> tuple[dict, dict]:
    """annotations: rows with patient_id, trial_id, criterion_type, criterion_text, expert_eligibility
    (a pandas DataFrame or a list of dicts). rules_by_trial: trial_id -> list of TrialRule.

    Returns (reference, report). reference maps (patient_id, trial_id, rule_id) -> RuleStatus.
    report counts exact, fuzzy and unmatched rows and lists the unmatched ones.
    """
    rows = annotations.to_dict("records") if hasattr(annotations, "to_dict") else list(annotations)
    reference, report = {}, {"rows": len(rows), "exact": 0, "fuzzy": 0, "unmatched": [], "conflicts": 0}
    for row in rows:
        kind = RuleKind(row["criterion_type"].strip().lower())
        rules = [r for r in rules_by_trial.get(row["trial_id"], []) if r.kind == kind]
        rule, how = match_rule(row.get("criterion_text") or "", rules)
        if rule is None:
            report["unmatched"].append({"patient_id": row["patient_id"], "trial_id": row["trial_id"],
                                        "criterion_type": kind.value, "why": how,
                                        "text": (row.get("criterion_text") or "")[:120]})
            continue
        report[how] += 1
        key = (row["patient_id"], row["trial_id"], rule.rule_id)
        status = expert_status(kind, row["expert_eligibility"])
        if key in reference and reference[key] != status:
            report["conflicts"] += 1        # two annotation rows for one rule disagree: drop it
            reference[key] = None
            continue
        reference.setdefault(key, status)
    reference = {k: v for k, v in reference.items() if v is not None}
    report["matched_rules"] = len(reference)
    return reference, report


def load_reference(raw: Path) -> tuple[dict, dict]:
    """The expert reference for every annotated SIGIR patient, from the downloaded files (T3)."""
    import pandas as pd
    from trialguide.pipeline import rules_of_trial
    ann = pd.read_parquet(Path(raw) / "annotations" / "criterion_annotations.parquet")
    rules_by_trial = {}
    with open(Path(raw) / "trialgpt" / "sigir" / "retrieved_trials.json", encoding="utf-8") as f:
        for entry in json.load(f):
            for label in ("0", "1", "2"):
                for trial in entry.get(label, []):
                    rules_by_trial.setdefault(trial["NCTID"], rules_of_trial(trial))
    return build_reference(ann, rules_by_trial)
