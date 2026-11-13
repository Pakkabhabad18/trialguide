"""Recorded checker calls, so the demo also runs on a laptop with no GPU.

A bundle is one JSON file with everything the demo needs for a few development patients:
their visible and hidden sentences, their candidate trials and rules, and every checker
result needed to answer the questions the demo can ask (recorded once on a GPU by
trialguide.demo.build_bundle). The app then replays these results; a path that was not
recorded is refused with a clear message instead of being guessed.
Owner: Prakash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from trialguide.schemas import RuleKind, Split, TrialRule, VisibleNote


class MissingRecording(KeyError):
    """The bundle has no recorded check for this note and rule."""


def key(note: VisibleNote, rule: TrialRule) -> str:
    """A stable id for one checker call: the exact visible note text plus the rule."""
    raw = "\n".join([note.text(), rule.trial_id, rule.rule_id, rule.kind.value, rule.text])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class RecordingChecker:
    """Calls the real checker and keeps every result (live mode, and building the bundle)."""

    def __init__(self, check, store: dict | None = None):
        self.check, self.store = check, store if store is not None else {}

    def __call__(self, note, rule):
        k = key(note, rule)
        if k not in self.store:
            self.store[k] = self.check(note, rule)
        return self.store[k]


class ReplayChecker:
    """Returns recorded results only (laptop mode)."""

    def __init__(self, store: dict):
        self.store = store

    def __call__(self, note, rule):
        try:
            return self.store[key(note, rule)]
        except KeyError:
            raise MissingRecording("this answer path was not recorded; run the demo in live mode") from None


def save_bundle(path: Path, patients: dict, store: dict, meta: dict) -> None:
    from trialguide.baselines.run_systems import to_json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"meta": meta, "patients": patients,
                                      "checks": {k: to_json(v) for k, v in store.items()}}, indent=1))


def load_bundle(path: Path) -> tuple[dict, dict, dict]:
    """(patients, store of RuleChecks, meta)."""
    from trialguide.policy.calibrate import check_from_json
    data = json.loads(Path(path).read_text())
    return data["patients"], {k: check_from_json(v) for k, v in data["checks"].items()}, data["meta"]


def patient_note(p: dict, pid: str) -> VisibleNote:
    return VisibleNote(pid, Split(p.get("split", "dev")), {int(i): s for i, s in p["visible"].items()})


def patient_rules(p: dict) -> list[TrialRule]:
    return [TrialRule(t["trial_id"], r["rule_id"], RuleKind(r["kind"]), r["text"])
            for t in p["trials"] for r in t["rules"]]
