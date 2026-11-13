"""Load and validate the shared TrialGPT corpus for one cohort."""

from __future__ import annotations

import json
from pathlib import Path

SUPPORTED_COHORTS = ("sigir", "trec_2021", "trec_2022")


# Load one cohort's shared corpus without involving patient queries or labels.
def load_trial_corpus(raw: Path, cohort: str) -> list[dict]:
    """Load validated trial dictionaries from ``raw/trialgpt/{cohort}/corpus.jsonl``."""
    if cohort not in SUPPORTED_COHORTS:
        supported = ", ".join(SUPPORTED_COHORTS)
        raise ValueError(f"unsupported cohort {cohort!r}; expected one of: {supported}")

    path = Path(raw) / "trialgpt" / cohort / "corpus.jsonl"
    trials: list[dict] = []
    first_line_by_id: dict[str, int] = {}
    with path.open(encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            try:
                trial = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: malformed JSON: {exc.msg}"
                ) from exc

            if not isinstance(trial, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")

            trial_id = trial.get("_id")
            if not isinstance(trial_id, str) or not trial_id.strip():
                raise ValueError(f"{path}:{line_number}: _id must be a non-empty string")
            if trial_id in first_line_by_id:
                raise ValueError(
                    f"{path}:{line_number}: duplicate trial _id {trial_id!r}; "
                    f"first seen on line {first_line_by_id[trial_id]}"
                )
            first_line_by_id[trial_id] = line_number

            for field in ("title", "text"):
                value = trial.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{path}:{line_number}: {field} must be a non-empty string"
                    )

            trials.append(trial)

    return trials


__all__ = ["SUPPORTED_COHORTS", "load_trial_corpus"]
