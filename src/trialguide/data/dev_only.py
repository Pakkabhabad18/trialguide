"""Strict split-filtered readers for the T18 development retrieval evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from trialguide.data.patients import Patient, split_sentences
from trialguide.data.splits import SPLIT_DIR
from trialguide.schemas import Split

COHORTS = ("sigir", "trec_2021", "trec_2022")


# Read the single approved ID list without opening CAL or TEST split files.
def load_dev_patient_ids(split_dir: Path = SPLIT_DIR) -> frozenset[str]:
    """Return patient IDs from ``dev.txt`` only, rejecting blank or duplicate entries."""
    path = Path(split_dir) / "dev.txt"
    ids = path.read_text(encoding="utf-8").splitlines()
    normalized = [patient_id.strip() for patient_id in ids if patient_id.strip()]
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(f"invalid DEV patient ID list in {path}")
    return frozenset(normalized)


# Read individual characters so non-DEV note payloads are never held as line strings.
class _CharStream:
    """Provide one-character reads with a single-character pushback slot."""

    def __init__(self, handle):
        """Wrap the open text file used to scan one JSONL record."""
        self.handle = handle
        self.pending = ""

    def read(self) -> str:
        """Read one character, first returning any character pushed back by the scanner."""
        if self.pending:
            value, self.pending = self.pending, ""
            return value
        return self.handle.read(1)

    def unread(self, value: str) -> None:
        """Push back one delimiter so the containing JSON scanner can consume it."""
        if value:
            if self.pending:
                raise ValueError("only one character can be pushed back")
            self.pending = value


# Decode a JSON string only when it is a top-level key or the selected record ID.
def _read_json_string(stream: _CharStream, opening_quote: bool = False) -> str:
    """Read and decode one JSON string without retaining unrelated field contents."""
    if not opening_quote and stream.read() != '"':
        raise ValueError("expected JSON string")
    encoded = ['"']
    escaped = False
    while True:
        char = stream.read()
        if not char or char in "\r\n":
            raise ValueError("unterminated JSON string")
        encoded.append(char)
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return json.loads("".join(encoded))


# Skip one non-ID JSON value with bounded memory.
def _skip_json_value(stream: _CharStream, first: str) -> None:
    """Advance over a string, scalar, list, or object without decoding its value."""
    if first == '"':
        escaped = False
        while True:
            char = stream.read()
            if not char or char in "\r\n":
                raise ValueError("unterminated JSON string")
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return
    if first in "[{":
        stack = ["]" if first == "[" else "}"]
        in_string = False
        escaped = False
        while stack:
            char = stream.read()
            if not char or char in "\r\n":
                raise ValueError("unterminated JSON container")
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "[":
                stack.append("]")
            elif char == "{":
                stack.append("}")
            elif char in "]}":
                if stack.pop() != char:
                    raise ValueError("mismatched JSON container")
        return

    char = first
    while char and char not in ",}]\r\n" and not char.isspace():
        char = stream.read()
    stream.unread(char)


# Extract only the top-level ID so non-DEV payloads are never decoded or buffered.
def _scan_top_level_id(stream: _CharStream, source: Path, line_number: int) -> str:
    """Read JSON object structure and decode only top-level keys and the ``_id`` value."""
    try:
        char = stream.read()
        while char and char.isspace() and char not in "\r\n":
            char = stream.read()
        if char != "{":
            raise ValueError("record is not a JSON object")
        while True:
            char = stream.read()
            while char and char.isspace() and char not in "\r\n":
                char = stream.read()
            if char == ",":
                continue
            if char != '"':
                raise ValueError("record has no top-level _id")
            key = _read_json_string(stream, opening_quote=True)
            char = stream.read()
            while char and char.isspace() and char not in "\r\n":
                char = stream.read()
            if char != ":":
                raise ValueError("malformed JSON object")
            char = stream.read()
            while char and char.isspace() and char not in "\r\n":
                char = stream.read()
            if key == "_id":
                if char != '"':
                    raise ValueError("_id is not a string")
                patient_id = _read_json_string(stream, opening_quote=True)
                if not patient_id.strip():
                    raise ValueError("_id is empty")
                return patient_id
            _skip_json_value(stream, char)
            char = stream.read()
            while char and char.isspace() and char not in "\r\n":
                char = stream.read()
            if char not in ",}":
                raise ValueError("malformed JSON object")
            stream.unread(char)
    except (json.JSONDecodeError, ValueError):
        raise ValueError(f"{source}:{line_number}: cannot identify record _id") from None


# Discard the rest of one non-DEV record without buffering its note payload.
def _skip_record_remainder(stream: _CharStream) -> None:
    """Consume characters through the current JSONL newline without retaining them."""
    while char := stream.read():
        if char == "\n":
            return


# Load only selected note records, parsing full JSON solely for DEV IDs.
def load_dev_patients(raw: Path, split_dir: Path = SPLIT_DIR) -> dict[str, Patient]:
    """Return the existing ``{patient_id: Patient}`` shape for DEV patients only."""
    dev_ids = load_dev_patient_ids(split_dir)
    found: dict[str, Patient] = {}
    for cohort in COHORTS:
        source = Path(raw) / "trialgpt" / cohort / "queries.jsonl"
        with source.open(encoding="utf-8") as handle:
            line_number = 0
            while True:
                record_start = handle.tell()
                if not handle.read(1):
                    break
                handle.seek(record_start)
                line_number += 1
                stream = _CharStream(handle)
                patient_id = _scan_top_level_id(stream, source, line_number)
                if patient_id not in dev_ids:
                    _skip_record_remainder(stream)
                    continue
                if patient_id in found:
                    raise ValueError(f"{source}:{line_number}: duplicate DEV patient record")
                handle.seek(record_start)
                line = handle.readline()
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{line_number}: malformed DEV JSON record") from exc
                if (not isinstance(record, dict) or record.get("_id") != patient_id
                        or not isinstance(record.get("text"), str) or not record["text"].strip()):
                    raise ValueError(f"{source}:{line_number}: invalid DEV patient record")
                found[patient_id] = Patient(
                    patient_id,
                    cohort,
                    Split.DEV,
                    split_sentences(record["text"], cohort),
                )

    if set(found) != set(dev_ids):
        raise ValueError(f"DEV note records are missing from the cohort query files: {len(set(dev_ids) - set(found))}")
    return found


# Stream qrels by patient ID and parse grade fields only for DEV rows.
def load_dev_gold_grades(raw: Path, split_dir: Path = SPLIT_DIR) -> dict[str, dict[str, int]]:
    """Return the existing ``{patient_id: {trial_id: grade}}`` shape for DEV qrels only."""
    dev_ids = load_dev_patient_ids(split_dir)
    gold: dict[str, dict[str, int]] = {patient_id: {} for patient_id in dev_ids}
    for cohort in COHORTS:
        source = Path(raw) / "trialgpt" / cohort / "qrels" / "test.tsv"
        with source.open(encoding="utf-8", newline="") as handle:
            header = next(handle, "").rstrip("\r\n").split("\t")
            if len(header) != 3:
                raise ValueError(f"{source}: expected a three-column qrels header")
            line_number = 1
            while True:
                first_field = []
                char = handle.read(1)
                if not char:
                    break
                while char and char not in "\t\r\n":
                    first_field.append(char)
                    char = handle.read(1)
                line_number += 1
                if char in "\r\n" and not first_field:
                    if char == "\r" and handle.read(1) != "\n":
                        raise ValueError(f"{source}:{line_number}: malformed blank qrel line")
                    continue
                if char != "\t":
                    raise ValueError(f"{source}:{line_number}: cannot identify qrel patient ID")
                patient_id = "".join(first_field).strip()
                if patient_id not in dev_ids:
                    while char := handle.read(1):
                        if char == "\n":
                            break
                    continue
                rest_of_line = handle.readline()
                line = patient_id + "\t" + rest_of_line
                fields = next(csv.reader([line], delimiter="\t"))
                if len(fields) != 3 or not fields[1].strip():
                    raise ValueError(f"{source}:{line_number}: invalid DEV qrel record")
                try:
                    grade = int(fields[2])
                except ValueError as exc:
                    raise ValueError(f"{source}:{line_number}: invalid DEV qrel grade") from exc
                gold[patient_id][fields[1]] = grade
    return gold


__all__ = ["load_dev_patient_ids", "load_dev_patients", "load_dev_gold_grades"]
