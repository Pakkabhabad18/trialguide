"""The frozen patient split (T8): 37 development / 73 calibration / 73 test.

    python -m trialguide.data.splits            # check the committed files still match the code
    python -m trialguide.data.splits --write    # write splits/*.txt (done once, then frozen)

Rules:
- A patient is usable only with a note and expert trial labels (183 patients).
- Patients with expert rule labels (52, all SIGIR) go to calibration and test only, half each,
  paired by number of labelled rules so both splits get a similar amount of gold evidence.
- The other 131 patients are spread over dev / cal / test in proportion 37 : 47 : 47,
  separately for each cohort, so every split sees SIGIR, TREC 2021 and TREC 2022.
- Everything is driven by one seed, so the files can be regenerated and checked.
Once written, the .txt files are never edited (they are the split; this code only verifies them).
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from trialguide.data.audit import cohort_of, load_queries, patients_with_trial_labels
from trialguide.schemas import Split

SEED = 20261027                      # fixed once, chosen before running (not searched for)
TARGET = {Split.DEV: 37, Split.CAL: 73, Split.TEST: 73}
SPLIT_DIR = Path(__file__).resolve().parents[3] / "splits"


def usable_patients(raw: Path) -> dict:
    """patient_id -> cohort, for patients with a note and expert trial labels."""
    out = {}
    for cohort, notes in load_queries(raw).items():
        labelled = patients_with_trial_labels(raw, cohort)
        out.update({p: cohort for p in notes if p in labelled})
    return out


def rule_rows_per_patient(raw: Path) -> dict:
    ann = pd.read_parquet(raw / "annotations" / "criterion_annotations.parquet", columns=["patient_id"])
    return ann.patient_id.value_counts().to_dict()


def _largest_remainder(n: int, weights: dict) -> dict:
    """Split n items over keys in proportion to weights, whole numbers, summing to n."""
    total = sum(weights.values())
    raw = {k: n * w / total for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (raw[k] - out[k], str(k)), reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def build_split(raw: Path, seed: int = SEED) -> dict:
    """Return {Split: sorted list of patient ids}."""
    rng = random.Random(seed)
    usable = usable_patients(raw)
    rows = rule_rows_per_patient(raw)
    assign = {s: [] for s in Split}

    # 1) Labelled patients: sort by rule count, pair neighbours, coin-flip inside each pair
    labelled = sorted((p for p in usable if p in rows), key=lambda p: (rows[p], p))
    assert len(labelled) % 2 == 0, "odd number of labelled patients; pairing needs a rule"
    for a, b in zip(labelled[0::2], labelled[1::2]):
        first, second = (Split.CAL, Split.TEST) if rng.random() < 0.5 else (Split.TEST, Split.CAL)
        assign[first].append(a)
        assign[second].append(b)

    # 2) Unlabelled patients: per cohort, shuffle, then give each split its share.
    #    The last cohort takes exactly what capacity is left, so totals hit the targets.
    capacity = {s: TARGET[s] - len(assign[s]) for s in Split}
    strata = {}
    for p, c in sorted(usable.items()):
        if p not in rows:
            strata.setdefault(c, []).append(p)
    cohorts = sorted(strata)
    for i, c in enumerate(cohorts):
        ids = strata[c][:]
        rng.shuffle(ids)
        if i == len(cohorts) - 1:
            quota = dict(capacity)
        else:
            quota = _largest_remainder(len(ids), {s: capacity[s] for s in Split})
        assert sum(quota.values()) == len(ids), (c, quota, len(ids))
        pos = 0
        for s in Split:
            assign[s] += ids[pos:pos + quota[s]]
            pos += quota[s]
            capacity[s] -= quota[s]
    assert all(v == 0 for v in capacity.values()), capacity
    return {s: sorted(v) for s, v in assign.items()}


def summarise(split: dict, raw: Path) -> dict:
    """Counts per split and cohort, plus how many expert-labelled rules each split holds."""
    rows = rule_rows_per_patient(raw)
    info = {}
    for s, ids in split.items():
        by_cohort = {}
        for p in ids:
            by_cohort[cohort_of(p)] = by_cohort.get(cohort_of(p), 0) + 1
        info[s.value] = {"patients": len(ids), "by_cohort": dict(sorted(by_cohort.items())),
                         "patients_with_rule_labels": sum(p in rows for p in ids),
                         "rule_label_rows": int(sum(rows.get(p, 0) for p in ids))}
    return info


def write_split(split: dict, directory: Path = SPLIT_DIR) -> None:
    for s, ids in split.items():
        (directory / f"{s.value}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8", newline="\n")


def load_split(directory: Path = SPLIT_DIR) -> dict:
    """Read the frozen files: {Split: [patient ids]}. This is what the rest of the code uses."""
    return {s: (directory / f"{s.value}.txt").read_text(encoding="utf-8").split() for s in Split}


def file_hashes(directory: Path = SPLIT_DIR) -> dict:
    """SHA-256 of each split file with line endings normalised, so Windows checkouts agree."""
    out = {}
    for s in Split:
        data = (directory / f"{s.value}.txt").read_bytes().replace(b"\r\n", b"\n")
        out[f"{s.value}.txt"] = hashlib.sha256(data).hexdigest()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="data")
    p.add_argument("--write", action="store_true", help="write the split files (first time only)")
    a = p.parse_args(argv)
    raw = Path(a.root) / "raw"
    built = build_split(raw)
    summary = summarise(built, raw)
    print(json.dumps(summary, indent=2))
    if a.write:
        if (SPLIT_DIR / "dev.txt").exists():
            raise SystemExit("splits/dev.txt exists: the split is frozen and is never rewritten")
        write_split(built)
        info = {"seed": SEED, "summary": summary, "sha256": file_hashes()}
        (SPLIT_DIR / "split_info.json").write_text(json.dumps(info, indent=2) + "\n")
        print("written")
        return 0
    if not (SPLIT_DIR / "dev.txt").exists():
        print("splits/*.txt not written yet (dry run); use --write once")
        return 1
    same = {s: built[s] == load_split()[s] for s in Split}
    print("committed files match the code:", all(same.values()))
    return 0 if all(same.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
