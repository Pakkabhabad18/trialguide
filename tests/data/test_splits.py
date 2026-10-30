"""T8: the frozen split files, and the algorithm that produced them."""

import json
from pathlib import Path

import pandas as pd
import pytest

from trialguide.data import splits
from trialguide.schemas import Split


# ---- invariants of the committed files (no downloaded data needed) ----

def test_frozen_sizes_and_disjoint():
    s = splits.load_split()
    assert {k: len(v) for k, v in s.items()} == {Split.DEV: 37, Split.CAL: 73, Split.TEST: 73}
    everyone = s[Split.DEV] + s[Split.CAL] + s[Split.TEST]
    assert len(everyone) == len(set(everyone)) == 183


def test_frozen_files_unchanged():
    # If this fails someone edited a frozen file. Revert it; never update the hashes.
    info = json.loads((splits.SPLIT_DIR / "split_info.json").read_text())
    assert splits.file_hashes() == info["sha256"]


def test_frozen_cohort_mix():
    s = splits.load_split()
    for ids in s.values():
        cohorts = {splits.cohort_of(p) for p in ids}
        assert cohorts == {"sigir", "trec_2021", "trec_2022"}


# ---- the algorithm, on a small synthetic dataset ----

def _raw(tmp_path, labelled_rows):
    raw = tmp_path / "raw"
    layout = {"sigir": ["s1", "s2", "s3", "s4", "s5", "s6"], "trec_2021": [f"a{i}" for i in range(5)],
              "trec_2022": [f"b{i}" for i in range(3)]}
    for c, ids in layout.items():
        d = raw / "trialgpt" / c
        (d / "qrels").mkdir(parents=True)
        (d / "queries.jsonl").write_text("\n".join(json.dumps({"_id": i, "text": "n"}) for i in ids))
        pd.DataFrame({"query-id": ids, "corpus-id": "N", "score": 1}).to_csv(d / "qrels" / "test.tsv", sep="\t", index=False)
    ann = pd.DataFrame({"patient_id": [p for p, n in labelled_rows.items() for _ in range(n)]})
    (raw / "annotations").mkdir()
    ann.to_parquet(raw / "annotations" / "criterion_annotations.parquet")
    return raw


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(splits, "TARGET", {Split.DEV: 3, Split.CAL: 5, Split.TEST: 6})


def test_algorithm_properties(tmp_path, small):
    raw = _raw(tmp_path, {"s1": 4, "s2": 5, "s3": 9, "s4": 10})   # 4 labelled patients
    out = splits.build_split(raw)
    assert {k: len(v) for k, v in out.items()} == {Split.DEV: 3, Split.CAL: 5, Split.TEST: 6}
    assert not set(out[Split.DEV]) & {"s1", "s2", "s3", "s4"}          # labelled never in dev
    assert len({p for v in out.values() for p in v}) == 14
    # neighbours by rule count (s1,s2) and (s3,s4) are split across cal and test
    for pair in (("s1", "s2"), ("s3", "s4")):
        assert {p in out[Split.CAL] for p in pair} == {True, False}


def test_algorithm_is_deterministic(tmp_path, small):
    raw = _raw(tmp_path, {"s1": 4, "s2": 5, "s3": 9, "s4": 10})
    assert splits.build_split(raw) == splits.build_split(raw)


# ---- the committed files against the real data (skipped where data is not downloaded) ----

def test_frozen_split_matches_code_on_real_data():
    raw = Path("data/raw")
    if not (raw / "annotations").exists():
        pytest.skip("run scripts/download_data.sh first")
    assert splits.build_split(raw) == splits.load_split()
