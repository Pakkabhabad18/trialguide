"""T3: the download manifest and the offline parts of the downloader."""

import json

from trialguide.data import download as dl


def test_manifest_covers_all_three_cohorts():
    files = dl.manifest()
    for corpus in ("sigir", "trec_2021", "trec_2022"):
        for name in ("queries.jsonl", "corpus.jsonl", "retrieved_trials.json", "qrels/test.tsv"):
            assert f"trialgpt/{corpus}/{name}" in files
    assert "annotations/criterion_annotations.parquet" in files


def test_trial_info_is_opt_in():
    # 1.2 GB, and the pipeline reads trial text from retrieved_trials.json instead
    assert "trialgpt/trial_info.json" not in dl.manifest()
    assert "trialgpt/trial_info.json" in dl.manifest(with_trial_info=True)


def test_urls_are_pinned_not_floating():
    for url in dl.manifest().values():
        assert "/main/" not in url and "/resolve/main" not in url


def test_existing_file_is_verified_not_redownloaded(tmp_path, monkeypatch):
    # Pre-place every file; a download attempt would fail the test
    monkeypatch.setattr(dl, "_fetch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")))
    sums = {}
    for rel in dl.manifest():
        p = tmp_path / "raw" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(rel.encode())
        sums[rel] = dl.sha256_of(p)
    sumfile = tmp_path / "checksums.json"
    sumfile.write_text(json.dumps(sums))
    monkeypatch.setattr(dl, "CHECKSUM_FILE", sumfile)
    assert dl.download(tmp_path) == 0


def test_checksum_mismatch_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_fetch", lambda *a, **k: None)
    sums = {}
    for rel in dl.manifest():
        p = tmp_path / "raw" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"data")
        sums[rel] = "0" * 64  # wrong on purpose
    sumfile = tmp_path / "checksums.json"
    sumfile.write_text(json.dumps(sums))
    monkeypatch.setattr(dl, "CHECKSUM_FILE", sumfile)
    assert dl.download(tmp_path) == len(sums)
