"""T35: merging the two Kaggle halves and comparing with the cluster reference, on hand-made folders."""

import json
from pathlib import Path

import pytest

from trialguide.eval import subset_run as sr


def _pp(values):
    return {p: {"ndcg": v, "p10": 0.5} for p, v in values.items()}


def _write(tmp_path, n=6):
    pats = [f"p{i}" for i in range(n)]
    half = {"a": pats[0::2], "b": pats[1::2]}
    for h, ps in half.items():
        s1 = tmp_path / "p1" / f"systems_{h}"; s1.mkdir(parents=True)
        res = {s: {"rule60": {"per_patient": _pp({p: base + 0.01 * i for i, p in enumerate(ps)})}}
               for s, base in (("B6", 0.60), ("B2", 0.62), ("B5", 0.61))}
        (s1 / "summary.json").write_text(json.dumps({"results": res}))
        s2 = tmp_path / "p2" / f"asking_{h}"; s2.mkdir(parents=True)
        sysm = {s: {"patients": len(ps), "questions_per_patient": 2.5, "answers": {"not_documented": 6, "sure": 1},
                    "per_patient": _pp({p: base + 0.01 * i for i, p in enumerate(ps)})}
                for s, base in (("B7", 0.65), ("B4", 0.64))}
        (s2 / "summary.json").write_text(json.dumps({"systems": sysm}))
    ref = tmp_path / "test_run"; ref.mkdir()
    allp = {p: 0.0 for p in pats}
    mk = lambda base: _pp({p: base + 0.01 * i for i, p in enumerate(pats)})
    (ref / "systems_B6_B2_B5.json").write_text(json.dumps({"results": {"B6": {"full": {"per_patient": mk(0.80)},
                                                                           "rule40": {"per_patient": mk(0.62)}}}}))
    (ref / "asking.json").write_text(json.dumps({"systems": {
        "B7": {"per_patient": mk(0.66), "questions_per_patient": 2.6, "answers": {"sure": 3}},
        "B4": {"per_patient": mk(0.65), "questions_per_patient": 2.8, "answers": {"sure": 2}}}}))
    return tmp_path / "p1", tmp_path / "p2", ref


def test_merge_halves_covers_every_patient(tmp_path):
    p1, p2, _ = _write(tmp_path)
    assert sorted(sr.merge_halves(p1, "systems", "B6", "rule60")) == [f"p{i}" for i in range(6)]
    assert len(sr.merge_halves(p2, "asking", "B7")) == 6


def test_asking_extras_is_patient_weighted_and_sums_answers(tmp_path):
    _, p2, _ = _write(tmp_path)
    e = sr.asking_extras(p2, "B7")
    assert e["patients"] == 6 and e["questions_per_patient"] == pytest.approx(2.5)
    assert e["answers"] == {"not_documented": 12, "sure": 2}


def test_analyse_contrasts(tmp_path):
    p1, p2, ref = _write(tmp_path)
    r = sr.analyse(p1, p2, ref, 60, n_boot=300)
    assert r["patients"] == 6
    assert r["within_kaggle_run"]["B7_minus_B6"].estimate == pytest.approx(0.05)     # 0.65 vs 0.60 at every patient
    assert r["within_kaggle_run"]["B7_minus_B4"].estimate == pytest.approx(0.01)
    assert r["cluster_40_same_patients"]["B1_minus_B6"].estimate == pytest.approx(0.18)
    assert r["B1_B6_gap_at_new_fraction"] == pytest.approx(0.215)   # B1 mean 0.825 over 6 patients, B6 mean 0.61
    assert "recovery_B7" in r["mixed_environment"]


def test_different_patient_sets_are_an_error(tmp_path):
    p1, p2, ref = _write(tmp_path)
    f = p2 / "asking_a" / "summary.json"
    s = json.loads(f.read_text()); del s["systems"]["B4"]["per_patient"]["p0"]; f.write_text(json.dumps(s))
    with pytest.raises(ValueError):
        sr.analyse(p1, p2, ref, 60, n_boot=50)
