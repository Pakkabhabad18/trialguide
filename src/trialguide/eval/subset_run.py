"""Compare a re-run at another masking fraction on a subset of the test patients (T35).

    python -m trialguide.eval.subset_run --fraction 60 --phase1 PHASE1_DIR --phase2 PHASE2_DIR --out results/t35_60pct

PHASE1_DIR holds the Kaggle phase 1 output (systems_a/, systems_b/: B6, B2, B5 at the new fraction) and
PHASE2_DIR the phase 2 output (asking_a/, asking_b/: B7 and B4 with the Llama doctor). Both halves are merged.
The reference runs come from the committed cluster test run (results/test_run/) restricted to the same patients:
B1 (the checker on the full note, as in the README) and B6, B7, B4 at 40% masking.

Which contrasts are clean: B7 vs B6 vs B4 at the new fraction come from ONE environment (Kaggle) on the SAME patients
and trials, so they are paired and comparable. Anything that sets a Kaggle number against a cluster number (B1, or the 40% run)
also carries the environment difference measured in results/resplits/README.md, and is labelled as such.
All intervals are paired patient-level bootstraps (eval.stats). With a few dozen patients they are wide.
Owner: Murthy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trialguide.eval.stats import (BootResult, MIN_GAP, mean_of, paired_bootstrap, recovery_difference_stat,
                                   recovery_stat)


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_halves(root: Path, prefix: str, system: str, condition: str = None) -> dict:
    """{patient: ndcg} for one system over both halves of a Kaggle output folder."""
    out = {}
    for h in "ab":
        s = _load(root / f"{prefix}_{h}" / "summary.json")
        entry = s["results"][system][condition] if condition else s["systems"][system]
        out.update({p: v["ndcg"] for p, v in entry["per_patient"].items()})
    return out


def asking_extras(root: Path, system: str) -> dict:
    """Questions per patient (patient-weighted) and the doctor's answers by certainty, over both halves."""
    n, q, answers = 0, 0.0, {}
    for h in "ab":
        e = _load(root / f"asking_{h}" / "summary.json")["systems"][system]
        n += e["patients"]
        q += e["questions_per_patient"] * e["patients"]
        for k, v in e["answers"].items():
            answers[k] = answers.get(k, 0) + v
    return {"patients": n, "questions_per_patient": q / n, "answers": answers}


def cluster_reference(test_run: Path, patients: list) -> dict:
    """B1 (B6 on the full note), and B6, B7, B4 at 40%, per patient, from the committed cluster test run."""
    base = _load(test_run / "systems_B6_B2_B5.json")["results"]
    ask = _load(test_run / "asking.json")["systems"]
    pick = lambda d: {p: d[p]["ndcg"] for p in patients}
    return {"B1": pick(base["B6"]["full"]["per_patient"]), "B6@40": pick(base["B6"]["rule40"]["per_patient"]),
            "B7@40": pick(ask["B7"]["per_patient"]), "B4@40": pick(ask["B4"]["per_patient"])}


def difference(a: np.ndarray, b: np.ndarray, n_boot: int = 10000) -> BootResult:
    return paired_bootstrap({"d": a - b}, mean_of("d"), n_boot)


def analyse(phase1: Path, phase2: Path, test_run: Path, fraction: int, n_boot: int = 10000) -> dict:
    cond = f"rule{fraction}"
    new = {s: merge_halves(phase1, "systems", s, cond) for s in ("B6", "B2", "B5")}
    new.update({s: merge_halves(phase2, "asking", s) for s in ("B7", "B4")})
    patients = sorted(new["B6"])
    for s, d in new.items():
        if sorted(d) != patients:
            raise ValueError(f"{s} covers different patients than B6 ({len(d)} vs {len(patients)})")
    ref = cluster_reference(test_run, patients)
    arr = {f"{s}@{fraction}": np.array([d[p] for p in patients]) for s, d in new.items()}
    arr.update({k: np.array([d[p] for p in patients]) for k, d in ref.items()})
    f = fraction
    res = {"fraction": fraction, "patients": len(patients),
           "mean_ndcg": {k: float(v.mean()) for k, v in arr.items()},
           "within_kaggle_run": {   # clean contrasts: same environment, same patients and trials
               "B7_minus_B6": difference(arr[f"B7@{f}"], arr[f"B6@{f}"], n_boot),
               "B7_minus_B4": difference(arr[f"B7@{f}"], arr[f"B4@{f}"], n_boot),
               "B7_minus_B2": difference(arr[f"B7@{f}"], arr[f"B2@{f}"], n_boot),
               "B4_minus_B6": difference(arr[f"B4@{f}"], arr[f"B6@{f}"], n_boot)},
           "cluster_40_same_patients": {   # cluster numbers, same 24 patients
               "B7_minus_B6": difference(arr["B7@40"], arr["B6@40"], n_boot),
               "B7_minus_B4": difference(arr["B7@40"], arr["B4@40"], n_boot),
               "B1_minus_B6": difference(arr["B1"], arr["B6@40"], n_boot)},
           "mixed_environment": {   # Kaggle number against a cluster number: includes the environment difference
               "B1_minus_B6_at_new_fraction": difference(arr["B1"], arr[f"B6@{f}"], n_boot)}}
    gap = float(arr["B1"].mean() - arr[f"B6@{f}"].mean())
    res["B1_B6_gap_at_new_fraction"] = gap
    if abs(gap) >= MIN_GAP:
        a = {"b1": arr["B1"], "b6": arr[f"B6@{f}"], "b7": arr[f"B7@{f}"], "b4": arr[f"B4@{f}"]}
        res["mixed_environment"]["recovery_B7"] = paired_bootstrap(a, recovery_stat("b7", "b1", "b6"), n_boot)
        res["mixed_environment"]["recovery_B4"] = paired_bootstrap(a, recovery_stat("b4", "b1", "b6"), n_boot)
        res["mixed_environment"]["recovery_gain_B7_minus_B4"] = paired_bootstrap(
            a, recovery_difference_stat("b7", "b4", "b1", "b6"), n_boot)
    else:
        res["mixed_environment"]["recovery"] = f"not reported: B1 minus B6 gap {gap:.3f} is under {MIN_GAP}"
    res["asking"] = {s: asking_extras(phase2, s) for s in ("B7", "B4")}
    cl = _load(test_run / "asking.json")["systems"]
    res["asking_cluster_40_all_73"] = {s: {"questions_per_patient": cl[s]["questions_per_patient"], "answers": cl[s]["answers"]}
                                      for s in ("B7", "B4")}
    res["per_patient_ndcg"] = {k: {p: float(v) for p, v in zip(patients, a)} for k, a in arr.items()}
    return res


def _plain(o):
    if isinstance(o, BootResult):
        return {"estimate": o.estimate, "low": o.low, "high": o.high, "dropped_resamples": o.n_undefined}
    raise TypeError(type(o))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fraction", type=int, required=True, help="masking percentage, e.g. 60")
    ap.add_argument("--phase1", type=Path, required=True)
    ap.add_argument("--phase2", type=Path, required=True)
    ap.add_argument("--test-run", type=Path, default=Path("results/test_run"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    res = analyse(args.phase1, args.phase2, args.test_run, args.fraction)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "subset_run.json").write_text(json.dumps(res, indent=1, default=_plain) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "per_patient_ndcg"}, indent=1, default=_plain))


if __name__ == "__main__":
    main()
