"""H1 and H2 verdicts and McNemar tests from the saved test run (T36, statistics half).

    python -m trialguide.eval.verdicts --out results/verdicts

No model is called. Everything is computed from files committed in results/test_run/ (and, for one
comparison the cluster run could not support, from the per-rule Kaggle re-run in results/resplits/).

H1 (RQ1): recovery of B7 at least 10 points above fixed-order questioning (B4), with a false-eligible
rate no higher. Recovery is the ratio of mean NDCG@10 gaps, (B7 - B6) / (B1 - B6), with a paired
patient-level bootstrap over the 73 test patients (eval.stats). The false-eligible rate is the team's
rule-level one (eval.recovery): of the rules the full-note checker settles but B6 leaves open, the share
a system settles wrongly toward eligibility.

H2 (RQ2): the harmful-inference rate of the risk-controlled policy against the 5% target, judged by
the proposal's rule on the 95% CI of the primary risk run (supported / contradicted / inconclusive), and
compared with the fixed 0.9 threshold (B5), accept-everything (B2) and an uncorrected tuned threshold.

McNemar (exact): paired rule-level outcomes, "does the system end with the full-note status on a rule
that masking made B6 leave open", for pairs of systems. Rules of one patient are not independent, so
the p-values overstate certainty; the bootstrap intervals are the main result.
Owner of the statistics: Prakash (H1, H2); this module computes them for him from saved outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trialguide.eval.recovery import DECIDED, asking_finals, policy_finals, reference
from trialguide.eval.recovery import counts as recovery_counts
from trialguide.eval.stats import (BootResult, h1_verdict, h2_verdict, mcnemar, paired_bootstrap,
                                   ratio_of_sums, recovery_difference_stat, recovery_stat)

ALPHA = 0.05


def per_patient_ndcg(run: Path) -> dict:
    """{system: {patient: NDCG@10}} for B1 (B6 on the full note), B6 (masked) and the asking systems."""
    out = {}
    base = json.load(open(run / "systems_B6_B2_B5.json"))["results"]
    out["B1"] = {p: v["ndcg"] for p, v in base["B6"]["full"]["per_patient"].items()}
    for name in ("B6", "B2", "B5"):
        out[name] = {p: v["ndcg"] for p, v in base[name]["rule40"]["per_patient"].items()}
    for name, s in json.load(open(run / "asking.json"))["systems"].items():
        out[name] = {p: v["ndcg"] for p, v in s["per_patient"].items()}
    return out


def aligned(ndcg: dict, names) -> tuple:
    """Sorted patient ids and one array per system. Patients with no defined NDCG depend on the gold
    labels only, so every system must have the same set; anything else is an error."""
    sets = {n: set(ndcg[n]) for n in names}
    if len({frozenset(s) for s in sets.values()}) != 1:
        raise ValueError("systems have different patient sets: " + ", ".join(f"{n}={len(s)}" for n, s in sets.items()))
    ids = sorted(sets[names[0]])
    return ids, {n: np.array([ndcg[n][p] for p in ids]) for n in names}


def rule_outcomes(ref: dict, lost_from: dict, final: dict) -> dict:
    """{(patient, rule): True if the system ends with the full-note status} over the rules that masking
    made B6 leave open (the 'lost' rules of eval.recovery)."""
    out = {}
    for pid, rules in ref.items():
        for key, (status, _) in rules.items():
            if status in DECIDED and lost_from[pid].get(key) not in DECIDED:
                out[(pid, key)] = final.get(pid, {}).get(key, "not_enough_info") == status
    return out


def uncorrected_threshold(checks_cal: dict, reference_: dict, alpha: float = ALPHA) -> float:
    """The 'uncorrected tuned' comparator (B5, second variant): the smallest threshold whose EMPIRICAL
    calibration risk is at most alpha, without the conformal correction n/(n+1) R + 1/(n+1)."""
    from trialguide.policy.risk_control import ACCEPT_NONE, counts_at, empirical_risk, is_permissive_inference
    lams = sorted({c.confidence for cs in checks_cal.values() for c in cs
                   if is_permissive_inference(c) and reference_.get((c.patient_id, c.trial_id, c.rule_id)) is not None})
    for lam in lams + [ACCEPT_NONE]:
        if empirical_risk(counts_at(checks_cal, reference_, lam)) <= alpha:
            return lam
    return ACCEPT_NONE


def _boot(d: dict) -> BootResult:
    return BootResult(d["estimate"], d["low"], d["high"], d.get("n_boot", 0), d.get("n_undefined", 0))


def h1(run: Path, n_boot: int = 10000) -> dict:
    names = ["B1", "B6", "B2", "B7", "B4"]
    ids, nd = aligned(per_patient_ndcg(run), names)
    arrays = {k.lower(): v for k, v in nd.items()}
    gap = float(arrays["b1"].mean() - arrays["b6"].mean())
    res = {"patients": len(ids), "oracle_floor_gap": gap, "recovery": {}}
    for s in ("b7", "b4", "b2"):
        res["recovery"][s.upper()] = paired_bootstrap(arrays, recovery_stat(s, "b1", "b6"), n_boot)
    gain = paired_bootstrap(arrays, recovery_difference_stat("b7", "b4", "b1", "b6"), n_boot)
    res["recovery_gain_B7_minus_B4"] = gain
    res["ndcg_difference_B7_minus_B4"] = paired_bootstrap(
        {"d": arrays["b7"] - arrays["b4"]}, lambda x: x["d"].mean(axis=-1), n_boot)

    # false-eligible rate (rule level, the team's definition): B7 against B4 on the same lost rules
    ref = reference(run / "checks_full.jsonl")
    b6 = policy_finals(run / "checks_rule40.jsonl", "B6")
    finals = {n: asking_finals(run / f"final_{n}.json") for n in ("B7", "B4")}
    c = {n: recovery_counts(ref, b6, finals[n]) for n in finals}
    pids = sorted(ref)
    fe = {"lost": [c["B7"][p]["lost"] for p in pids],
          "fe_b7": [c["B7"][p]["false_eligible"] for p in pids],
          "fe_b4": [c["B4"][p]["false_eligible"] for p in pids]}
    rate7, rate4 = ratio_of_sums("fe_b7", "lost"), ratio_of_sums("fe_b4", "lost")
    res["false_eligible"] = {
        "B7": paired_bootstrap(fe, rate7, n_boot), "B4": paired_bootstrap(fe, rate4, n_boot),
        "B7_minus_B4": paired_bootstrap(fe, lambda d: rate7(d) - rate4(d), n_boot),
        "lost_rules": int(sum(fe["lost"]))}
    not_higher = res["false_eligible"]["B7_minus_B4"].estimate <= 0
    res["false_eligible_not_higher"] = bool(not_higher)
    res["verdict"] = h1_verdict(gain, not_higher)
    return res


def h2(risk_test: Path, kaggle_dir: Path, raw: Path = Path("data/raw"), n_boot: int = 10000) -> dict:
    """The primary verdict uses the cluster risk run; the comparisons use the per-rule Kaggle checks."""
    primary = json.load(open(risk_test))["at_threshold"]
    res = {"primary_cluster_run": {n: {"harmful_rate": _boot(primary[n]["harmful_rate"]),
                                       "accepted": primary[n]["accepted_inferences"],
                                       "harmful": primary[n]["harmful_inferences"]} for n in primary}}
    res["verdict"] = h2_verdict(res["primary_cluster_run"]["B7"]["harmful_rate"], ALPHA)

    # paired comparisons on the Kaggle re-run (per-rule checks; calibration and test in one environment)
    import pandas as pd
    from trialguide.baselines.run_risk import EXPERT
    from trialguide.policy.calibrate import read_checks
    from trialguide.policy.risk_control import counts_at, risk_arrays
    from trialguide.schemas import RuleStatus
    cal, test = read_checks(kaggle_dir / "checks_cal.jsonl"), read_checks(kaggle_dir / "checks_test.jsonl")
    ann = pd.read_parquet(raw / "annotations" / "criterion_annotations.parquet")
    ann = ann[ann.patient_id.isin(set(cal) | set(test))]
    ref = {(r.patient_id, r.trial_id, f"ann_{r.annotation_id}"): RuleStatus(EXPERT[r.expert_eligibility])
           for r in ann.itertuples()}
    lam_unc = uncorrected_threshold(cal, ref)
    frozen = res["primary_cluster_run"]["B7"] and float(primary["B7"]["threshold"])
    thresholds = {"B7": frozen, "B5": 0.9, "B2": 0.0, "uncorrected_tuned": lam_unc}
    arr = {n: risk_arrays(counts_at(test, ref, lam)) for n, lam in thresholds.items()}
    flat = {f"{n}_{k}": v for n, a in arr.items() for k, v in a.items()}
    mean = lambda key: (lambda d: d[key].mean(axis=-1))
    res["kaggle_rerun"] = {
        "thresholds": {n: (lam if np.isfinite(lam) else None) for n, lam in thresholds.items()},
        "harmful_rate": {n: paired_bootstrap(flat, mean(f"{n}_loss"), n_boot) for n in thresholds},
        "B7_minus": {n: paired_bootstrap(flat, lambda d, n=n: d["B7_loss"].mean(axis=-1) - d[f"{n}_loss"].mean(axis=-1), n_boot)
                     for n in ("B5", "B2", "uncorrected_tuned")}}
    return res


def mcnemar_pairs(run: Path) -> dict:
    ref = reference(run / "checks_full.jsonl")
    b6 = policy_finals(run / "checks_rule40.jsonl", "B6")
    out = {name: rule_outcomes(ref, b6, asking_finals(run / f"final_{name}.json"))
           for name in ("B7", "B4", "A1", "A2", "A3", "A5", "A6", "B3")}
    out["B6"] = rule_outcomes(ref, b6, b6)
    keys = sorted(out["B7"])
    res = {"lost_rules": len(keys)}
    for other in ("B6", "B4", "A1", "A6", "B3"):
        r = mcnemar([out["B7"][k] for k in keys], [out[other][k] for k in keys])
        res[f"B7_vs_{other}"] = {"B7_only_right": r.a_only, f"{other}_only_right": r.b_only, "p_value": r.p_value}
    return res


def _plain(o):
    if isinstance(o, BootResult):
        return {"estimate": o.estimate, "low": o.low, "high": o.high,
                "resamples": o.n_boot, "dropped_resamples": o.n_undefined}
    raise TypeError(type(o))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, default=Path("results/test_run"))
    ap.add_argument("--kaggle", type=Path, default=Path("results/resplits"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = {"H1": h1(args.run), "H2": h2(args.run / "risk_test.json", args.kaggle, args.raw),
           "McNemar": mcnemar_pairs(args.run)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verdicts.json").write_text(json.dumps(out, indent=1, default=_plain) + "\n")
    print(json.dumps(out, indent=1, default=_plain))


if __name__ == "__main__":
    main()
