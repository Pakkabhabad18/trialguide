"""Planned analyses that were missing, computed from saved outputs only (no model is called).

    python -m trialguide.eval.post_hoc --out results/post_hoc

Run after the test run, from files already in results/, so everything here is post hoc. Four parts:

1. Question budget sweep (proposal: budgets 0, 1, 2, 3, 5). Budgets 0 to 3 are recovered exactly from the
   saved test run, without calling a model: the question chosen at turn k does not depend on the budget
   (only the stopping does), and an answer re-checks only the rules of its own question. So with budget k,
   a rule keeps its final status from the budget-3 run if it belongs to one of the first k questions, and
   otherwise has the status the policy gives it before any question. The script checks this on every
   patient: at k = 3 it must give back the saved final statuses exactly. Budget 5 needs new model calls.
2. Trial-level decisions (proposal metrics table): a trial is "ruled out" when a settled rule excludes the
   patient (an inclusion rule not met, or an exclusion rule met), i.e. its score is below 0. Against the
   expert trial labels (2 eligible, 1 excluded, 0 not relevant): false-exclusion rate = eligible trials
   ruled out; false-eligible rate = ineligible trials (label 0 or 1) left in; precision and recall of
   "left in" for the eligible trials.
3. Risk control at alpha = 2%, 5% and 10%, the risk-coverage curve on test, and the checker's calibration
   error, all on the Kaggle re-run of the calibration and test checks (results/resplits/: same frozen code,
   one environment for both splits; the cluster's per-rule test checks were not saved). The cluster's
   lambda-hat for each alpha is also read off its calibration curve (results/dev_run/calibration.json).
4. A plot of the risk-coverage curve (risk_coverage.png).

Owner: Prakash.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from trialguide.eval.stats import mean_of, paired_bootstrap, ratio_of_sums

LAMBDA_HAT = 0.7660011405599875       # frozen cluster lambda-hat (results/dev_run/calibration.json)
DECIDED = ("met", "not_met")
N_BOOT = 10000


# ---------------------------------------------------------------------------
# Scoring, as in selector.trial_score, on (status, kind) pairs
# ---------------------------------------------------------------------------

def trial_score(rules) -> float:
    """TrialGPT's linear score: share of inclusion rules met (not-applicable left out), minus 1 if an
    inclusion rule is not met, minus 1 if an exclusion rule is met. `rules` = [(status, kind), ...]."""
    met = not_met = no_info = 0
    excluded = False
    for status, kind in rules:
        if kind == "inclusion":
            met += status == "met"
            not_met += status == "not_met"
            no_info += status == "not_enough_info"
        elif status == "met":
            excluded = True
    score = met / (met + not_met + no_info) if met + not_met + no_info else 0.0
    return score - (1 if not_met else 0) - (1 if excluded else 0)


def scores_of(final: dict, kinds: dict) -> dict:
    """{'trial|rule': status} of one patient -> {trial: score}."""
    by_trial = {}
    for key, status in final.items():
        by_trial.setdefault(key.split("|")[0], []).append((status, kinds[key]))
    return {t: trial_score(r) for t, r in by_trial.items()}


def gold_of(manifest: Path) -> dict:
    """patient -> {trial: expert grade}, from the run's manifest."""
    m = json.loads(manifest.read_text())
    return {pid: {t: int(lab) for lab, ts in p["trials"].items() for t in ts} for pid, p in m["patients"].items()}


def ndcg(scores: dict, grades: dict, k: int = 10):
    from trialguide.eval.ranking import ndcg_at_k, ranked_gains
    return ndcg_at_k(ranked_gains({t: scores.get(t, 0.0) for t in grades}, grades), k)


# ---------------------------------------------------------------------------
# Loading the saved run
# ---------------------------------------------------------------------------

def read_checks(path: Path) -> dict:
    """patient -> list of check dicts (and the row's saved actions)."""
    out = {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        out[row["patient_id"]] = row
    return out


def non_asking_finals(rows: dict, policy: str) -> dict:
    """Effective statuses of a policy that never asks (B6, B2, B5), from the saved actions."""
    from trialguide.pipeline import ACCEPTED
    out = {}
    for pid, row in rows.items():
        ok = {f"{a['trial_id']}|{a['rule_id']}" for a in row["actions"][policy] if a["action"] in ACCEPTED}
        out[pid] = {f"{c['trial_id']}|{c['rule_id']}": (c["status"] if f"{c['trial_id']}|{c['rule_id']}" in ok
                                                         else "not_enough_info") for c in row["checks"]}
    return out


def before_questions(rows: dict, system: str) -> dict:
    """Effective statuses of an asking system before its first question: the policy on the initial checks,
    with every unasked candidate turned into an abstention (policy.finalize)."""
    from trialguide.pipeline import effective_checks
    from trialguide.policy.calibrate import check_from_json
    from trialguide.policy.policy import decide_all, finalize, system_config
    name = {"B4": "B7", "B7-eq1": "B7"}.get(system, system)
    config = system_config(name, threshold=None if name in ("A1", "A6") else LAMBDA_HAT)
    out = {}
    for pid, row in rows.items():
        checks = [check_from_json(c) for c in row["checks"]]
        eff = effective_checks(checks, finalize(decide_all(checks, config)))
        out[pid] = {f"{c.trial_id}|{c.rule_id}": c.status.value for c in eff}
    return out


def turn_refs(turns_file: Path, system: str) -> dict:
    """patient -> [set of 'trial|rule' each question could settle], in turn order."""
    out = {}
    for line in open(turns_file, encoding="utf-8"):
        row = json.loads(line)
        if row["system"] == system:
            out[row["patient_id"]] = [{f"{t}|{r}" for t, r in turn["question"]["rule_refs"]} for turn in row["turns"]]
    return out


# ---------------------------------------------------------------------------
# 1. Budget sweep
# ---------------------------------------------------------------------------

def at_budget(final: dict, start: dict, refs: list, k: int) -> dict:
    asked = set().union(*refs[:k]) if refs[:k] else set()
    return {key: (final[key] if key in asked else start[key]) for key in final}


def budget_sweep(run: Path, systems=("B7", "B4", "A1", "A6"), budgets=(0, 1, 2, 3)) -> dict:
    rows = read_checks(run / "checks_rule40.jsonl")
    kinds = {pid: {f"{c['trial_id']}|{c['rule_id']}": c["kind"] for c in row["checks"]} for pid, row in rows.items()}
    gold = gold_of(run / "manifest.json")
    saved = json.loads((run / "asking.json").read_text())["systems"]
    b6 = non_asking_finals(rows, "B6")
    full = non_asking_finals(read_checks(run / "checks_full.jsonl"), "B6")      # B1: the full note
    base = {pid: ndcg(scores_of(b6[pid], kinds[pid]), gold[pid]) for pid in rows}
    oracle = {pid: ndcg(scores_of(full[pid], kinds[pid]), gold[pid]) for pid in rows}
    pids = sorted(p for p in rows if base[p] is not None)
    out = {"patients": len(pids), "B6": float(np.mean([base[p] for p in pids])),
           "B1": float(np.mean([oracle[p] for p in pids])), "systems": {}}
    for system in systems:
        final = {pid: {k: v[0] for k, v in r.items()} for pid, r in
                 json.loads((run / f"final_{system}.json").read_text()).items()}
        start = before_questions(rows, system)
        refs = turn_refs(run / "turns_asking.jsonl", system)
        # exactness check: budget 3 must give back every saved final status
        mismatch = sum(at_budget(final[p], start[p], refs.get(p, []), 3)[k] != final[p][k]
                       for p in final for k in final[p])
        res = {"exact_at_budget_3": mismatch == 0, "mismatched_rules_at_budget_3": int(mismatch), "budgets": {}}
        for k in budgets:
            per = {p: ndcg(scores_of(at_budget(final[p], start[p], refs.get(p, []), k), kinds[p]), gold[p])
                   for p in pids}
            arrays = {"sys": [per[p] for p in pids], "b6": [base[p] for p in pids], "b1": [oracle[p] for p in pids]}
            diff = paired_bootstrap(arrays, lambda d: d["sys"].mean(-1) - d["b6"].mean(-1), N_BOOT, 0)
            gap = arrays["b1"]
            rec = (np.mean(arrays["sys"]) - np.mean(arrays["b6"])) / (np.mean(gap) - np.mean(arrays["b6"]))
            res["budgets"][str(k)] = {
                "ndcg": float(np.mean(arrays["sys"])),
                "minus_B6": {"estimate": diff.estimate, "low": diff.low, "high": diff.high},
                "recovery": float(rec),
                "questions_per_patient": float(np.mean([min(k, len(refs.get(p, []))) for p in pids])),
            }
        res["saved_ndcg_budget_3"] = saved[system]["NDCG@10"] if system in saved else None
        out["systems"][system] = res
    return out


# ---------------------------------------------------------------------------
# 2. Trial-level decisions
# ---------------------------------------------------------------------------

def trial_level(run: Path) -> dict:
    rows = read_checks(run / "checks_rule40.jsonl")
    kinds = {pid: {f"{c['trial_id']}|{c['rule_id']}": c["kind"] for c in row["checks"]} for pid, row in rows.items()}
    gold = gold_of(run / "manifest.json")
    finals = {"B1": non_asking_finals(read_checks(run / "checks_full.jsonl"), "B6")}
    for p in ("B6", "B2", "B5"):
        finals[p] = non_asking_finals(rows, p)
    for f in sorted(run.glob("final_*.json")):
        finals[f.stem[len("final_"):]] = {pid: {k: v[0] for k, v in r.items()}
                                          for pid, r in json.loads(f.read_text()).items()}
    out = {"definition": "ruled out = score < 0 (a settled rule excludes the patient); ineligible = expert "
                         "label 0 or 1; eligible = label 2", "systems": {}}
    pids = sorted(rows)
    for name, final in finals.items():
        a = {k: [] for k in ("elig", "elig_out", "inelig", "inelig_in", "left_in", "left_in_elig")}
        for p in pids:
            s = scores_of(final[p], kinds[p])
            g = gold[p]
            counts = dict.fromkeys(a, 0)
            for t, grade in g.items():
                out_ = s.get(t, 0.0) < 0
                if grade == 2:
                    counts["elig"] += 1
                    counts["elig_out"] += out_
                else:
                    counts["inelig"] += 1
                    counts["inelig_in"] += not out_
                if not out_:
                    counts["left_in"] += 1
                    counts["left_in_elig"] += grade == 2
            for k in a:
                a[k].append(counts[k])
        def ci(num, den):
            r = paired_bootstrap(a, ratio_of_sums(num, den), N_BOOT, 0)
            return {"estimate": r.estimate, "low": r.low, "high": r.high}
        out["systems"][name] = {
            "pairs": int(sum(a["elig"]) + sum(a["inelig"])), "eligible_pairs": int(sum(a["elig"])),
            "false_exclusion_rate": ci("elig_out", "elig"),
            "false_eligible_rate": ci("inelig_in", "inelig"),
            "precision_left_in": ci("left_in_elig", "left_in"),
            "recall_eligible": {"estimate": 1 - ci("elig_out", "elig")["estimate"]},
            "left_in_share": float(sum(a["left_in"]) / (sum(a["elig"]) + sum(a["inelig"]))),
        }
    return out


# ---------------------------------------------------------------------------
# 3. Alpha sweep, risk-coverage curve and calibration error (Kaggle re-run of the risk checks)
# ---------------------------------------------------------------------------

EXPERT = {"included": "met", "not included": "not_met", "excluded": "met", "not excluded": "not_met",
          "not enough information": "not_enough_info", "not applicable": "not_applicable"}


def expert_reference(raw: Path) -> dict:
    """(patient, trial, 'ann_<id>') -> RuleStatus, as baselines/run_risk.py builds it."""
    import pandas as pd
    from trialguide.schemas import RuleStatus
    ann = pd.read_parquet(raw / "annotations" / "criterion_annotations.parquet")
    return {(r.patient_id, r.trial_id, f"ann_{r.annotation_id}"): RuleStatus(EXPERT[r.expert_eligibility])
            for r in ann.itertuples()}


def cluster_lambdas(calibration_json: Path, alphas) -> dict:
    """lambda-hat for each alpha from the cluster's saved calibration curve (lambda, R_hat, bound)."""
    curve = json.loads(calibration_json.read_text())["curve"]
    n = json.loads(calibration_json.read_text())["patients"]
    out = {}
    for alpha in alphas:
        lam = next((c[0] for c in curve if c[2] <= alpha), "inf")
        out[f"{alpha:.2f}"] = {"lambda_hat": lam, "certifiable": 1 / (n + 1) <= alpha}
    return out


def ece_of(conf, ok, bins: int = 10):
    """Expected calibration error with equal-width bins, and the per-bin table."""
    conf, ok = np.asarray(conf, dtype=float), np.asarray(ok, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - ok[m].mean())
            table.append({"bin": [round(lo, 1), round(hi, 1)], "rules": int(m.sum()),
                          "mean_confidence": float(conf[m].mean()), "agreement": float(ok[m].mean())})
    return float(ece), table


def calibration_error(checks: dict, reference: dict, bins: int = 10) -> dict:
    """Expected calibration error of the checker's confidence (probability of the status it chose) for
    'the status is the expert's', over every expert-labelled rule and over the rules it decided (met or
    not met); plus agreement with the experts."""
    conf, ok, dec, decided_ok, decided = [], [], [], 0, 0
    for cs in checks.values():
        for c in cs:
            ref = reference.get((c.patient_id, c.trial_id, c.rule_id))
            if ref is None:
                continue
            conf.append(c.confidence)
            ok.append(c.status == ref)
            dec.append(c.status.value in DECIDED)
            if c.status.value in DECIDED:
                decided += 1
                decided_ok += c.status == ref
    ece, table = ece_of(conf, ok, bins)
    dconf = [c for c, d in zip(conf, dec) if d]
    dok = [o for o, d in zip(ok, dec) if d]
    ece_decided, table_decided = ece_of(dconf, dok, bins)
    conf, ok = np.asarray(conf), np.asarray(ok, dtype=float)
    return {"rules": int(len(ok)), "agreement": float(ok.mean()), "mean_confidence": float(conf.mean()),
            "ece": float(ece), "decided_rules": decided,
            "accuracy_of_decided": decided_ok / decided if decided else None,
            "ece_decided": ece_decided, "bins": table, "bins_decided": table_decided}


def risk_sweep(resplits: Path, raw: Path, alphas=(0.02, 0.05, 0.10)) -> dict:
    from trialguide.policy.calibrate import read_checks as read_rule_checks
    from trialguide.policy.risk_control import calibrate, counts_at, is_permissive_inference, risk_report
    reference = expert_reference(raw)
    cal = read_rule_checks(resplits / "checks_cal.jsonl")
    test = read_rule_checks(resplits / "checks_test.jsonl")
    out = {"source": "Kaggle re-run of the calibration and test checks (results/resplits/), same frozen code",
           "alphas": {}, "curve_test": [], "calibration_error": {}}
    for alpha in alphas:
        c = calibrate(cal, reference, alpha)
        lam = c.lambda_hat
        rep = risk_report(counts_at(test, reference, lam), n_boot=N_BOOT)
        out["alphas"][f"{alpha:.2f}"] = {
            "lambda_hat": lam if math.isfinite(lam) else "inf", "certifiable": c.certifiable,
            "calibration_risk": c.risk_at_lambda,
            "test": {k: (v if isinstance(v, int) else {"estimate": v.estimate, "low": v.low, "high": v.high})
                     for k, v in rep.items()},
        }
    lams = sorted({x.confidence for cs in test.values() for x in cs
                   if is_permissive_inference(x) and reference.get((x.patient_id, x.trial_id, x.rule_id)) is not None})
    for lam in [0.0] + lams + [math.inf]:
        counts = counts_at(test, reference, lam)
        acc, uns = sum(x.accepted for x in counts), sum(x.unsettled for x in counts)
        out["curve_test"].append({"lambda": lam if math.isfinite(lam) else "inf",
                                  "harmful_rate": float(np.mean([x.loss for x in counts])),
                                  "inference_coverage": acc / uns if uns else 0.0,
                                  "useful_coverage": sum(x.accepted_correct for x in counts) / uns if uns else 0.0})
    out["calibration_error"] = {"calibration": calibration_error(cal, reference),
                                "test": calibration_error(test, reference)}
    return out


def plot_curve(sweep: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = sorted(sweep["curve_test"], key=lambda p: p["inference_coverage"])
    x = [p["inference_coverage"] * 100 for p in pts]
    y = [p["harmful_rate"] * 100 for p in pts]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    ax.plot(x, y, color="#0f766e", lw=2, marker="o", ms=2.5, label="every threshold on test")
    groups = {}
    for alpha, a in sweep["alphas"].items():
        groups.setdefault(str(a["lambda_hat"]), []).append((alpha, a))
    for lam, items in groups.items():
        a = items[0][1]
        cov, risk = a["test"]["inference_coverage"]["estimate"], a["test"]["harmful_rate"]["estimate"]
        names = " and ".join(f"{float(al):.0%}" for al, _ in items)
        shown = "none accepted" if lam == "inf" else f"{float(lam):.3f}"
        ax.scatter([cov * 100], [risk * 100], zorder=3, s=46, label=f"alpha {names}: lambda-hat {shown}")
    ax.axhline(5, color="#dc2626", ls="--", lw=1, label="alpha = 5% target")
    ax.set_xlabel("Inference coverage on test (%)")
    ax.set_ylabel("Harmful-inference rate on test (%)")
    ax.set_title("Risk vs coverage, 26 expert-labelled test patients\n(Kaggle re-run of the frozen checker)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_ylim(-0.3, 6)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, default=Path("results/test_run"))
    ap.add_argument("--resplits", type=Path, default=Path("results/resplits"))
    ap.add_argument("--calibration", type=Path, default=Path("results/dev_run/calibration.json"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("results/post_hoc"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    sweep = budget_sweep(args.run)
    (args.out / "budget_sweep.json").write_text(json.dumps(sweep, indent=1) + "\n")
    for s, r in sweep["systems"].items():
        print(s, "exact at 3:", r["exact_at_budget_3"], {k: round(v["ndcg"], 4) for k, v in r["budgets"].items()},
              "saved:", r["saved_ndcg_budget_3"])

    trials = trial_level(args.run)
    (args.out / "trial_level.json").write_text(json.dumps(trials, indent=1) + "\n")
    for s, r in trials["systems"].items():
        print(f"{s:7} false-excl {r['false_exclusion_rate']['estimate']:.3f}  false-elig "
              f"{r['false_eligible_rate']['estimate']:.3f}  precision {r['precision_left_in']['estimate']:.3f}")

    risk = risk_sweep(args.resplits, args.raw)
    risk["cluster_lambda_hat"] = cluster_lambdas(args.calibration, (0.02, 0.05, 0.10))
    (args.out / "risk_sweep.json").write_text(json.dumps(risk, indent=1) + "\n")
    plot_curve(risk, args.out / "risk_coverage.png")
    for a, r in risk["alphas"].items():
        print("alpha", a, "lambda", r["lambda_hat"], "test harmful", r["test"]["harmful_rate"])
    print("ECE", {k: (round(v["ece"], 3), round(v["ece_decided"], 3)) for k, v in risk["calibration_error"].items()})


if __name__ == "__main__":
    main()
