"""E1 evaluation (RQ3, D14): the QLoRA-tuned Qwen3-8B against the untuned Qwen3-8B as the rule checker.

Both models are run the same way, with 4-bit weights, tools off, one greedy call and the same rule-masked
notes, on the expert-labelled rules of the calibration and test patients (none of them was used for
training). The runs are the normal risk runs, one folder per model:

    # GPU (Kaggle: one T4 per model). Untuned:
    LLM_HF_4BIT=1 LLM_CACHE=outputs/e1/cache_untuned.jsonl \
        python -m trialguide.baselines.run_risk --split cal --no-tools --out outputs/e1/untuned
    # tuned: the same, plus the adapter
    LLM_HF_4BIT=1 LLM_HF_ADAPTER=outputs/e1/qlora_r16/adapter LLM_CACHE=outputs/e1/cache_tuned.jsonl \
        python -m trialguide.baselines.run_risk --split cal --no-tools --out outputs/e1/tuned
    # then --split test --allow-test --thresholds B2=0 B7=<that model's own lambda-hat>

    # CPU: compare
    python -m trialguide.finetune.compare --untuned outputs/e1/untuned --tuned outputs/e1/tuned

Each model gets its own lambda-hat from its own calibration run: lambda-hat is only valid for the
checker it was fitted on.

What is reported, per split, against the expert labels (never against the teacher):
- agreement: the checker's status equals the expert's (met / not_met / not_enough_info / n/a);
- decided: share of rules the checker settled (met or not_met), and how often those are right;
- unsafe: share of rules the checker settled WRONGLY toward eligibility, whatever its category
  (stated answers bypass risk control, so this is the risk the threshold cannot remove);
- hidden evidence: rules the experts settled from sentences that are all hidden by the mask. The
  right answer there is "not enough information"; we report how often each model guesses instead;
- risk control: lambda-hat, harmful-inference rate and coverage at lambda-hat (from run_risk).
Differences (tuned minus untuned) come with patient-level paired bootstrap CIs and an exact
McNemar test on rule-level agreement. Writes results/e1/e1_compare.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DECIDED = ("met", "not_met")
SPLITS = ("cal", "test")


def favours_eligibility(kind: str, status: str) -> bool:
    return (kind == "inclusion" and status == "met") or (kind == "exclusion" and status == "not_met")


def read_checks(path: Path) -> dict:
    """checks_{split}.jsonl -> {(patient, trial, rule): check dict}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for c in row["checks"]:
                out[(row["patient_id"], c["trial_id"], c["rule_id"])] = c
    return out


def expert_reference(raw: Path, patients) -> tuple[dict, dict]:
    """Expert status and expert evidence ids per (patient, trial, rule), with the rule ids run_risk uses."""
    import pandas as pd
    from trialguide.baselines.run_risk import EXPERT
    ann = pd.read_parquet(raw / "annotations" / "criterion_annotations.parquet")
    ann = ann[ann.patient_id.isin(set(patients))]
    status, evidence = {}, {}
    for r in ann.itertuples():
        key = (r.patient_id, r.trial_id, f"ann_{r.annotation_id}")
        status[key] = EXPERT[r.expert_eligibility]
        ids = json.loads(r.expert_sentences) if isinstance(r.expert_sentences, str) else []
        evidence[key] = set(ids)
    return status, evidence


def hidden_sentences(raw: Path, split: str, fraction: float, patient_ids) -> dict:
    """The sentence ids the rule mask hid for each patient, recomputed exactly as run_risk masks.
    Scoring only: never given to a checker."""
    from trialguide.data.masking import candidate_rules, mask_rule, rule_demand
    from trialguide.data.patients import load_patients
    from trialguide.schemas import Split
    pts = load_patients(raw, Split(split), allow_test=(split == "test"))
    out = {}
    for pid in patient_ids:
        pt = pts[pid]
        out[pid] = set(mask_rule(pt, rule_demand(pt.sentences, candidate_rules(raw, pt.cohort, pid)),
                                 fraction).hidden_ids)
    return out


def rule_outcomes(checks: dict, status: dict, evidence: dict, hidden: dict) -> dict:
    """One row per expert-labelled rule: the yes/no outcomes every metric is built from."""
    rows = {}
    for key, ref in status.items():
        c = checks.get(key)
        if c is None:
            continue
        pid, _, _ = key
        decided = c["status"] in DECIDED
        ev = evidence.get(key, set())
        rows[key] = {
            "agree": c["status"] == ref,
            "decided": decided,
            "decided_right": decided and c["status"] == ref,
            "unsafe": decided and c["status"] != ref and favours_eligibility(c["kind"], c["status"]),
            # the experts settled it from sentences that are all hidden: only asking can recover it
            "evidence_hidden": ref in DECIDED and bool(ev) and ev <= hidden[pid],
        }
    return rows


def summary(rows: dict) -> dict:
    n = len(rows)
    decided = sum(r["decided"] for r in rows.values())
    hid = [r for r in rows.values() if r["evidence_hidden"]]
    return {"rules": n,
            "agreement": sum(r["agree"] for r in rows.values()) / n if n else None,
            "decided_rate": decided / n if n else None,
            "decided_accuracy": sum(r["decided_right"] for r in rows.values()) / decided if decided else None,
            "unsafe_rate": sum(r["unsafe"] for r in rows.values()) / n if n else None,
            "evidence_hidden_rules": len(hid),
            "guess_rate_when_evidence_hidden": sum(r["decided"] for r in hid) / len(hid) if hid else None,
            "unsafe_rate_when_evidence_hidden": sum(r["unsafe"] for r in hid) / len(hid) if hid else None}


def paired(untuned: dict, tuned: dict) -> dict:
    """Tuned minus untuned on the rules both models answered: bootstrap CIs over patients, McNemar."""
    from trialguide.eval.stats import mcnemar, paired_bootstrap, ratio_of_sums
    keys = sorted(set(untuned) & set(tuned))
    pids = sorted({k[0] for k in keys})
    out = {}
    for metric in ("agree", "unsafe", "decided"):
        arrays = {"n": [], "u": [], "t": []}
        for p in pids:
            ks = [k for k in keys if k[0] == p]
            arrays["n"].append(len(ks))
            arrays["u"].append(sum(untuned[k][metric] for k in ks))
            arrays["t"].append(sum(tuned[k][metric] for k in ks))

        def diff(d):
            return ratio_of_sums("t", "n")(d) - ratio_of_sums("u", "n")(d)
        b = paired_bootstrap(arrays, diff)
        out[metric] = {"tuned_minus_untuned": b.estimate, "ci95": [b.low, b.high]}
    m = mcnemar([tuned[k]["agree"] for k in keys], [untuned[k]["agree"] for k in keys])
    out["mcnemar_agreement"] = {"tuned_only_right": m.a_only, "untuned_only_right": m.b_only,
                                "p_value": m.p_value}
    return out


def risk_numbers(folder: Path) -> dict:
    """lambda-hat from the calibration run and the test risk at each threshold run_risk reported."""
    out = {}
    cal = folder / "calibration.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        out.update({"lambda_hat": c.get("lambda_hat"), "certifiable": c.get("certifiable"),
                    "cal_risk_at_lambda": c.get("risk_at_lambda")})
    test = folder / "risk_test.json"
    if test.exists():
        out["test_at_threshold"] = json.loads(test.read_text()).get("at_threshold")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--untuned", type=Path, required=True)
    ap.add_argument("--tuned", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--fraction", type=float, default=0.4, help="the mask fraction run_risk used")
    ap.add_argument("--out", type=Path, default=Path("results/e1/e1_compare.json"))
    args = ap.parse_args(argv)

    result = {"untuned": str(args.untuned), "tuned": str(args.tuned), "splits": {}}
    for split in SPLITS:
        files = [args.untuned / f"checks_{split}.jsonl", args.tuned / f"checks_{split}.jsonl"]
        if not all(f.exists() for f in files):
            print(f"{split}: checks missing for one of the models, skipped")
            continue
        u, t = (read_checks(f) for f in files)
        pids = sorted({k[0] for k in u} | {k[0] for k in t})
        status, evidence = expert_reference(args.raw, pids)
        hidden = hidden_sentences(args.raw, split, args.fraction, pids)
        ru, rt = (rule_outcomes(c, status, evidence, hidden) for c in (u, t))
        result["splits"][split] = {"patients": len(pids), "untuned": summary(ru), "tuned": summary(rt),
                                   "difference": paired(ru, rt)}
    result["risk_control"] = {"untuned": risk_numbers(args.untuned), "tuned": risk_numbers(args.tuned)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for split, s in result["splits"].items():
        print(f"\n{split} ({s['patients']} patients, {s['untuned']['rules']} rules)")
        for k in ("agreement", "decided_rate", "decided_accuracy", "unsafe_rate",
                  "guess_rate_when_evidence_hidden", "unsafe_rate_when_evidence_hidden"):
            fmt = lambda v: "  n/a" if v is None else f"{v:6.1%}"
            print(f"  {k:34s} untuned {fmt(s['untuned'][k])}   tuned {fmt(s['tuned'][k])}")
    print("\nwritten to", args.out)


if __name__ == "__main__":
    main()
