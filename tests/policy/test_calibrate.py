"""The calibration runner's file formats and the fit step (T30), without a model."""

import json
import math

from trialguide.baselines.run_systems import to_json
from trialguide.policy.calibrate import check_from_json, fit, main, read_checks, reference_to_json
from trialguide.schemas import InfoCategory, RuleCheck, RuleKind, RuleStatus


def chk(pid, rule, status, conf, category=InfoCategory.INFERRED):
    ev = [0] if category in (InfoCategory.STATED, InfoCategory.INFERRED) else []
    return RuleCheck(patient_id=pid, trial_id="t1", rule_id=rule, kind=RuleKind.INCLUSION,
                     status=status, category=category, evidence_ids=ev, confidence=conf)


def test_check_round_trips_through_json():
    c = chk("p", "inc_0", RuleStatus.MET, 0.73)
    c.fact_key = "age"
    assert check_from_json(json.loads(json.dumps(to_json(c)))) == c


def write_population(tmp_path, n_harmful, n=20):
    # same population as test_risk_control: harmful at 0.7 for the first n_harmful, correct at 0.8
    ref = {}
    with open(tmp_path / "checks_rule40.jsonl", "w") as f:
        for i in range(n):
            pid = f"p{i}"
            cs = [chk(pid, "ok", RuleStatus.MET, 0.8)]
            ref[(pid, "t1", "ok")] = RuleStatus.MET
            if i < n_harmful:
                cs.append(chk(pid, "bad", RuleStatus.MET, 0.7))
                ref[(pid, "t1", "bad")] = RuleStatus.NOT_MET
            for j in range(10 - len(cs)):
                cs.append(chk(pid, f"m{j}", RuleStatus.NOT_ENOUGH_INFO, 1.0, InfoCategory.MISSING))
                ref[(pid, "t1", f"m{j}")] = RuleStatus.NOT_ENOUGH_INFO
            f.write(json.dumps({"patient_id": pid, "checks": to_json(cs)}) + "\n")
    (tmp_path / "expert_reference.json").write_text(json.dumps({"reference": reference_to_json(ref)}))
    return tmp_path / "checks_rule40.jsonl"


def test_fit_writes_lambda_hat_per_alpha(tmp_path):
    path = write_population(tmp_path, n_harmful=15)
    assert len(read_checks(path)) == 20
    main(["fit", "--checks", str(path), "--alphas", "0.04", "0.10", "--n-boot", "200",
          "--results", str(tmp_path / "results")])
    out = json.loads((tmp_path / "results" / "calibration_rule40.json").read_text())
    assert out["n_patients"] == 20 and out["rules"] == 200
    assert out["alphas"]["0.10"]["lambda_hat"] == 0.8               # as in test_risk_control
    assert out["alphas"]["0.04"]["accept_none"] and not out["alphas"]["0.04"]["certifiable"]
    assert out["alphas"]["0.04"]["lambda_hat"] is None
