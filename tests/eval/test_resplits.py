"""T36: exploratory resplits, on a tiny synthetic pool."""

import numpy as np
import pytest

from trialguide.eval import resplits
from trialguide.schemas import InfoCategory, RuleCheck, RuleKind, RuleStatus


def _check(pid, rid, status, conf, kind=RuleKind.INCLUSION):
    return RuleCheck(patient_id=pid, trial_id="T1", rule_id=rid, kind=kind, status=status,
                     category=InfoCategory.INFERRED, evidence_ids=[0], confidence=conf, n_samples=1)


def _pool(n=12, wrong_every=4):
    """n patients, each with 6 rules the checker infers MET; every `wrong_every`-th is wrong (reference NOT_MET)."""
    checks, ref = {}, {}
    for p in range(n):
        pid = f"p{p:02d}"
        checks[pid] = []
        for r in range(6):
            wrong = (p * 6 + r) % wrong_every == 0
            checks[pid].append(_check(pid, f"r{r}", RuleStatus.MET, 0.5 + 0.08 * r))
            ref[(pid, "T1", f"r{r}")] = RuleStatus.NOT_MET if wrong else RuleStatus.MET
    return checks, ref


def test_resplit_rows_and_halves():
    checks, ref = _pool()
    rows = resplits.resplit(checks, ref, n_resplits=20, seed=1)
    assert len(rows) == 20 and all(r["n_cal"] == 6 and r["n_test"] == 6 for r in rows)
    for r in rows:
        assert set(r) >= {"lambda_hat", "accepts_any", "B7", "B5", "B2"}
        assert r["B2"]["accepted"] >= r["B5"]["accepted"]          # lower threshold accepts more


def test_resplit_is_seeded_and_varies():
    checks, ref = _pool()
    a = resplits.resplit(checks, ref, 30, seed=3)
    assert a == resplits.resplit(checks, ref, 30, seed=3)
    assert a != resplits.resplit(checks, ref, 30, seed=4)
    assert len({r["B2"]["harmful_rate"] for r in a}) > 1            # different halves, different rates


def test_accept_all_on_the_same_test_half_is_the_harmful_ceiling():
    checks, ref = _pool()
    for r in resplits.resplit(checks, ref, 15, seed=2):
        assert r["B7"]["harmful"] <= r["B2"]["harmful"]


def test_summary_fields():
    checks, ref = _pool()
    s = resplits.summarise(resplits.resplit(checks, ref, 25, seed=5))
    assert s["n_resplits"] == 25 and 0.0 <= s["share_accepting_nothing"] <= 1.0
    for name in ("B7", "B5", "B2"):
        assert s[name]["p2.5"] <= s[name]["median"] <= s[name]["p97.5"]
        assert 0.0 <= s[name]["share_at_or_below_alpha"] <= 1.0
