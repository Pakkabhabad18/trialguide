"""One call for the headline ranking numbers (T17): NDCG@10, P@10 and recovery with CIs.

    rep = ranking_report({"B1": r1, "B6": r6, "B7": r7}, gold, oracle="B1", floor="B6")

Every system's rankings are {patient_id: {trial_id: score}} over the SAME patients and trials.
All intervals are paired patient-level bootstrap 95% CIs (the same resampled patients for every
system and for the recovery ratio).
"""

from trialguide.eval.ranking import per_patient_metrics, same_candidates
from trialguide.eval.stats import MIN_GAP, mean_of, paired_bootstrap, recovery_stat


def ranking_report(rankings_by_system: dict, gold: dict, oracle: str = "B1", floor: str = "B6",
                   n_boot: int = 10000, seed: int = 0) -> dict:
    same_candidates(rankings_by_system)
    metrics = {s: per_patient_metrics(r, gold) for s, r in rankings_by_system.items()}
    # Patients without a defined NDCG depend on gold only, so every system has the same set
    patients = sorted(metrics[oracle])
    arrays = {s: {"ndcg": [m[p]["ndcg"] for p in patients], "p10": [m[p]["p10"] for p in patients]}
              for s, m in metrics.items()}
    ndcg = {s: a["ndcg"] for s, a in arrays.items()}
    out = {"n_patients": len(patients), "systems": {}}
    gap = sum(ndcg[oracle]) / len(patients) - sum(ndcg[floor]) / len(patients)
    out["oracle_floor_gap"] = gap
    for s, a in arrays.items():
        row = {"ndcg@10": paired_bootstrap({"x": a["ndcg"]}, mean_of("x"), n_boot, seed),
               "p@10": paired_bootstrap({"x": a["p10"]}, mean_of("x"), n_boot, seed)}
        if s not in (oracle, floor) and abs(gap) >= MIN_GAP:
            row["recovery"] = paired_bootstrap(ndcg, recovery_stat(s, oracle, floor), n_boot, seed)
        out["systems"][s] = row
    return out
