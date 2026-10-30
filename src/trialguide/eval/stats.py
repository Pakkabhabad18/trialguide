"""Statistics (T17): paired patient-level bootstrap, recovery, McNemar and the H1 / H2 verdicts.

Bootstrap convention: the unit is the PATIENT. Each statistic is a function of per-patient arrays
and is evaluated on `n_boot` resamples at once, so it must reduce over the LAST axis
(np.mean(x, axis=-1), not np.mean(x)). All systems share the same resampled patients, which makes
the intervals paired, as the proposal requires.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.stats import binomtest

MIN_GAP = 0.02      # below this average B1 - B6 gap recovery is not reported (proposal, Eq. 1)


@dataclass(frozen=True)
class BootResult:
    estimate: float
    low: float
    high: float
    n_boot: int
    n_undefined: int        # resamples where the statistic had a zero denominator (dropped)

    def excludes(self, value: float) -> bool:
        return not (self.low <= value <= self.high)


def paired_bootstrap(arrays: dict, stat: Callable, n_boot: int = 10000, seed: int = 0,
                     level: float = 0.95) -> BootResult:
    """Percentile CI of stat(arrays) over patient resamples.

    arrays: name -> 1-D array with one value per patient (same order for all names).
    stat:   receives {name: array of shape (m, n_patients)} and returns shape (m,).
    """
    arrays = {k: np.asarray(v, dtype=float) for k, v in arrays.items()}
    n = {len(v) for v in arrays.values()}
    if len(n) != 1:
        raise ValueError("all arrays need one value per patient, in the same order")
    n = n.pop()
    point = float(stat({k: v[None, :] for k, v in arrays.items()})[0])
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    boot = np.asarray(stat({k: v[idx] for k, v in arrays.items()}), dtype=float)
    ok = boot[~np.isnan(boot)]
    if len(ok) == 0:
        return BootResult(point, float("nan"), float("nan"), n_boot, n_boot)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(ok, [tail, 100 - tail])
    return BootResult(point, float(low), float(high), n_boot, int(n_boot - len(ok)))


# ---- ready-made statistics ----

def mean_of(key: str) -> Callable:
    return lambda d: d[key].mean(axis=-1)


def ratio_of_sums(num: str, den: str) -> Callable:
    """sum(num) / sum(den) over the resampled patients; nan when the denominator is zero."""
    def stat(d):
        n, m = d[num].sum(axis=-1), d[den].sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(m > 0, n / m, np.nan)
    return stat


def recovery_stat(system: str, oracle: str, floor: str, min_gap: float = MIN_GAP) -> Callable:
    """(mean_sys - mean_floor) / (mean_oracle - mean_floor): a ratio of averages, never an average
    of ratios. Resamples whose B1-B6 gap is under min_gap are undefined (nan)."""
    def stat(d):
        s, o, f = d[system].mean(axis=-1), d[oracle].mean(axis=-1), d[floor].mean(axis=-1)
        gap = o - f
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(gap) >= min_gap, (s - f) / gap, np.nan)
    return stat


def recovery(system, oracle, floor, min_gap: float = MIN_GAP) -> Optional[float]:
    """Recovery of the per-patient NDCG arrays, or None if the average gap is under min_gap
    (then report raw NDCG differences instead)."""
    s, o, f = (float(np.mean(x)) for x in (system, oracle, floor))
    return None if abs(o - f) < min_gap else (s - f) / (o - f)


def recovery_difference_stat(a: str, b: str, oracle: str, floor: str, min_gap: float = MIN_GAP) -> Callable:
    """recovery(a) - recovery(b) inside every resample, e.g. B7 vs B4 for H1."""
    ra, rb = recovery_stat(a, oracle, floor, min_gap), recovery_stat(b, oracle, floor, min_gap)
    return lambda d: ra(d) - rb(d)


# ---- McNemar ----

@dataclass(frozen=True)
class McNemarResult:
    a_only: int         # A right, B wrong
    b_only: int         # B right, A wrong
    p_value: float      # exact two-sided binomial test on the discordant pairs


def mcnemar(a_correct, b_correct) -> McNemarResult:
    """Exact McNemar test for paired yes/no outcomes (rule- or trial-level decisions of two systems).
    Items are treated as independent, which overstates certainty when many items share a patient;
    the patient-level bootstrap is the main result and this is the paired test the proposal asks for."""
    a, b = np.asarray(a_correct, dtype=bool), np.asarray(b_correct, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("both systems need one outcome per item, in the same order")
    n_a, n_b = int((a & ~b).sum()), int((~a & b).sum())
    p = 1.0 if n_a + n_b == 0 else float(binomtest(min(n_a, n_b), n_a + n_b, 0.5).pvalue)
    return McNemarResult(n_a, n_b, p)


# ---- verdicts exactly as worded in the proposal ----

def h1_verdict(gain: BootResult, false_eligible_not_higher: bool, target: float = 0.10) -> str:
    """H1: recovery at least `target` above fixed-order questioning with no higher false-eligible rate.
    supported: estimate >= target, CI excludes zero, false-eligible not higher.
    not supported: CI upper bound below target. Otherwise inconclusive."""
    if gain.high < target:
        return "not supported"
    if gain.estimate >= target and gain.excludes(0.0) and false_eligible_not_higher:
        return "supported"
    return "inconclusive"


def h2_verdict(rate: BootResult, alpha: float = 0.05) -> str:
    """H2: supported if the upper 95% bound of the harmful-inference rate is <= alpha,
    contradicted if the lower bound is > alpha, inconclusive if the interval crosses alpha."""
    if rate.high <= alpha:
        return "supported"
    if rate.low > alpha:
        return "contradicted"
    return "inconclusive"
