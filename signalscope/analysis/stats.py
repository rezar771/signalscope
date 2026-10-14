"""Guarded correlations, calendar lag scans and shared-trend diagnostics."""

from dataclasses import dataclass
from math import atanh, isfinite, sqrt, tanh

import numpy as np
import pandas as pd
from scipy.stats import t

from signalscope.analysis.align import lagged_pairs

MIN_N = 10
MAX_LAG = 7


@dataclass(frozen=True)
class Correlation:
    """Correlation with autocorrelation-adjusted uncertainty."""
    r: float
    p_value: float
    ci_low: float
    ci_high: float
    n: int
    n_eff: float


def _undefined(n: int) -> Correlation:
    return Correlation(np.nan, np.nan, np.nan, np.nan, n, np.nan)


def _autocorr(values: pd.Series) -> float:
    if len(values) < 3 or values.iloc[:-1].std(ddof=0) == 0 or values.iloc[1:].std(ddof=0) == 0:
        return 0.0
    # Positional lag-1 autocorrelation is the specified approximation for gappy data.
    rho = float(values.autocorr(1))
    return rho if isfinite(rho) else 0.0


def pearson(x: pd.Series, y: pd.Series) -> Correlation:
    """Compute Pearson r, effective sample size, adjusted p-value and Fisher CI."""
    n = len(x)
    if n < 4 or x.std(ddof=0) == 0 or y.std(ddof=0) == 0:
        return _undefined(n)
    r = float(np.clip(np.corrcoef(x, y)[0, 1], -1, 1))
    product = _autocorr(x) * _autocorr(y)
    n_eff = float(np.clip(n * (1 - product) / (1 + product), 3, n)) if 1 + product > 0 else float(n)
    p = 0.0 if abs(r) >= 0.999999 else float(2 * t.sf(abs(r * sqrt((n_eff - 2) / (1 - r**2))), n_eff - 2))
    low = high = np.nan
    if n_eff > 3:
        z, se = atanh(float(np.clip(r, -0.999999, 0.999999))), 1 / sqrt(n_eff - 3)
        low, high = tanh(z - 1.96 * se), tanh(z + 1.96 * se)
    return Correlation(r, p, low, high, n, n_eff)


def spearman(x: pd.Series, y: pd.Series) -> Correlation:
    """Compute the same adjusted correlation on ranks."""
    return pearson(x.rank(), y.rank())


def differenced(df: pd.DataFrame) -> Correlation:
    """Correlate consecutive-day changes, never differences across calendar gaps."""
    changes = df.asfreq("D").diff().dropna()
    return _undefined(len(changes)) if len(changes) < MIN_N else pearson(changes.driver, changes.target)


def lag_scan(driver: pd.Series, target: pd.Series, max_lag: int = MAX_LAG) -> pd.DataFrame:
    """Scan calendar lags with at least MIN_N observed pairs."""
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        pairs = lagged_pairs(driver, target, lag)
        if len(pairs) >= MIN_N:
            c = pearson(pairs.driver, pairs.target)
            rows.append({"lag": lag, "r": c.r, "p_value": c.p_value, "n": c.n})
    return pd.DataFrame(rows, columns=["lag", "r", "p_value", "n"])


def best_lag(scan: pd.DataFrame, min_gain: float = 0.05) -> int:
    """Prefer zero unless a lag improves |r|; ties favor smaller, then positive lags."""
    valid = scan.dropna(subset=["r"])
    zero = valid.loc[valid.lag == 0, "r"]
    if zero.empty:
        return 0
    best = valid.assign(strength=valid.r.abs(), distance=valid.lag.abs()).sort_values(
        ["strength", "distance", "lag"], ascending=[False, True, False],
    ).iloc[0]
    return int(best.lag) if abs(best.r) - abs(zero.iloc[0]) > min_gain else 0


def bonferroni(p: float, tests: int) -> float:
    """Correct for the tested lags while preserving undefined p-values."""
    return min(1.0, p * tests) if isfinite(p) else np.nan
