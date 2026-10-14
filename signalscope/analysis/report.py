"""Evidence verdicts and a stable, JSON-safe facts contract for the UI and summary."""

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Literal

import pandas as pd

from signalscope.analysis.align import align, has_gaps, lagged_pairs
from signalscope.analysis.model import ModelResult, fit_and_forecast
from signalscope.analysis.stats import (
    MIN_N, Correlation, best_lag, bonferroni, differenced, lag_scan, pearson, spearman,
)
from signalscope.sources.base import TimeSeries

Verdict = Literal["strong", "moderate", "weak", "none", "insufficient"]


def _round(value: float | None, digits: int = 2) -> float | None:
    return round(float(value), digits) if value is not None and isfinite(value) else None


def _r2(value: float | None) -> float | None:
    return _round(value, 2)


def _sig3(value: float) -> float | None:
    return float(f"{value:.3g}") if isfinite(value) else None


def _p_value(value: float | None) -> str | float | None:
    return "<0.001" if value is not None and isfinite(value) and value < 0.001 else _round(value, 3)


def _p(value: float) -> str:
    rounded = _p_value(value)
    return "< 0.001" if rounded == "<0.001" else f"= {rounded:.3f}"


@dataclass(frozen=True)
class AnalysisReport:
    """Observed data, statistics and model results without hidden I/O."""
    driver: TimeSeries
    target: TimeSeries
    data: pd.DataFrame
    pairs: pd.DataFrame
    n: int
    lag: int
    lags_tested: int
    pearson: Correlation | None
    spearman: Correlation | None
    differenced: Correlation | None
    lag_scan: pd.DataFrame
    model: ModelResult | None
    verdict: Verdict
    verdict_reason: str
    caveats: list[str]

    def facts(self) -> dict[str, Any]:
        """Return rounded facts with identical keys and no non-finite JSON numbers."""
        c, model = self.pearson, self.model
        r = c.r if c else float("nan")
        ci = [_r2(c.ci_low), _r2(c.ci_high)] if c else None
        forecast = None if model is None or model.forecast is None else [
            {"date": date.date().isoformat(), **{key: _sig3(row[key]) for key in ("yhat", "lower", "upper")}}
            for date, row in model.forecast.iterrows()
        ]
        return {
            "driver": self.driver.label, "target": self.target.label,
            "driver_unit": self.driver.unit, "target_unit": self.target.unit,
            "window_start": self.data.index.min().date().isoformat() if self.n else None,
            "window_end": self.data.index.max().date().isoformat() if self.n else None,
            "n": self.n, "n_eff": _round(c.n_eff, 1) if c else None, "lag_days": self.lag,
            "pearson_r": _r2(r), "spearman_rho": _r2(self.spearman.r) if self.spearman else None,
            "differenced_r": _r2(self.differenced.r) if self.differenced else None,
            "pearson_p": _p_value(c.p_value) if c else None,
            "pearson_ci": ci if ci is not None and None not in ci else None,
            "direction": "none" if not isfinite(r) or abs(r) < 0.1 else ("positive" if r > 0 else "negative"),
            "verdict": self.verdict, "verdict_reason": self.verdict_reason,
            "slope": _sig3(model.slope) if model else None,
            "model_skill": _r2(model.skill) if model else None, "r2_test": _r2(model.r2_test) if model else None,
            "forecast": forecast, "caveats": list(self.caveats),
        }


def _verdict(c: Correlation, changes: Correlation, trend: bool, n: int) -> tuple[Verdict, str]:
    if not isfinite(c.r):
        return "none", "One of the series is constant, so no correlation can be computed."
    r, p, rd = _r2(c.r), _p(c.p_value), _r2(changes.r)
    if abs(c.r) >= 0.5 and c.p_value < 0.01 and not trend:
        verdict = "strong"
    elif abs(c.r) >= 0.3 and c.p_value < 0.05 and not trend:
        verdict = "moderate"
    else:
        verdict = "weak" if c.p_value < 0.05 or trend else "none"
    if verdict in ("strong", "moderate"):
        reason = f"r = {r} with p {p} over {n} days"
        reason += f", confirmed on day-to-day changes (r = {rd})." if rd is not None else "."
    elif trend:
        reason = f"Both series share a trend, but their day-to-day changes are unrelated (r = {rd})."
    elif verdict == "weak":
        reason = f"r = {r} with p {p}: detectable but small."
    else:
        reason = f"r = {r} with p {p}: no reliable relationship over {n} days."
    return verdict, reason


def analyze(driver: TimeSeries, target: TimeSeries) -> AnalysisReport:
    """Align observations, assess lagged evidence, evaluate a model and report caveats."""
    data = align(driver, target)
    n, lag, trend = len(data), 0, False
    pairs = data
    scan = pd.DataFrame(columns=["lag", "r", "p_value", "n"])
    c = ranked = changes = model = None
    verdict, reason = "insufficient", f"Only {n} overlapping days; at least {MIN_N} are needed."
    if n >= MIN_N:
        scan = lag_scan(driver.history, target.history)
        if driver.exogenous:
            # Weather/air quality cannot be caused by the target, so "target leads driver" lags are
            # physically meaningless (live run: "ice cream views lead Milan temperature by 7 days").
            scan = scan[scan.lag >= 0].reset_index(drop=True)
        lag = best_lag(scan)
        pairs = lagged_pairs(driver.history, target.history, lag)
        c, ranked, changes = pearson(pairs.driver, pairs.target), spearman(pairs.driver, pairs.target), differenced(pairs)
        if lag:
            c = replace(c, p_value=bonferroni(c.p_value, len(scan)))
        # The descriptive lag may be negative (target moves first); the predictive model may not.
        model_lag = lag if lag >= 0 else best_lag(scan[scan.lag >= 0])
        model = fit_and_forecast(driver.history, target.history, model_lag, driver.future)
        trend = abs(c.r) >= 0.3 and isfinite(changes.r) and abs(changes.r) < 0.15
        verdict, reason = _verdict(c, changes, trend, n)
    caveats = []
    if n < 30:
        caveats.append(f"Small sample ({n} days): results are fragile.")
    if trend:
        caveats.append("The correlation is likely driven by a shared trend or season, not by a direct link.")
    if lag:
        relation = "leads" if lag > 0 else "lags behind"
        caveats.append(f"Strongest relation when {driver.label} {relation} {target.label} by {abs(lag)} day(s); "
                       f"the p-value is adjusted for testing {len(scan)} lags."
                       + (" The forecast model only uses past values of the driver." if lag < 0 else ""))
    if model and model.skill is not None and model.skill <= 0:
        caveats.append("The model does not beat a naive baseline; treat the forecast as unreliable.")
    if has_gaps(data):
        caveats.append("Exchange rates exist on business days only; weekends are excluded, not filled."
                       if any(s.source.startswith("Frankfurter") for s in (driver, target)) else
                       "Some days are missing in one of the series; they are excluded, not filled.")
    caveats.append("Correlation does not imply causation.")
    return AnalysisReport(driver, target, data, pairs, n, lag, len(scan), c, ranked, changes,
                          scan, model, verdict, reason, caveats)
