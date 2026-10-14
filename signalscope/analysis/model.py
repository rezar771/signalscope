"""Chronological regression evaluation and forecasts from available driver dates."""

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from signalscope.analysis.align import lagged_pairs
from signalscope.analysis.stats import MIN_N, best_lag, lag_scan


@dataclass(frozen=True)
class ModelResult:
    """Final fitted model, separate hold-out metrics, and optional forecasts."""
    lag: int
    slope: float
    intercept: float
    uses_weekday: bool
    n_fit: int
    n_train: int
    n_test: int
    eval_lag: int | None
    r2_test: float | None
    mae_test: float | None
    mae_baseline: float | None
    skill: float | None
    residual_std: float
    fitted: pd.Series
    forecast: pd.DataFrame | None


def _features(index: pd.DatetimeIndex, driver_values: pd.Series, uses_weekday: bool) -> pd.DataFrame:
    features = pd.DataFrame({"driver": np.asarray(driver_values)}, index=index)
    if uses_weekday:
        weekdays = pd.get_dummies(pd.Categorical(index.dayofweek, categories=range(7)),
                                 prefix="dow", drop_first=True, dtype=float)
        weekdays.index = index
        features = pd.concat([features, weekdays], axis=1)
    return features


def fit_and_forecast(driver: pd.Series, target: pd.Series, lag: int,
                     driver_future: pd.Series | None, horizon: int = 7) -> ModelResult | None:
    """Evaluate on later dates, refit on all pairs, then forecast available dates.

    `lag` must be >= 0: the model is predictive, so it may only use past driver values.
    """
    if lag < 0:
        raise ValueError("fit_and_forecast needs lag >= 0 (driver must not come from the future)")
    pairs = lagged_pairs(driver, target, lag)
    n = len(pairs)
    if n < MIN_N:
        return None
    uses_weekday = n >= 28
    n_train = n_test = 0
    eval_lag = r2_test = mae_test = mae_baseline = skill = None
    if n >= 30:
        cutoff = pairs.index[n - max(7, round(0.2 * n))]
        scan = lag_scan(driver[driver.index < cutoff], target[target.index < cutoff])
        # Prospective only: a negative lag would predict the target from later driver values.
        selected = best_lag(scan[scan.lag >= 0])
        ep = lagged_pairs(driver, target, selected)
        train, test = ep[ep.index < cutoff], ep[ep.index >= cutoff]
        if len(train) >= MIN_N and not test.empty:
            n_train, n_test, eval_lag = len(train), len(test), selected
            evaluation = LinearRegression().fit(_features(train.index, train.driver, uses_weekday), train.target)
            predicted = evaluation.predict(_features(test.index, test.driver, uses_weekday))
            overall = float(train.target.mean())
            baseline = np.full(n_test, overall)
            if uses_weekday:
                by_weekday = train.target.groupby(train.index.dayofweek).mean()
                baseline = np.array([by_weekday.get(day, overall) for day in test.index.dayofweek])
            mae_test = float(np.mean(np.abs(test.target.to_numpy() - predicted)))
            mae_baseline = float(np.mean(np.abs(test.target.to_numpy() - baseline)))
            skill = 1 - mae_test / mae_baseline if mae_baseline != 0 else None
            sst = float(np.sum((test.target.to_numpy() - test.target.mean())**2))
            r2_test = 1 - float(np.sum((test.target.to_numpy() - predicted)**2)) / sst if sst != 0 else None

    features = _features(pairs.index, pairs.driver, uses_weekday)
    model = LinearRegression().fit(features, pairs.target)
    fitted = pd.Series(model.predict(features), index=pairs.index, name="fitted")
    residual_std = sqrt(float(np.sum((pairs.target - fitted)**2)) / max(1, n - features.shape[1] - 1))
    driver_all = driver if driver_future is None else pd.concat(
        [driver, driver_future[~driver_future.index.isin(driver.index)]],
    ).sort_index()
    lagged = driver_all.shift(lag, freq="D")
    dates = pd.date_range(target.index.max() + pd.Timedelta(days=1), periods=horizon, name="date")
    dates = dates[dates.isin(lagged.index) & dates.dayofweek.isin(target.index.dayofweek)]
    forecast = None
    if len(dates):
        yhat = model.predict(_features(dates, lagged.loc[dates], uses_weekday))
        forecast = pd.DataFrame({"yhat": yhat, "lower": yhat - 1.96 * residual_std,
                                 "upper": yhat + 1.96 * residual_std}, index=dates)
    return ModelResult(lag, float(model.coef_[0]), float(model.intercept_), uses_weekday, n,
                       n_train, n_test, eval_lag, r2_test, mae_test, mae_baseline, skill,
                       residual_std, fitted, forecast)
