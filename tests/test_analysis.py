"""Seeded synthetic checks for the analysis algorithms and their boundary cases."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

from signalscope.agent.summarizer import check_summary
from signalscope.analysis import model as model_module
from signalscope.analysis.align import align, has_gaps, lagged_pairs
from signalscope.analysis.model import _features, fit_and_forecast
from signalscope.analysis.report import analyze
from signalscope.analysis.stats import best_lag, bonferroni, differenced, lag_scan, pearson, spearman
from signalscope.sources.base import TimeSeries

pytestmark = pytest.mark.filterwarnings("error")


def ts(values, start="2026-01-01", label="X", unit="u", source="Synthetic", future=None, index=None):
    idx = index if index is not None else pd.date_range(start, periods=len(values), name="date")
    return TimeSeries(key=label.lower(), label=label, unit=unit, source=source, url="", license="",
                      history=pd.Series(np.asarray(values, float), index=idx, name=label.lower()), future=future)


def test_linear_relation_and_input_immutability():
    rng = np.random.default_rng(0)
    x = rng.normal(20, 5, 90)
    driver, target = ts(x), ts(3 * x + rng.normal(0, 1, 90), label="Y")
    original_driver, original_target = driver.history.copy(deep=True), target.history.copy(deep=True)
    report = analyze(driver, target)
    assert report.verdict == "strong" and report.pearson.r > 0.95 and report.lag == 0
    assert report.model.slope == pytest.approx(3, abs=0.2)
    assert report.facts()["direction"] == "positive"
    pd.testing.assert_series_equal(driver.history, original_driver)
    pd.testing.assert_series_equal(target.history, original_target)
    pd.testing.assert_frame_equal(report.data, report.pairs)
    residuals = report.pairs.target - report.model.fitted
    assert report.model.residual_std == pytest.approx(np.sqrt(np.sum(residuals**2) / (90 - 7 - 1)))


def test_negative_relation():
    rng = np.random.default_rng(0)
    x = rng.normal(20, 5, 90)
    report = analyze(ts(x), ts(-2 * x + rng.normal(0, 1, 90), label="Y"))
    assert report.verdict == "strong" and report.pearson.r < -0.95
    assert report.facts()["direction"] == "negative"


def test_false_positive_rates_on_unrelated_series():
    # Property over many datasets instead of one lucky seed. Measured on 300 seeds: white noise
    # 4.7 % strong/moderate (nominal 5 % despite 15 lags), random walks 0 % (naive Pearson: 75.7 %).
    def claimed(make) -> int:
        hits = 0
        for seed in range(60):
            rng = np.random.default_rng(seed)
            x, y = make(rng)
            hits += analyze(ts(x), ts(y, label="Y")).verdict in {"strong", "moderate"}
        return hits

    assert claimed(lambda r: (r.normal(size=90), r.normal(size=90))) <= 6
    assert claimed(lambda r: (r.normal(size=90).cumsum(), r.normal(size=90).cumsum())) == 0


def test_identifiable_lag_and_bonferroni():
    rng = np.random.default_rng(0)
    x = rng.normal(size=120)
    driver = ts(x)
    target = ts(x[:-3] + 0.3 * rng.normal(size=117), index=driver.history.index[3:], label="Y")
    scan = lag_scan(driver.history, target.history)
    assert best_lag(scan) == 3
    report = analyze(driver, target)
    assert report.lag == 3 and report.lags_tested == 15
    raw = pearson(report.pairs.driver, report.pairs.target)
    assert report.pearson.p_value == pytest.approx(bonferroni(raw.p_value, report.lags_tested))
    assert any("X leads Y by 3 day(s)" in note and "testing 15 lags" in note for note in report.caveats)
    reversed_report = analyze(target, driver)
    assert reversed_report.lag == -3
    assert any("Y lags behind X by 3 day(s)" in note for note in reversed_report.caveats)


def test_shared_trend_is_downgraded():
    rng = np.random.default_rng(1)
    days = np.arange(90)
    x, y = 0.5 * days + rng.normal(0, 2, 90), 0.5 * days + rng.normal(0, 2, 90)
    report = analyze(ts(x), ts(y, label="Y"))
    assert abs(report.pearson.r) > 0.8 and abs(report.differenced.r) < 0.15
    assert report.verdict == "weak"
    assert "The correlation is likely driven by a shared trend or season, not by a direct link." in report.caveats
    days = np.arange(90)
    for seed in range(30):  # property: a shared trend alone is never reported as strong/moderate
        r = np.random.default_rng(seed)
        verdict = analyze(ts(0.5 * days + r.normal(0, 2, 90)), ts(0.5 * days + r.normal(0, 2, 90), label="Y")).verdict
        assert verdict in {"none", "weak"}


def test_negative_lag_is_descriptive_only():
    rng = np.random.default_rng(0)
    y = rng.normal(size=120)
    target = ts(y, label="Y")
    driver = ts(y[:-3] + 0.3 * rng.normal(size=117), index=target.history.index[3:])  # target moves first
    report = analyze(driver, target)
    assert report.lag == -3
    assert report.model is not None and report.model.lag >= 0
    assert any("only uses past values" in c for c in report.caveats)
    with pytest.raises(ValueError):
        fit_and_forecast(driver.history, target.history, -3, None)


def test_autocorrelation_reduces_effective_sample_size():
    rng = np.random.default_rng(0)
    days = np.arange(90)
    x = pd.Series(np.sin(2 * np.pi * days / 30))
    y = pd.Series(np.sin(2 * np.pi * days / 30 + 0.3) + 0.05 * rng.normal(size=90))
    result = pearson(x, y)
    assert result.n_eff < len(x) / 3
    assert result.p_value > pearsonr(x, y).pvalue
    assert result.n_eff == pytest.approx(max(3, 90 * (1 - x.autocorr() * y.autocorr()) / (1 + x.autocorr() * y.autocorr())))
    assert spearman(x, y) == pearson(x.rank(), y.rank())


def test_insufficient_and_empty_data():
    report = analyze(ts(np.arange(8)), ts(np.arange(8), label="Y"))
    assert report.verdict == "insufficient" and report.n == 8
    assert report.pearson is report.spearman is report.differenced is report.model is None
    assert report.lag == report.lags_tested == 0
    assert report.lag_scan.empty and list(report.lag_scan) == ["lag", "r", "p_value", "n"]
    assert report.verdict_reason == "Only 8 overlapping days; at least 10 are needed."
    assert report.caveats == ["Small sample (8 days): results are fragile.", "Correlation does not imply causation."]
    empty = analyze(ts([]), ts([], label="Y"))
    assert empty.facts()["window_start"] is empty.facts()["window_end"] is None
    assert not has_gaps(empty.data)
    assert fit_and_forecast(ts([]).history, ts([]).history, 0, None) is None


def test_constant_inputs_without_warnings():
    report = analyze(ts(np.ones(60)), ts(np.arange(60), label="Y"))
    assert np.isnan(report.pearson.r) and report.verdict == "none"
    assert report.facts()["pearson_r"] is report.facts()["pearson_ci"] is None
    assert report.facts()["direction"] == "none"
    constant_target = fit_and_forecast(ts(np.arange(60)).history, ts(np.ones(60)).history, 0, None)
    assert constant_target.mae_baseline == 0 and constant_target.skill is None
    assert constant_target.r2_test is None
    # A constant lagged slice must not call autocorr even when the whole series varies.
    assert np.isfinite(pearson(pd.Series([0., 1., 1., 1.]), pd.Series([3., 2., 1., 0.])).r)


def test_gappy_target_uses_sunday_driver_and_excludes_weekend_forecasts():
    rng = np.random.default_rng(0)
    driver = ts(rng.normal(size=60))
    weekdays = driver.history.index[driver.history.index.dayofweek < 5]
    target = ts(driver.history.loc[weekdays].to_numpy(), index=weekdays, label="FX", source="Frankfurter (ECB)")
    pairs = lagged_pairs(driver.history, target.history, 1)
    assert pairs.loc["2026-01-05", "driver"] == driver.history.loc["2026-01-04"]
    assert has_gaps(align(driver, target))
    report = analyze(driver, target)
    assert "Exchange rates exist on business days only; weekends are excluded, not filled." in report.caveats
    future = pd.Series(rng.normal(size=7), index=pd.date_range("2026-03-02", periods=7))
    fitted = fit_and_forecast(driver.history, target.history, 0, future)
    assert fitted.forecast is not None and (fitted.forecast.index.dayofweek < 5).all()


def test_differencing_never_bridges_calendar_gaps():
    index = pd.date_range("2026-01-01", periods=20).delete(10)
    values = np.arange(19, dtype=float)**2
    frame = pd.DataFrame({"driver": values, "target": values * 2}, index=index)
    original = frame.copy(deep=True)
    result = differenced(frame)
    assert result.n == 17  # 18 adjacent rows minus the pair across the missing day.
    assert result.r == pytest.approx(1)
    assert differenced(frame.iloc[:8]).n == 7
    assert np.isnan(differenced(frame.iloc[:8]).r)
    pd.testing.assert_frame_equal(frame, original)


def test_forecasts_with_future_driver_or_positive_lag():
    rng = np.random.default_rng(0)
    driver = ts(rng.normal(size=90))
    target = ts(2 * driver.history.to_numpy() + rng.normal(0, 0.5, 90), label="Y")
    expected_dates = pd.date_range("2026-04-01", periods=7, name="date")
    future = pd.Series(rng.normal(size=7), index=expected_dates)
    saved = future.copy(deep=True)
    result = fit_and_forecast(driver.history, target.history, 0, future)
    assert result.forecast.index.equals(expected_dates)
    assert list(result.forecast) == ["yhat", "lower", "upper"]
    assert (result.forecast.lower < result.forecast.yhat).all()
    assert (result.forecast.yhat < result.forecast.upper).all()
    np.testing.assert_allclose(result.forecast.yhat - result.forecast.lower, 1.96 * result.residual_std)
    assert len(fit_and_forecast(driver.history, target.history, 2, None).forecast) == 2
    assert fit_and_forecast(driver.history, target.history, 0, None).forecast is None
    overlap = pd.concat([pd.Series([-999.], index=driver.history.index[-1:]), future])
    a = fit_and_forecast(driver.history, target.history, 2, overlap)
    b = fit_and_forecast(driver.history, target.history, 2, future)
    pd.testing.assert_frame_equal(a.forecast, b.forecast)
    pd.testing.assert_series_equal(future, saved)


def test_weekday_features_keep_fixed_columns():
    rng = np.random.default_rng(0)
    driver = ts(rng.normal(size=60))
    index = driver.history.index
    y = 3 * driver.history.to_numpy() + 5 * index.dayofweek.to_numpy() + rng.normal(0, 0.2, 60)
    keep = ~((index >= index[-12]) & index.dayofweek.isin([0, 2]))
    target = ts(y[keep], index=index[keep], label="Y")
    result = fit_and_forecast(driver.history, target.history, 0, None)
    assert result.uses_weekday and result.n_test > 0 and np.isfinite(result.mae_test)
    assert result.slope == pytest.approx(3, abs=0.2)
    full = _features(index, driver.history, True)
    subset = _features(index[index.dayofweek == 0], driver.history[index.dayofweek == 0], True)
    assert list(full) == list(subset) == ["driver", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6"]


def test_evaluation_counts_and_training_only_lag_selection(monkeypatch):
    rng = np.random.default_rng(0)
    driver = ts(rng.normal(size=90))
    target = ts(3 * driver.history.to_numpy() + rng.normal(0, 0.2, 90), label="Y")
    observed = []
    original_scan = model_module.lag_scan

    def capture_scan(x, y):
        observed.append((x.index.max(), y.index.max()))
        return original_scan(x, y)

    monkeypatch.setattr(model_module, "lag_scan", capture_scan)
    result = fit_and_forecast(driver.history, target.history, 0, None)
    assert result.n_fit == 90 and result.n_train == 72 and result.n_test == 18
    assert result.eval_lag == 0 and result.skill is not None
    assert observed == [(driver.history.index[71], target.history.index[71])]
    short = fit_and_forecast(driver.history.iloc[:20], target.history.iloc[:20], 0, None)
    assert short.n_train == short.n_test == 0 and short.eval_lag is None
    assert short.skill is short.r2_test is short.mae_test is short.mae_baseline is None


def test_facts_are_stable_json_and_summary_numbers_are_grounded():
    rng = np.random.default_rng(0)
    x = rng.normal(20, 5, 90)
    future = pd.Series(rng.normal(20, 5, 7), index=pd.date_range("2026-04-01", periods=7))
    reports = [analyze(ts(x, future=future), ts(3 * x + rng.normal(size=90), label="Y")),
               analyze(ts(x[:8]), ts(x[:8], label="Y")), analyze(ts(np.ones(90)), ts(x, label="Y"))]
    keys = set(reports[0].facts())
    assert keys == set("driver target driver_unit target_unit window_start window_end n n_eff lag_days "
                       "pearson_r spearman_rho differenced_r pearson_p pearson_ci direction verdict "
                       "verdict_reason slope model_skill r2_test forecast caveats".split())
    for report in reports:
        facts = report.facts()
        assert set(facts) == keys
        assert json.loads(json.dumps(facts, allow_nan=False)) == facts
        issues = check_summary(facts["verdict_reason"], facts, "")
        assert not any("numbers not in the facts" in issue for issue in issues)
    assert len(reports[0].facts()["forecast"]) == 7


def test_statistics_boundary_rules():
    scan = pd.DataFrame({"lag": [-3, -2, 0, 2], "r": [0.8, -0.8, 0.1, 0.8]})
    assert best_lag(scan) == 2
    assert best_lag(scan, min_gain=0.8) == 0
    assert best_lag(scan[scan.lag != 0]) == 0
    assert best_lag(scan.assign(r=np.nan)) == 0
    empty = lag_scan(ts([]).history, ts([]).history)
    assert empty.empty and list(empty) == ["lag", "r", "p_value", "n"]
    assert best_lag(empty) == 0
    assert np.isnan(pearson(pd.Series([], dtype=float), pd.Series([], dtype=float)).r)
    perfect = pearson(pd.Series([0., 1., 2., 3.]), pd.Series([0., 2., 4., 6.]))
    assert perfect.p_value == 0 and perfect.n_eff == 3 and np.isnan(perfect.ci_low)
    assert bonferroni(0.2, 15) == 1 and np.isnan(bonferroni(np.nan, 15))


def test_exogenous_driver_only_scans_non_negative_lags():
    rng = np.random.default_rng(0)
    y = rng.normal(size=120)
    target = ts(y, label="Y")
    weather = TimeSeries("w", "Weather", "u", "Open-Meteo", "", "",
                         pd.Series(y[:-3] + 0.3 * rng.normal(size=117), index=target.history.index[3:], name="w"),
                         exogenous=True)
    report = analyze(weather, target)
    assert report.lag >= 0 and report.lags_tested == 8 and (report.lag_scan.lag >= 0).all()
