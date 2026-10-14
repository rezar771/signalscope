"""Chart contract checks using real reports built from deterministic synthetic data."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from signalscope.analysis.report import analyze
from signalscope.sources.base import TimeSeries
from signalscope.trace import TraceEvent
from signalscope.ui import charts

pytestmark = pytest.mark.filterwarnings("error")


def ts(values, start="2026-01-01", label="Driver", unit="u", future=None):
    index = pd.date_range(start, periods=len(values), name="date")
    return TimeSeries(key=label.lower(), label=label, unit=unit, source="Synthetic", url="", license="",
                      history=pd.Series(np.asarray(values, dtype=float), index=index, name=label.lower()),
                      future=future)


@pytest.fixture
def report():
    index = pd.date_range("2026-01-01", periods=80, name="date")
    x = np.linspace(0, 12, len(index)) + np.sin(np.arange(len(index)))
    future_index = pd.date_range(index[-1] + pd.Timedelta(days=1), periods=7, name="date")
    future = pd.Series(np.linspace(12, 14, 7), index=future_index)
    driver = ts(x, future=future)
    target = ts(4 * x + 2 * np.cos(np.arange(len(index))), label="Target", unit="MW")
    return analyze(driver, target)


@pytest.fixture
def insufficient_report():
    return analyze(ts(np.arange(8)), ts(np.arange(8), label="Target"))


def test_timeseries_chart_uses_dual_axes_theme_and_forecast(report):
    fig = charts.timeseries_chart(report)
    assert isinstance(fig, go.Figure)
    assert fig.layout.yaxis.title.text == "u" and fig.layout.yaxis.title.font.color == charts.DRIVER_COLOR
    assert fig.layout.yaxis2.title.text == "MW"
    assert charts.BRAND_RED == charts.TARGET_COLOR
    assert fig.layout.paper_bgcolor == charts.BG
    assert any(trace.name == "model forecast (95% band)" for trace in fig.data)
    assert any(trace.name in ("driver forecast", "weather/driver forecast") for trace in fig.data)


def test_scatter_uses_lag_adjusted_pairs_and_safe_ols(report, insufficient_report):
    fig = charts.scatter_chart(report)
    assert isinstance(fig, go.Figure)
    points = next(trace for trace in fig.data if trace.name == "Observed pairs")
    assert len(points.x) == report.n
    assert [list(stop) for stop in points.marker.colorscale] == charts.DATE_SCALE
    assert points.marker.colorbar.title.text == "date"
    assert any(trace.name == "OLS fit" for trace in fig.data)
    assert any(trace.name == "95% mean CI" for trace in fig.data)
    assert any(item.text == "Not enough overlapping data" for item in charts.scatter_chart(insufficient_report).layout.annotations)


def test_lag_chart_marks_best_lag_and_positive_driver_convention(report):
    fig = charts.lag_chart(report)
    assert isinstance(fig, go.Figure)
    bars = next(trace for trace in fig.data if isinstance(trace, go.Bar))
    best_index = list(bars.x).index(report.lag)
    assert bars.marker.color[best_index] == charts.TARGET_COLOR
    assert "positive: Driver leads" == fig.layout.xaxis.title.text.split("— ")[1]
    assert any(shape.line.dash == "dash" for shape in fig.layout.shapes)


def test_forecast_chart_returns_none_without_forecast_and_figure_with_one(report, insufficient_report):
    assert report.model.forecast is not None
    assert isinstance(charts.forecast_chart(report), go.Figure)
    assert charts.forecast_chart(insufficient_report) is None


def test_trace_chart_has_one_bar_row_per_finished_event_and_kind_colors():
    events = [
        TraceEvent("api", "Fetch history", "ok", 10.0, 22.5, {"rows": 30, "cache_hit": True}),
        TraceEvent("llm", "Plan", "running", 35.0),
        TraceEvent("analysis", "Compute report", "ok", 60.0, 4.0, {"tokens": 12}),
        TraceEvent("system", "Fallback", "warning", 70.0),
    ]
    fig = charts.trace_chart(events)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == sum(event.finished for event in events)
    assert fig.layout.height == 40 + 28 * 3
    assert fig.layout.xaxis.title.text == "ms since start"
    assert fig.data[0].marker.color == charts.KIND_COLORS["api"]
    assert "cache_hit: True" in fig.data[0].customdata[0][0]


def test_nan_and_constant_scatter_inputs_do_not_emit_warnings():
    x = ts(np.ones(20))
    y = ts(np.ones(20), label="Target")
    constant = analyze(x, y)
    fig = charts.scatter_chart(constant)
    assert isinstance(fig, go.Figure)
    assert not any(trace.name == "OLS fit" for trace in fig.data)
    assert isinstance(charts.lag_chart(constant), go.Figure)
