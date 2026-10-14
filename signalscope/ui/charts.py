"""Pure Plotly chart builders for analysis reports and execution traces."""

from __future__ import annotations

from math import isfinite, sqrt

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import t

from signalscope.analysis.report import AnalysisReport
from signalscope.trace import TraceEvent

BG = "rgba(0,0,0,0)"
GRID = "#1e293b"
AXIS_TEXT = "#94a3b8"
TEXT = "#e2e8f0"
BRAND_RED = "#E04F4E"
DRIVER_COLOR = "#38bdf8"
TARGET_COLOR = BRAND_RED
FORECAST_COLOR = "#a78bfa"
NEUTRAL = "#475569"
KIND_COLORS = {"llm": "#a78bfa", "api": "#38bdf8", "analysis": "#34d399", "system": "#64748b"}
VERDICT_COLORS = {"strong": "#34d399", "moderate": "#a3e635", "weak": "#fbbf24", "none": "#94a3b8", "insufficient": "#64748b"}
FONT = "Inter, -apple-system, Segoe UI, sans-serif"

LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family=FONT, color=TEXT, size=13),
    margin=dict(l=10, r=10, t=30, b=10),
    hoverlabel=dict(bgcolor="#0f172a", bordercolor="#334155", font=dict(family=FONT, color=TEXT)),
    # Legend above the plot: below it, it collided with the date tick labels.
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=AXIS_TEXT, size=12)),
    xaxis=dict(gridcolor=GRID, zeroline=False, color=AXIS_TEXT),
    yaxis=dict(gridcolor=GRID, zeroline=False, color=AXIS_TEXT),
    # automargin: the title reserves its own space above the legend, whatever the legend height.
    title=dict(font=dict(size=15, color=TEXT), x=0.01, xanchor="left", yref="container", y=1,
               yanchor="top", pad=dict(t=12, b=10), automargin=True),
)

DATE_SCALE = [[0, "#0ea5e9"], [1, "#a78bfa"]]


def apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the shared dark theme, including subplot and secondary axes."""
    fig.update_layout(**LAYOUT)
    if fig.layout.hovermode is None:   # default only: scatter/timeline set their own hover mode
        fig.update_layout(hovermode="x unified")
    for axis_name in fig.layout:
        if axis_name.startswith(("xaxis", "yaxis")):
            fig.layout[axis_name].update(gridcolor=GRID, zeroline=False, color=AXIS_TEXT)
    return fig


def _insufficient(report: AnalysisReport, fig: go.Figure) -> go.Figure:
    if report.verdict == "insufficient":
        fig.add_annotation(text="Not enough overlapping data", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(color=AXIS_TEXT, size=14))
    return fig


def _window_end(report: AnalysisReport):
    return report.data.index.max() if not report.data.empty else None


def timeseries_chart(report: AnalysisReport) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    driver, target = report.driver, report.target
    fig.add_trace(go.Scatter(x=driver.history.index, y=driver.history, name=driver.label,
                             mode="lines+markers", line=dict(color=DRIVER_COLOR, width=2.2),
                             marker=dict(size=7, opacity=0.85)), secondary_y=False)
    fig.add_trace(go.Scatter(x=target.history.index, y=target.history, name=target.label,
                             mode="lines+markers", line=dict(color=TARGET_COLOR, width=2.2),
                             marker=dict(size=7, opacity=0.85)), secondary_y=True)
    if driver.future is not None and not driver.future.empty:
        fig.add_trace(go.Scatter(x=driver.future.index, y=driver.future, name="weather/driver forecast" if driver.exogenous else "driver forecast",
                                 mode="lines+markers", line=dict(color=DRIVER_COLOR, width=2.2, dash="dash"),
                                 marker=dict(size=7, opacity=0.85)), secondary_y=False)
    model = report.model
    if model is not None and model.forecast is not None and not model.forecast.empty:
        fc = model.forecast
        # A closed polygon gives the interval a stable colour and opacity in Plotly.
        x_band = list(fc.index) + list(fc.index[::-1])
        y_band = list(fc.upper) + list(fc.lower[::-1])
        fig.add_trace(go.Scatter(x=x_band, y=y_band, mode="lines", line=dict(width=0),
                                 fill="toself", fillcolor="rgba(167, 139, 250, 0.15)",
                                 name="Forecast interval", hoverinfo="skip", showlegend=False), secondary_y=True)
        fig.add_trace(go.Scatter(x=fc.index, y=fc.yhat, name="model forecast (95% band)", mode="lines+markers",
                                 line=dict(color=FORECAST_COLOR, width=2.2, dash="dash"),
                                 marker=dict(size=7, opacity=0.85)), secondary_y=True)
    end = _window_end(report)
    if end is not None:
        fig.add_vline(x=end, line_dash="dot", line_color=NEUTRAL,
                      annotation_text="last observed day", annotation_position="bottom left")
    fig.update_layout(title=f"{driver.label} vs {target.label}")
    # Units only, coloured like their line: full labels are in the legend and were cut off as axis titles.
    fig.update_yaxes(title_text=driver.unit, title_font_color=DRIVER_COLOR, secondary_y=False)
    fig.update_yaxes(title_text=target.unit, title_font_color=TARGET_COLOR, secondary_y=True)
    return apply_theme(_insufficient(report, fig))


def scatter_chart(report: AnalysisReport) -> go.Figure:
    fig = go.Figure()
    pairs = report.pairs
    x = pairs.driver.to_numpy(dtype=float) if "driver" in pairs else np.array([], dtype=float)
    y = pairs.target.to_numpy(dtype=float) if "target" in pairs else np.array([], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    dates = pairs.index[valid]
    if len(x):
        color = dates.asi8.astype(float) / 1e9
        custom = np.asarray([d.strftime("%Y-%m-%d") for d in dates]).reshape(-1, 1)
        # Colour bar ticks as readable dates (raw values are Unix seconds, e.g. 1.79e9).
        tick_dates = dates[np.linspace(0, len(dates) - 1, min(5, len(dates))).round().astype(int)]
        colorbar = dict(title=dict(text="date", side="top"), thickness=12, len=0.9,
                        tickvals=list(tick_dates.asi8.astype(float) / 1e9),
                        ticktext=[d.strftime("%b %d") for d in tick_dates], ticks="outside")
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Observed pairs",
                                 marker=dict(size=7, opacity=0.85, color=color, colorscale=DATE_SCALE,
                                             showscale=True, colorbar=colorbar),
                                 customdata=custom, hovertemplate="date=%{customdata[0]}<br>driver=%{x}<br>target=%{y}<extra></extra>"))
    corr = report.pearson
    r = corr.r if corr is not None and isfinite(corr.r) else float("nan")
    p = corr.p_value if corr is not None else float("nan")
    p_text = "p < 0.001" if isfinite(p) and p < 0.001 else (f"p = {p:.3f}" if isfinite(p) else "p unavailable")
    if len(x) >= 3 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(float(x.min()), float(x.max()), 100)
        y_line = slope * x_line + intercept
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="OLS fit",
                                 line=dict(color=TEXT, width=1.5)))
        df = len(x) - 2
        residual = y - (slope * x + intercept)
        s = sqrt(float(np.sum(residual**2) / df))
        denom = float(np.sum((x - np.mean(x))**2))
        if denom > 0 and isfinite(s):
            margin = float(t.ppf(0.975, df)) * s * np.sqrt(1 / len(x) + (x_line - np.mean(x))**2 / denom)
            fig.add_trace(go.Scatter(x=np.r_[x_line, x_line[::-1]],
                                     y=np.r_[y_line + margin, (y_line - margin)[::-1]],
                                     mode="lines", line=dict(width=0), fill="toself",
                                     fillcolor="rgba(226, 232, 240, 0.15)", name="95% mean CI",
                                     hoverinfo="skip"))
    fig.update_layout(title=f"r = {r:.2f} ({p_text}) · lag {report.lag} d · n = {report.n}", hovermode="closest",
                      margin=dict(l=10, r=70, t=30, b=10))  # room for the colour-bar date labels
    fig.update_xaxes(title_text=f"{report.driver.label} ({report.driver.unit})")
    fig.update_yaxes(title_text=f"{report.target.label} ({report.target.unit})")
    return apply_theme(_insufficient(report, fig))


def lag_chart(report: AnalysisReport) -> go.Figure:
    fig = go.Figure()
    scan = report.lag_scan
    if not scan.empty:
        lags = scan.lag.to_numpy(dtype=int)
        values = scan.r.to_numpy(dtype=float)
        finite = np.isfinite(values)
        lags, values = lags[finite], values[finite]
        colors = [TARGET_COLOR if lag == report.lag else NEUTRAL for lag in lags]
        fig.add_trace(go.Bar(x=lags, y=values, marker_color=colors, name="Correlation by lag"))
        if report.n > 0:
            threshold = 1.96 / sqrt(report.n)
            fig.add_hline(y=threshold, line_dash="dash", line_color=AXIS_TEXT,
                          annotation_text="95 % significance")
            fig.add_hline(y=-threshold, line_dash="dash", line_color=AXIS_TEXT)
    fig.update_layout(title="Correlation across time lags")
    fig.update_xaxes(title_text=f"lag (days) — positive: {report.driver.label} leads")
    fig.update_yaxes(title_text="correlation (r)")
    return apply_theme(_insufficient(report, fig))


def forecast_chart(report: AnalysisReport) -> go.Figure | None:
    model = report.model
    if model is None or model.forecast is None or model.forecast.empty:
        return None
    fig = go.Figure()
    target = report.target.history.tail(30)
    fig.add_trace(go.Scatter(x=target.index, y=target, name=report.target.label,
                             mode="lines+markers", line=dict(color=TARGET_COLOR, width=2.2),
                             marker=dict(size=7, opacity=0.85)))
    fitted = model.fitted.reindex(model.fitted.index.intersection(target.index))
    if not fitted.empty:
        fig.add_trace(go.Scatter(x=fitted.index, y=fitted, name="Fitted", mode="lines",
                                 line=dict(color=DRIVER_COLOR, width=1.5, dash="dot")))
    fc = model.forecast
    fig.add_trace(go.Scatter(x=list(fc.index) + list(fc.index[::-1]),
                             y=list(fc.upper) + list(fc.lower[::-1]), mode="lines",
                             line=dict(width=0), fill="toself",
                             fillcolor="rgba(167, 139, 250, 0.15)", name="Forecast interval",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc.yhat, name="Forecast", mode="lines+markers",
                             line=dict(color=FORECAST_COLOR, width=2.2, dash="dash"),
                             marker=dict(size=7, opacity=0.85)))
    fig.update_layout(title=f"{report.target.label} forecast")
    fig.update_yaxes(title_text=f"{report.target.label} ({report.target.unit})")
    return apply_theme(_insufficient(report, fig))


def trace_chart(events: list[TraceEvent]) -> go.Figure:
    finished = [event for event in events if event.finished]
    fig = go.Figure()
    shown: set[str] = set()
    for index, event in enumerate(finished):
        details = "<br>".join(f"{key}: {value}" for key, value in event.details.items())
        # Numbered so identical messages keep separate rows; truncated so labels don't crush the plot.
        label = f"{index + 1}. {event.message[:42]}{'…' if len(event.message) > 42 else ''}"
        fig.add_trace(go.Bar(x=[max(0.0, float(event.duration_ms or 0.0))],
                             y=[label], base=[event.start_ms], orientation="h",
                             marker_color=KIND_COLORS.get(event.kind, NEUTRAL),
                             name=event.kind, legendgroup=event.kind, showlegend=event.kind not in shown,
                             customdata=[[details]],
                             hovertemplate=f"{event.message}<br>start=%{{base:.1f}} ms<br>duration=%{{x:.1f}} ms<br>%{{customdata[0]}}<extra></extra>"))
        shown.add(event.kind)
    fig.update_layout(title="Execution timeline", height=40 + 28 * len(finished),
                      barmode="overlay", hovermode="closest")
    fig.update_xaxes(title_text="ms since start")
    fig.update_yaxes(title_text=None, autorange="reversed", automargin=True)  # labels need room on the left
    return apply_theme(fig)
