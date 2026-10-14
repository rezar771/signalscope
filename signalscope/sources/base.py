"""Data contracts shared by every data source."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd


@dataclass(frozen=True)
class Window:
    """Inclusive daily analysis window; `end` is normally yesterday (last complete day)."""

    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    @classmethod
    def last_n_days(cls, n: int, today: date | None = None) -> "Window":
        end = (today or date.today()) - timedelta(days=1)
        return cls(start=end - timedelta(days=n - 1), end=end)


@dataclass(frozen=True)
class TimeSeries:
    key: str
    label: str
    unit: str
    source: str
    url: str
    license: str
    history: pd.Series                  # daily DatetimeIndex, tz-naive, normalized, sorted, unique; float; no NaN
    future: pd.Series | None = None     # values for dates after window.end, same format
    exogenous: bool = False             # natural driver (weather, air quality)


@dataclass(frozen=True)
class FetchResult:
    series: TimeSeries
    cache_hit: bool
    http_calls: int


class SourceError(Exception):
    """User-facing error: the message must be readable by an end user and by the LLM."""


class ToolCallError(ValueError):
    """Invalid tool call from the LLM; the message is fed back to it in the repair round,
    so it must say exactly what was wrong and what is allowed."""


MIN_POINTS = 5


def to_daily_series(values: pd.Series, name: str) -> pd.Series:
    """Normalize to the TimeSeries.history format: tz-naive midnight index, float, sorted, no NaN/dupes."""
    s = pd.to_numeric(values, errors="coerce").astype(float)
    idx = s.index if isinstance(s.index, pd.DatetimeIndex) else pd.DatetimeIndex(pd.to_datetime(s.index, format="ISO8601"))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    s.index.name = "date"
    s.name = name
    return s


def split_window(s: pd.Series, window: Window) -> tuple[pd.Series, pd.Series | None]:
    """Split a daily series into (history inside window, future after window.end or None)."""
    start, end = pd.Timestamp(window.start), pd.Timestamp(window.end)
    history = s[(s.index >= start) & (s.index <= end)]
    future = s[s.index > end]
    return history, (future if not future.empty else None)


def slug(text: str) -> str:
    """'São Paulo' -> 's_o_paulo'; used for TimeSeries.key."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def require_points(history: pd.Series, what: str) -> None:
    if len(history) < MIN_POINTS:
        raise SourceError(f"Not enough data for {what}: only {len(history)} days available in the window.")
