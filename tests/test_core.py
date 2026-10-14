from datetime import date

import pandas as pd
import pytest

from signalscope.config import Settings
from signalscope.sources.base import Window, slug, split_window, to_daily_series
from signalscope.trace import Tracer


def test_window_last_n_days_ends_yesterday():
    w = Window.last_n_days(30, today=date(2026, 10, 14))
    assert w.end == date(2026, 10, 13)
    assert w.start == date(2026, 9, 14)
    assert w.days == 30


def test_to_daily_series_normalizes():
    raw = pd.Series([3.0, None, 1.0, 2.0],
                    index=["2026-10-03T00:00", "2026-10-02", "2026-10-01", "2026-10-03T00:00"])
    s = to_daily_series(raw, "x")
    assert list(s.index) == [pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-03")]
    assert s.loc["2026-10-03"] == 2.0          # duplicate date: last value kept
    assert s.dtype == float and s.index.name == "date" and s.name == "x"


def test_split_window():
    s = pd.Series(range(10), index=pd.date_range("2026-10-01", periods=10), dtype=float)
    history, future = split_window(s, Window(date(2026, 10, 3), date(2026, 10, 7)))
    assert len(history) == 5
    assert future is not None and future.index.min() == pd.Timestamp("2026-10-08")
    _, none_future = split_window(s, Window(date(2026, 10, 1), date(2026, 10, 10)))
    assert none_future is None


def test_slug():
    assert slug("São Paulo") == "s_o_paulo"


def test_tracer_step_and_totals():
    seen = []
    tracer = Tracer(on_event=lambda e: seen.append((e.message, e.status)))
    with tracer.step("api", "Calling weather API", cache_hit=True):
        pass
    llm = tracer.start("llm", "Planning")
    tracer.end(llm, prompt_tokens=100, completion_tokens=20)
    with pytest.raises(ValueError):
        with tracer.step("analysis", "Fitting"):
            raise ValueError("boom")
    totals = tracer.totals()
    assert totals["api_calls"] == 1 and totals["cache_hits"] == 1
    assert totals["prompt_tokens"] == 100 and totals["completion_tokens"] == 20
    assert tracer.events[-1].status == "error"
    assert seen[0] == ("Calling weather API", "running")


def test_settings_cloud_enabled():
    assert not Settings(openrouter_api_key=None).cloud_enabled
    assert Settings(openrouter_api_key="k").cloud_enabled
    assert not Settings(openrouter_api_key="k", llm_mode="local").cloud_enabled
