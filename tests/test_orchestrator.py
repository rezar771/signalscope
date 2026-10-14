"""Orchestrator tests: real registry validation, fake fetchers, fake analysis, scripted LLM."""

from __future__ import annotations

import dataclasses
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from signalscope.agent import orchestrator
from signalscope.agent.llm import CascadeLLM
from signalscope.agent.orchestrator import run
from signalscope.sources.base import FetchResult, SourceError, TimeSeries
from signalscope.sources.registry import TOOLS
from test_agent import FACTS, GOOD, FakeClient, reply

WIND = ("get_weather", {"city": "Milan", "variable": "wind"})          # alias, normalized by registry
PM10 = ("get_air_quality", {"city": "Milan", "pollutant": "pm10"})
QUESTION = "Does wind reduce PM10 pollution in Milan?"


def series(key: str, exogenous: bool) -> TimeSeries:
    idx = pd.date_range("2026-07-16", periods=90, name="date")
    return TimeSeries(key, key.upper(), "u", "Fake", "https://example", "CC0",
                      pd.Series(np.arange(90, dtype=float), index=idx, name=key), exogenous=exogenous)


def fake_tools(fail: str | None = None, calls: list | None = None):
    def make(name: str, key: str):
        def fetch(args, window, session, settings):
            if calls is not None:
                calls.append(name)
            if name == fail:
                raise SourceError(f"{name} is down.")
            return FetchResult(series(key, name == "get_weather"), cache_hit=False, http_calls=2)
        return dataclasses.replace(TOOLS[name], fetch=fetch)
    return {**TOOLS, "get_weather": make("get_weather", "wind"), "get_air_quality": make("get_air_quality", "pm10")}


def fake_analyze(driver, target):
    facts = {**FACTS, "driver": driver.label, "target": target.label}
    model = SimpleNamespace(skill=0.2, n_fit=90, forecast=None)
    return SimpleNamespace(facts=lambda: facts, model=model, driver=driver, target=target)


@pytest.fixture(autouse=True)
def clear_caches():
    orchestrator.PLAN_CACHE.clear()
    orchestrator.SUMMARY_CACHE.clear()


def llm_for(*replies) -> tuple[CascadeLLM, FakeClient]:
    client = FakeClient("c", list(replies))
    return CascadeLLM([client]), client


def test_happy_path_picks_weather_as_driver_even_if_called_second(settings):
    llm, _ = llm_for(reply("c", [PM10, WIND]), reply("c", content=GOOD.replace("Pearson", "Pearson")))
    result = run(QUESTION, settings, llm, tools=fake_tools(), analyze=fake_analyze)
    assert result.ok, [e.message for e in result.trace.events]
    assert result.driver.key == "wind" and result.target.key == "pm10"
    assert result.plan.calls[1].args.variable == "wind_speed"          # alias normalized
    kinds = [e.kind for e in result.trace.events if e.duration_ms is not None]
    assert kinds.count("api") == 2 and "analysis" in kinds
    assert result.summary is not None


def test_repeated_question_uses_plan_and_summary_cache(settings):
    llm, client = llm_for(reply("c", [WIND, PM10]), reply("c", content=GOOD))
    first = run(QUESTION, settings, llm, tools=fake_tools(), analyze=fake_analyze)
    second = run("  does wind reduce pm10 pollution in milan ", settings, llm, tools=fake_tools(),
                 analyze=fake_analyze)
    assert first.ok and second.ok
    assert len(client.received) == 2                                     # no LLM calls on the repeat
    assert second.trace.totals()["llm_calls"] == 0
    assert any("cache" in e.message for e in second.trace.events)


def test_fetch_failure_is_reported_not_raised(settings):
    llm, _ = llm_for(reply("c", [WIND, PM10]))
    result = run(QUESTION, settings, llm, tools=fake_tools(fail="get_air_quality"), analyze=fake_analyze)
    assert not result.ok and "get_air_quality is down" in result.error
    assert any(e.status == "error" for e in result.trace.events)


def test_planning_failure_returns_suggestions(settings):
    llm, _ = llm_for(reply("c", content="hello"), reply("c", content="still text"))
    result = run(QUESTION, settings, llm, tools=fake_tools(), analyze=fake_analyze)
    assert result.error and result.suggestions and result.report is None


@pytest.mark.parametrize("question,message", [("   ", "ask a question"), ("x" * 301, "under 300")])
def test_question_validation(settings, question, message):
    result = run(question, settings, CascadeLLM([FakeClient("c", [])]))
    assert message in result.error


def test_unexpected_analysis_bug_never_raises(settings):
    def broken(driver, target):
        raise ZeroDivisionError("bug")
    llm, _ = llm_for(reply("c", [WIND, PM10]))
    result = run(QUESTION, settings, llm, tools=fake_tools(), analyze=broken)
    assert result.error and "ZeroDivisionError" in result.trace.events[-1].message


def test_concurrent_identical_runs_share_one_fetch(settings):
    calls: list[str] = []
    gate = threading.Barrier(3)

    def slow_tools():
        tools = fake_tools(calls=calls)
        def slow(tool):
            def fetch(*a):
                import time
                time.sleep(0.2)
                return tool.fetch(*a)
            return dataclasses.replace(tool, fetch=fetch)
        return {**tools, "get_weather": slow(tools["get_weather"]), "get_air_quality": slow(tools["get_air_quality"])}

    tools = slow_tools()
    orchestrator.PLAN_CACHE.clear()
    results = []

    class RoleClient(FakeClient):
        # Answers by request type, not by script order: a user that reuses another user's plan
        # skips the planning call, so a scripted order would hand it the wrong reply.
        def chat(self, messages, tools=None):
            self.received.append(messages)
            return reply("c", [WIND, PM10]) if tools else reply("c", content=GOOD)

    def worker():
        llm = CascadeLLM([RoleClient("c", [])])
        gate.wait()
        results.append(run(QUESTION, settings, llm, tools=tools, analyze=fake_analyze))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r.ok for r in results)
    assert sorted(calls) == ["get_air_quality", "get_weather"]          # 3 users, 1 fetch per dataset
    assert sum(r.trace.totals()["llm_calls"] for r in results) == 2      # not 6: plan/summary shared
