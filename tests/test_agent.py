"""Agent tests with scripted fake LLM clients (no network)."""

from __future__ import annotations

import threading
import time
from datetime import date
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from signalscope.agent.cache import TTLCache, normalize_question
from signalscope.agent.llm import (
    CascadeLLM, LLMReply, LLMUnavailable, OpenAICompatClient, ToolCall, build_llm,
)
from signalscope.agent.planner import PlanningError, plan
from signalscope.agent.singleflight import SingleFlight
from signalscope.agent.summarizer import check_summary, summarize, template_summary
from signalscope.agent.window import parse_window
from signalscope.config import Settings
from signalscope.sources.base import ToolCallError
from signalscope.trace import Tracer

TODAY = date(2026, 10, 14)


# --- fakes --------------------------------------------------------------------------------------

class FakeClient:
    """Scripted client: each chat() pops the next item; an Exception item is raised."""

    def __init__(self, name: str, script: list[Any]) -> None:
        self.name, self.label = name, f"{name}-model"
        self.script = list(script)
        self.received: list[list[dict]] = []

    def chat(self, messages, tools=None) -> LLMReply:
        self.received.append(messages)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def health(self) -> bool:
        return True


def reply(backend: str, calls: list[tuple[str, dict]] = (), content: str = "") -> LLMReply:
    return LLMReply(content, [ToolCall(n, a) for n, a in calls], backend, f"{backend}-model", 100, 20, 5.0)


class CityArgs(BaseModel):
    city: str
    variable: Literal["temperature", "precipitation"]


class TopicArgs(BaseModel):
    topic: str


def fake_validate(name: str, raw: Any) -> BaseModel:
    models = {"get_weather": CityArgs, "get_wikipedia_interest": TopicArgs}
    if name not in models:
        raise ToolCallError(f"Unknown tool '{name}'.")
    try:
        return models[name](**raw)
    except Exception as exc:
        raise ToolCallError(f"invalid arguments: {exc}") from exc


WEATHER = ("get_weather", {"city": "Rome", "variable": "temperature"})
WIKI = ("get_wikipedia_interest", {"topic": "Ice cream"})
SCHEMAS: list[dict] = []


# --- window -------------------------------------------------------------------------------------

@pytest.mark.parametrize("question,days,has_note", [
    ("Temperature vs ice cream in Rome over the last 60 days?", 60, False),
    ("past 3 weeks of wind and PM10 in Milan", 21, False),
    ("last two months of solar output", 60, False),
    ("this quarter", 90, False),
    ("Is there a correlation today?", 14, True),
    ("over the last 2 years", 90, True),
    ("Does rain affect umbrellas?", 90, True),
])
def test_parse_window(question, days, has_note):
    window, note = parse_window(question, Settings(), today=TODAY)
    assert window.days == days and window.end == date(2026, 10, 13)
    assert (note is not None) is has_note


# --- cascade ------------------------------------------------------------------------------------

def test_cascade_falls_back_and_reports():
    cloud = FakeClient("cloud", [LLMUnavailable("cloud is rate-limited (HTTP 429)")])
    local = FakeClient("local", [reply("local", content="hi")])
    hops = []
    r = CascadeLLM([cloud, local]).chat([], on_fallback=lambda a, b, why: hops.append((a.name, b.name, why)))
    assert r.backend == "local" and hops == [("cloud", "local", "cloud is rate-limited (HTTP 429)")]


def test_cascade_all_unavailable():
    with pytest.raises(LLMUnavailable, match="No language model available"):
        CascadeLLM([FakeClient("a", [LLMUnavailable("down")])]).chat([])


def test_client_semaphore_rejects_when_busy():
    client = OpenAICompatClient("x", "x", "http://127.0.0.1:9", "m", max_concurrency=1, acquire_timeout_s=0.05)
    client._slots.acquire()  # simulate one request in flight
    with pytest.raises(LLMUnavailable, match="busy"):
        client.chat([{"role": "user", "content": "hi"}])


def test_build_llm_modes():
    assert [c.name for c in build_llm(Settings(openrouter_api_key="k")).clients] == ["openrouter", "ollama"]
    assert [c.name for c in build_llm(Settings(openrouter_api_key=None)).clients] == ["ollama"]
    assert [c.name for c in build_llm(Settings(openrouter_api_key="k"), mode="local").clients] == ["ollama"]
    with pytest.raises(ValueError):
        build_llm(Settings(openrouter_api_key=None), mode="cloud")


# --- planner ------------------------------------------------------------------------------------

def run_plan(*clients: FakeClient):
    tracer = Tracer()
    return plan("Does temperature in Rome affect interest in ice cream?", CascadeLLM(list(clients)),
                tracer, SCHEMAS, fake_validate), tracer


def test_plan_valid_first_try():
    result, tracer = run_plan(FakeClient("cloud", [reply("cloud", [WEATHER, WIKI])]))
    assert [c.tool for c in result.calls] == ["get_weather", "get_wikipedia_interest"]
    assert result.llm_calls == 1 and tracer.totals()["prompt_tokens"] == 100


def test_plan_repair_round_recovers():
    bad = ("get_weather", {"city": "Rome", "variable": "heat"})
    cloud = FakeClient("cloud", [reply("cloud", [bad, WIKI]), reply("cloud", [WEATHER, WIKI])])
    result, _ = run_plan(cloud)
    assert result.llm_calls == 2 and len(result.calls) == 2
    repair_prompt = cloud.received[1][-1]["content"]
    assert "get_weather" in repair_prompt and "Valid so far" in repair_prompt


def test_plan_escalates_to_next_model():
    cloud = FakeClient("cloud", [reply("cloud", [WIKI]), reply("cloud", content="I think...")])
    local = FakeClient("local", [reply("local", [WEATHER, WIKI])])
    result, tracer = run_plan(cloud, local)
    assert result.model == "local-model" and result.llm_calls == 3
    assert any("Escalating" in e.message for e in tracer.events)


def test_plan_rejects_copied_example_and_unrelated_topic():
    copied = ("get_wikipedia_interest", {"topic": "Umbrella"})       # from the prompt examples
    unrelated = ("get_wikipedia_interest", {"topic": "Football"})
    cloud = FakeClient("c", [reply("c", [WEATHER, copied]), reply("c", [WEATHER, unrelated])])
    local = FakeClient("l", [reply("l", [WEATHER, WIKI])])
    result, _ = run_plan(cloud, local)
    assert "examples" in cloud.received[1][-1]["content"]
    assert [c.describe() for c in result.calls][1] == "get_wikipedia_interest(topic='Ice cream')"


def test_plan_dedupes_and_truncates():
    extra = ("get_weather", {"city": "Milan", "variable": "temperature"})
    result, _ = run_plan(FakeClient("c", [reply("c", [WEATHER, WEATHER, WIKI, extra])]))
    assert len(result.calls) == 2 and result.warnings


def test_plan_gives_up_with_suggestions():
    garbage = FakeClient("c", [reply("c", content="hello"), reply("c", [("nope", {})])])
    with pytest.raises(PlanningError) as err:
        run_plan(garbage)
    assert err.value.suggestions


def test_plan_llm_unavailable_is_planning_error():
    with pytest.raises(PlanningError, match="No language model available"):
        run_plan(FakeClient("c", [LLMUnavailable("down")]))


# --- summarizer ---------------------------------------------------------------------------------

FACTS = {
    "driver": "Mean temperature — Rome", "driver_unit": "°C",
    "target": "Wikipedia interest — Ice cream", "target_unit": "views/day",
    "window_start": "2026-07-16", "window_end": "2026-10-13", "n": 90, "n_eff": 31.4,
    "lag_days": 0, "pearson_r": 0.62, "pearson_p": "<0.001", "pearson_ci": [0.47, 0.73],
    "direction": "positive", "spearman_rho": 0.58, "differenced_r": 0.21,
    "verdict": "strong", "verdict_reason": "r = 0.62 with p < 0.001 over 90 days.",
    "slope": 41.2, "model_skill": 0.18, "r2_test": 0.33, "forecast": None,
    "caveats": ["Correlation does not imply causation."],
}
GOOD = ("There is strong evidence of a positive relationship: Pearson r = 0.62 (p < 0.001) over 90 days "
        "from 2026-07-16 to 2026-10-13. Correlation does not imply causation.")


def test_check_summary_accepts_grounded_text():
    assert check_summary(GOOD, FACTS, "question") == []


@pytest.mark.parametrize("text,issue", [
    (GOOD.replace("0.62", "0.71"), "numbers not in the facts"),
    (GOOD.replace("strong", "clear"), "verdict"),
    (GOOD.replace("positive", "negative"), "says negative"),
    (GOOD + " Heat causes people to search for ice cream.", "causal claim"),
    (GOOD + " Temperature has a positive effect on interest in ice cream.", "causal claim"),
])
def test_check_summary_rejects(text, issue):
    assert any(issue in i for i in check_summary(text, FACTS, "question"))


def test_summarize_falls_back_to_template_on_invented_number():
    llm = CascadeLLM([FakeClient("c", [reply("c", content=GOOD.replace("90 days", "120 days"))])])
    s = summarize("q", FACTS, llm, Tracer())
    assert s.source == "template" and s.issues and "0.62" in s.text


def test_summarize_uses_verified_llm_text():
    s = summarize("q", FACTS, CascadeLLM([FakeClient("c", [reply("c", content=GOOD)])]), Tracer())
    assert s.source == "llm" and s.text == GOOD


def test_template_summary_is_self_consistent():
    for verdict in ("strong", "moderate", "weak", "none", "insufficient"):
        facts = {**FACTS, "verdict": verdict}
        assert check_summary(template_summary(facts), facts, "q") == []


# --- concurrency helpers ------------------------------------------------------------------------

def test_singleflight_coalesces_concurrent_calls():
    sf, calls, results = SingleFlight(), [], []

    def slow():
        calls.append(1)
        time.sleep(0.1)
        return "data"

    threads = [threading.Thread(target=lambda: results.append(sf.do("k", slow))) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert sorted(shared for _, shared in results) == [False, True, True, True, True]


def test_singleflight_propagates_errors_and_releases_key():
    sf = SingleFlight()
    with pytest.raises(RuntimeError):
        sf.do("k", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert sf.do("k", lambda: 1) == (1, False)


def test_ttl_cache_lru_and_expiry():
    cache: TTLCache[int] = TTLCache(maxsize=2, ttl_s=0.05)
    cache.set("a", 1); cache.set("b", 2); cache.get("a"); cache.set("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1
    time.sleep(0.06)
    assert cache.get("a") is None


def test_normalize_question():
    assert normalize_question("  Does RAIN   affect umbrellas?? ") == "does rain affect umbrellas"


@pytest.mark.parametrize("quota,ok,text", [
    ({"used": 57, "limit": 50, "remaining": 0}, False, "daily free limit reached (57/50)"),
    ({"used": 2, "limit": 50, "remaining": 48}, True, "48 free requests left today"),
    (None, True, "available"),
])
def test_openrouter_status_reads_free_quota(monkeypatch, quota, ok, text):
    class Resp:
        status_code = 200
        def json(self):
            return {"data": {"free_model_daily_requests": quota}}
    monkeypatch.setattr("signalscope.agent.llm.requests.get", lambda *a, **k: Resp())
    client = OpenAICompatClient("openrouter", "nemotron", "https://x", "nvidia/model:free", api_key="k")
    available, detail = client.status()
    assert available is ok and text in detail and client.health() is ok
