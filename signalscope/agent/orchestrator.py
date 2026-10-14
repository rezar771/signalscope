"""One user question → one RunResult. The only entry point the UI (or an API) needs.

Flow: validate → window (code) → plan (LLM, cached) → parallel fetch (single-flight, per-task
HTTP session) → driver/target choice (code) → analysis (code) → summary (LLM, cached, verified).
`run` never raises: failures end up in `RunResult.error` and in the trace.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from signalscope.agent.cache import TTLCache, normalize_question
from signalscope.agent.llm import CascadeLLM
from signalscope.agent.planner import Plan, PlannedCall, PlanningError, plan as make_plan
from signalscope.agent.singleflight import FETCHES, SingleFlight
from signalscope.agent.summarizer import Summary, summarize
from signalscope.agent.window import parse_window
from signalscope.analysis.report import AnalysisReport, analyze as analyze_report
from signalscope.config import Settings
from signalscope.http import make_session
from signalscope.sources.base import FetchResult, SourceError, TimeSeries, Window
from signalscope.sources.registry import TOOLS, Tool, validate_call
from signalscope.trace import TraceEvent, Tracer

MAX_QUESTION_CHARS = 300

PROVIDERS = {
    "get_weather": "Open-Meteo weather API",
    "get_air_quality": "Open-Meteo air-quality API",
    "get_wikipedia_interest": "Wikimedia Pageviews API",
    "get_exchange_rate": "Frankfurter (ECB) API",
    "get_electricity": "Energy-Charts API",
}
# Natural drivers first: "Does wind reduce PM10?" must not make PM10 the cause (measured: the
# model may call the tools in either order).
DRIVER_PRIORITY = {"get_weather": 2, "get_air_quality": 1}

PLAN_CACHE: TTLCache[Plan] = TTLCache(maxsize=512, ttl_s=3600)
SUMMARY_CACHE: TTLCache[Summary] = TTLCache(maxsize=512, ttl_s=3600)
# Identical questions arriving at the same moment share one LLM call (the caches above only help
# once the first answer exists).
PLANS_IN_FLIGHT = SingleFlight()
SUMMARIES_IN_FLIGHT = SingleFlight()

AnalyzeFn = Callable[[TimeSeries, TimeSeries], AnalysisReport]


@dataclass
class RunResult:
    question: str
    trace: Tracer
    window: Window | None = None
    window_note: str | None = None
    plan: Plan | None = None
    driver: TimeSeries | None = None
    target: TimeSeries | None = None
    report: AnalysisReport | None = None
    summary: Summary | None = None
    error: str | None = None
    suggestions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and self.report is not None


def run(question: str, settings: Settings, llm: CascadeLLM | None,
        on_event: Callable[[TraceEvent], None] | None = None, *,
        tools: dict[str, Tool] | None = None, analyze: AnalyzeFn | None = None) -> RunResult:
    """Answer one question. `tools`/`analyze` are injectable for tests."""
    tracer = Tracer(on_event)
    result = RunResult(question=question.strip(), trace=tracer)
    try:
        _run(result, settings, llm, tools or TOOLS, analyze or analyze_report)
    except Exception as exc:  # last line of defence: the UI must always get a result
        tracer.info("system", f"Unexpected error: {type(exc).__name__}: {exc}", status="error")
        result.error = "Something went wrong while answering. The execution log shows where."
    return result


def _run(result: RunResult, settings: Settings, llm: CascadeLLM | None,
         tools: dict[str, Tool], analyze: AnalyzeFn) -> None:
    tracer, question = result.trace, result.question
    if not question:
        result.error = "Please ask a question."
        return
    if len(question) > MAX_QUESTION_CHARS:
        result.error = f"Please keep the question under {MAX_QUESTION_CHARS} characters."
        return
    if llm is None:
        result.error = "No language model is configured."
        return

    # 1. Time window: deterministic.
    window, note = parse_window(question, settings)
    result.window, result.window_note = window, note
    tracer.info("system", f"Time window {window.start} → {window.end} ({window.days} days)"
                + (f". {note}" if note else ""))

    # 2. Plan: LLM chooses the datasets (cached per normalized question).
    plan_key = (normalize_question(question), tuple(tools))
    plan = PLAN_CACHE.get(plan_key)
    if plan is not None:
        tracer.info("llm", f"Plan reused from cache (0 LLM calls): {_describe(plan.calls)}", cached=True)
    else:
        def plan_and_cache() -> Plan:
            # Double-checked: another flight may have finished between our cache miss and here.
            # Writing the cache inside the flight closes the other gap (both races were caught by
            # the concurrency test).
            if (cached := PLAN_CACHE.get(plan_key)) is not None:
                return cached
            new_plan = make_plan(question, llm, tracer, [t.schema() for t in tools.values()], validate_call)
            PLAN_CACHE.set(plan_key, new_plan)
            return new_plan

        try:
            plan, shared = PLANS_IN_FLIGHT.do(plan_key, plan_and_cache)
        except PlanningError as exc:
            result.error, result.suggestions = str(exc), exc.suggestions
            return
        if shared:
            tracer.info("llm", "Plan shared with an identical question in progress (0 LLM calls)", cached=True)
    result.plan = plan
    result.warnings.extend(plan.warnings)
    tracer.info("llm", f"Agent chose: {_describe(plan.calls)}")

    # 3. Fetch both datasets in parallel.
    fetched = _fetch_all(plan.calls, window, settings, tools, tracer)
    if isinstance(fetched, str):
        result.error = fetched
        return

    # 4. Driver/target: deterministic priority, then the model's order.
    ordered = sorted(zip(plan.calls, fetched), key=lambda cf: -DRIVER_PRIORITY.get(cf[0].tool, 0))
    result.driver, result.target = ordered[0][1], ordered[1][1]
    tracer.info("system", f"Testing whether {result.driver.label} (driver) relates to "
                          f"{result.target.label} (target)")

    # 5. Analysis: statistics + model, pure code.
    with tracer.step("analysis", "Aligning series, scanning lags −7…+7 days, computing correlations…") as ev:
        report = analyze(result.driver, result.target)
        facts = report.facts()
        ev.message = (f"Aligned {facts['n']} days · lag {facts['lag_days']} d · Pearson r = "
                      f"{facts['pearson_r']} (p {_p(facts['pearson_p'])}) · verdict: {facts['verdict']}")
    result.report = report
    if report.model is not None:
        model = report.model
        skill = "n/a" if model.skill is None else f"{model.skill:.2f}"
        horizon = 0 if model.forecast is None else len(model.forecast)
        tracer.info("analysis", f"Trained linear regression on {model.n_fit} days "
                                f"(skill vs baseline: {skill}); forecast {horizon} day(s)")

    # 6. Summary: LLM phrasing of computed facts, verified (cached per question + facts).
    summary_key = hashlib.sha256(
        (normalize_question(question) + json.dumps(facts, sort_keys=True, default=str)).encode()
    ).hexdigest()
    summary = SUMMARY_CACHE.get(summary_key)
    if summary is not None:
        tracer.info("llm", "Summary reused from cache (0 LLM calls)", cached=True)
    else:
        def summarize_and_cache() -> Summary:
            if (cached := SUMMARY_CACHE.get(summary_key)) is not None:
                return cached
            new_summary = summarize(question, facts, llm, tracer)
            if new_summary.source == "llm":   # templates are not cached: a later retry may verify
                SUMMARY_CACHE.set(summary_key, new_summary)
            return new_summary

        summary, shared = SUMMARIES_IN_FLIGHT.do(summary_key, summarize_and_cache)
        if shared:
            tracer.info("llm", "Summary shared with an identical question in progress (0 LLM calls)", cached=True)
    result.summary = summary


def _fetch_all(calls: list[PlannedCall], window: Window, settings: Settings,
               tools: dict[str, Tool], tracer: Tracer) -> list[TimeSeries] | str:
    """Fetch in parallel; returns series in call order, or a user-facing error message.

    The tracer is only touched on this (main) thread; workers just return results.
    """
    def task(call: PlannedCall) -> tuple[FetchResult, bool]:
        session = make_session(settings)          # one session per worker thread
        tool = tools[call.tool]
        return FETCHES.do(call.cache_key(window), lambda: tool.fetch(call.args, window, session, settings))

    events = {}
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = {}
        for i, call in enumerate(calls):
            events[i] = tracer.start("api", f"Calling {PROVIDERS.get(call.tool, call.tool)} "
                                            f"({_args_text(call)})…", tool=call.tool)
            futures[pool.submit(task, call)] = i
        series: dict[int, TimeSeries] = {}
        errors: list[str] = []
        for future in as_completed(futures):
            i = futures[future]
            try:
                fetched, shared = future.result()
            except SourceError as exc:
                tracer.end(events[i], "error", message=str(exc))
                errors.append(str(exc))
                continue
            s = fetched.series
            source = "shared in-flight request" if shared else ("cache" if fetched.cache_hit else "network")
            tracer.end(events[i], message=f"{s.label}: {len(s.history)} days from {source}",
                       rows=len(s.history), cache_hit=fetched.cache_hit or shared, shared=shared,
                       http_calls=0 if (shared or fetched.cache_hit) else fetched.http_calls)
            series[i] = s
    if errors:
        return "Could not fetch the data: " + " ".join(errors)
    return [series[i] for i in range(len(calls))]


def _describe(calls: list[PlannedCall]) -> str:
    return " + ".join(c.describe() for c in calls)


def _args_text(call: PlannedCall) -> str:
    return ", ".join(f"{k}={v}" for k, v in call.args.model_dump().items())


def _p(p: Any) -> str:
    return str(p) if isinstance(p, str) else f"= {p}"
