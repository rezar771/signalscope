"""Streamlit presentation layer; analysis and orchestration stay in their own modules."""

from __future__ import annotations

import json
from html import escape

import pandas as pd
import streamlit as st

from signalscope.agent.llm import CascadeLLM, build_llm
from signalscope.agent.orchestrator import MAX_QUESTION_CHARS, RunResult, run
from signalscope.agent.planner import SUGGESTIONS
from signalscope.config import Settings
from signalscope.sources.registry import TOOLS
from signalscope.trace import TraceEvent
from signalscope.ui.charts import (
    BRAND_RED, VERDICT_COLORS, forecast_chart, lag_chart, scatter_chart,
    timeseries_chart, trace_chart,
)
from signalscope.ui.theme import badge, hero, inject_css, log_line, verdict_pill, stat_cards

st.set_page_config(page_title="SignalScope", page_icon="📡", layout="wide")
inject_css()

EXTRA_EXAMPLES = [
    "Is temperature in Milan related to interest in ice cream?",
    "Is the euro–dollar exchange rate related to interest in Bitcoin?",
]
EXAMPLES = [*SUGGESTIONS, *EXTRA_EXAMPLES]
EXAMPLE_SOURCES = ["Open-Meteo · Energy-Charts", "Open-Meteo · Energy-Charts",
                   "Open-Meteo · Open-Meteo Air Quality"]
KIND_ICONS = {"llm": "🧠", "api": "🌐", "analysis": "📊", "system": "⚙️"}


@st.cache_resource
def get_settings() -> Settings:
    """Load process-wide configuration once."""
    return Settings.from_env()


@st.cache_resource
def get_llm(mode: str) -> CascadeLLM:
    """Share each mode's LLM clients and concurrency limits across sessions."""
    return build_llm(get_settings(), mode)  # type: ignore[arg-type]


@st.cache_data(ttl=30)
def cached_health(mode: str, key_configured: bool) -> list[tuple[str, str, bool, str]]:
    """Cache provider health checks because Streamlit reruns frequently."""
    try:
        llm = get_llm(mode)
    except ValueError:
        llm = get_llm("local")
    return [(client.label, client.name, *client.status()) for client in llm.clients]


def render_sidebar(settings: Settings) -> tuple[str, CascadeLLM]:
    """Render controls, model status, examples, sources, and AI transparency."""
    st.logo("assets/logo-inspectiondrone.svg", size="large")
    st.sidebar.caption("Open data. Clear evidence. Human judgment.")
    st.sidebar.markdown("### Language models")
    default_index = ["auto", "cloud", "local"].index(settings.llm_mode)
    mode = st.sidebar.radio("LLM mode", ["auto", "cloud", "local"], index=default_index,
                            horizontal=True, help="Overrides LLM_MODE for this browser session.",
                            disabled=st.session_state.running)  # a click would restart the script and cancel the run
    llm = get_llm(mode) if mode != "cloud" or settings.openrouter_api_key else get_llm("local")
    health = cached_health(mode, bool(settings.openrouter_api_key))
    for index, (label, name, online, detail) in enumerate(health):
        role = "primary" if index == 0 else "fallback"
        css_class = "ss-health-ok" if online else "ss-health-down"
        st.sidebar.markdown(
            f'<div class="{css_class}">● {escape(label)} ({escape(role)}) — '
            f'{escape(detail)}</div>', unsafe_allow_html=True)
        if not online and name == "ollama":
            st.sidebar.caption("Start the local model with `ollama serve`.")
    if not settings.openrouter_api_key:
        st.sidebar.caption("Cloud disabled (no OPENROUTER_API_KEY).")
    if mode == "cloud" and not settings.openrouter_api_key:
        st.sidebar.warning("Cloud mode needs OPENROUTER_API_KEY; using the local model for this session.")

    with st.sidebar.expander("Try an example", expanded=False):
        for index, question in enumerate(EXAMPLES):
            if st.button(question, key=f"sidebar_example_{index}", width="stretch"):
                st.session_state.pending = question
                st.rerun()

    with st.sidebar.expander("Data sources"):
        source_info = {
            "get_weather": ("Open-Meteo", "CC BY 4.0"),
            "get_air_quality": ("Open-Meteo Air Quality", "CC BY 4.0"),
            "get_wikipedia_interest": ("Wikimedia Pageviews", "CC0"),
            "get_exchange_rate": ("Frankfurter / ECB", "ECB data, free reuse"),
            "get_electricity": ("Energy-Charts / Fraunhofer ISE", "CC BY 4.0"),
        }
        for name, tool in TOOLS.items():
            provider, license_name = source_info[name]
            st.markdown(f"**{tool.name}** · {provider}  \n{tool.description}  \nLicense: {license_name}")

    with st.sidebar.expander("About & AI transparency"):
        st.write("The plan and summary are produced by an AI model; all numbers come from deterministic statistics. "
                 "Questions stay only in this browser session. No accounts or personal data are collected.")
        if mode == "local" or not settings.openrouter_api_key:
            st.write("The AI model runs on this machine; your question is not sent to any AI provider.")
        else:
            model = settings.openrouter_model
            st.write(f"Your question and aggregate statistics are sent to OpenRouter ({model}); "
                     "a local model is used if it is unavailable.")
        st.write("To fetch data, city names, topics and currency codes are sent to the public data APIs listed above.")
    return mode, llm


def render_log(events: list[TraceEvent], live: bool = False) -> None:
    """Render finished trace rows, and optionally keep one replaceable running row."""
    for event in events:
        if event.finished:
            st.markdown(log_line(event), unsafe_allow_html=True)
    if live:
        st.caption("Each completed step appears above as the run progresses.")


def run_with_live_log(question: str, settings: Settings, llm: CascadeLLM) -> RunResult:
    """Execute one run and update the log as trace events arrive on this script thread."""
    completed = st.container()
    active = st.empty()

    def on_event(event: TraceEvent) -> None:
        if event.finished:
            active.empty()
            with completed:
                st.markdown(log_line(event), unsafe_allow_html=True)
        else:
            active.markdown(log_line(event), unsafe_allow_html=True)

    st.session_state.running = True
    try:
        active.markdown(badge("Running…", "neutral"), unsafe_allow_html=True)
        result = run(question, settings, llm, on_event=on_event)
        seconds = result.trace.totals()["total_ms"] / 1000
        active.markdown(badge(f"✓ Completed in {seconds:.1f} s", "strong") if result.ok
                        else badge(f"Finished with an error after {seconds:.1f} s", "weak"), unsafe_allow_html=True)
        history = st.session_state.history
        history.append(result)
        st.session_state.history = history[-10:]
        return result
    finally:
        st.session_state.running = False


def render_summary(result: RunResult) -> None:
    """Render the checked model summary and any fallback explanation."""
    if result.summary is None:
        return
    summary = result.summary
    label = "AI-generated" if summary.source == "llm" else "Computed summary (template)"
    st.markdown(f'<div class="ss-ai-card"><div class="ss-ai-label">{label}</div>'
                f'<p>{escape(summary.text)}</p></div>', unsafe_allow_html=True)
    if summary.source == "llm":
        st.markdown(badge(f"✓ AI-generated by {summary.model or 'model'}, verified against computed facts", "strong"),
                    unsafe_allow_html=True)
    else:
        st.markdown(badge("Template summary", "neutral"), unsafe_allow_html=True)
        if summary.issues:
            st.caption("LLM text rejected: " + "; ".join(summary.issues))


def render_metrics(result: RunResult) -> None:
    """Show the deterministic report metrics when the report has enough data."""
    report = result.report
    if report is None or report.pearson is None or report.model is None:
        return
    facts = report.facts()
    ci = facts.get("pearson_ci")
    ci_help = f"95% confidence interval: {ci[0]} to {ci[1]}" if ci else "Confidence interval unavailable."
    lag = facts.get("lag_days")
    lag_text = f"{lag:+d} d" if lag else "0 d"
    values = [facts.get("pearson_r"), facts.get("pearson_p"), lag_text,
              facts.get("model_skill"), facts.get("n")]
    labels = ["Pearson r", "p-value", "Lag", "Skill", "Days"]
    help_text = [ci_help, "Autocorrelation-adjusted p-value, corrected for lag search when applicable.",
                 "Positive means the driver leads; negative means the target moves first.",
                 "Model skill: hold-out improvement over a simple historical baseline (> 0 beats the baseline).",
                 "Days analysed: common observed dates used in the report."]
    items = [(label, "—" if value is None else str(value), tip) for label, value, tip in zip(labels, values, help_text)]
    st.markdown(stat_cards(items), unsafe_allow_html=True)


def render_result(result: RunResult) -> None:
    """Render errors or the complete result from the orchestrator contract."""
    if st.button("← New question", key="new_question"):
        st.session_state.home = True
        st.rerun()
    st.caption(result.question)
    if result.error:
        st.error(result.error)
        for suggestion in result.suggestions:
            st.markdown(f"- {suggestion}")
        return
    report = result.report
    if report is None:
        st.info("No analysis report was returned.")
        return
    st.markdown(verdict_pill(report.verdict, report.verdict_reason), unsafe_allow_html=True)
    if result.window_note:
        st.caption(result.window_note)
    render_metrics(result)
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)

    over_time, relationship, lag, forecast, provenance, performance = st.tabs(
        ["Over time", "Relationship", "Lag analysis", "Forecast", "Data & provenance", "Performance"])
    with over_time:
        st.plotly_chart(timeseries_chart(report), width="stretch")
    with relationship:
        st.plotly_chart(scatter_chart(report), width="stretch")
    with lag:
        st.plotly_chart(lag_chart(report), width="stretch")
    with forecast:
        fig = forecast_chart(report)
        if fig is None:
            st.info("No forecast: driver has no future values and lag is 0.")
        else:
            st.plotly_chart(fig, width="stretch")
    with provenance:
        series = [s for s in (result.driver, result.target) if s is not None]
        table = pd.DataFrame([{"Series": s.label, "Source": s.source, "License": s.license,
                               "Rows": len(s.history), "Request URL": s.url} for s in series])
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"Request URL": st.column_config.LinkColumn("Request URL", display_text="Open")})
        data = report.data.rename(columns={"driver": report.driver.label, "target": report.target.label})
        st.download_button("Download aligned data (CSV)", data.to_csv(index_label="date"),
                           file_name="signalscope-data.csv", mime="text/csv", key="download_current")
    with performance:
        render_performance(result)

    render_summary(result)
    if report.caveats:
        st.markdown("**Caveats**")
        for caveat in report.caveats:
            st.markdown(f"- {caveat}")
    with st.expander("Agent plan"):
        if result.plan:
            st.caption(f"Planned by {result.plan.model} · {result.plan.llm_calls} LLM call(s)")
            for call in result.plan.calls:
                st.code(json.dumps({"tool": call.tool, "arguments": call.args.model_dump()}, indent=2), language="json")
            for warning in result.plan.warnings:
                st.warning(warning)


def render_performance(result: RunResult) -> None:
    """Show trace timing, call counts, tokens, cache hits, and timeline."""
    totals = result.trace.totals()
    st.markdown(stat_cards([
        ("Total", f"{totals['total_ms'] / 1000:.1f} s", "Wall time of the whole run"),
        ("LLM", f"{totals['llm_ms'] / 1000:.1f} s", f"{totals['llm_calls']} LLM call(s)"),
        ("Data APIs", f"{totals['api_ms'] / 1000:.1f} s", "Parallel fetches, summed"),
        ("Tokens", f"{totals['prompt_tokens'] + totals['completion_tokens']:,}",
         f"{totals['prompt_tokens']:,} in / {totals['completion_tokens']:,} out"),
        ("Cache hits", f"{totals['cache_hits']}/{totals['api_calls']}", "Data requests served from cache"),
    ]), unsafe_allow_html=True)
    st.plotly_chart(trace_chart(result.trace.events), width="stretch")


def render_history(history: list[RunResult], include_latest: bool = False) -> None:
    """Show earlier browser-session runs in compact expanders."""
    earlier = history if include_latest else history[:-1]
    if not earlier:
        return
    st.markdown("### Previous runs")
    for index, result in enumerate(reversed(earlier)):
        verdict = result.report.verdict if result.report else ("error" if result.error else "pending")
        summary = result.summary.text if result.summary else (result.error or "No summary")
        with st.expander(f"{result.question} · {verdict}", expanded=False):
            st.caption(f"Verdict: {verdict}")
            st.write(summary)


def render_empty_state() -> None:
    """Offer three one-click sample questions before the first run."""
    st.markdown("### Start with an example")
    columns = st.columns(3)
    for index, (column, question) in enumerate(zip(columns, SUGGESTIONS)):
        with column:
            st.markdown(f"#### {['☀️', '⚡', '🌬️'][index]} Example {index + 1}")
            st.write(question)
            st.caption(f"Data: {EXAMPLE_SOURCES[index]}")
            if st.button("Try this question", key=f"main_example_{index}", type="primary"):
                st.session_state.pending = question
                st.rerun()


def main() -> None:
    settings = get_settings()
    mode, llm = render_sidebar(settings)
    st.markdown(hero("SignalScope", "Ask how two real-world phenomena are related. The agent picks open datasets, "
                     "tests the hypothesis and shows its work."), unsafe_allow_html=True)
    pending = st.session_state.pop("pending", None)
    # Inside a container the chat input renders inline (right under the hero) instead of pinned
    # to the bottom of the window.
    with st.container():
        typed = st.chat_input("Ask about a relationship between two real-world phenomena…",
                              max_chars=MAX_QUESTION_CHARS, disabled=st.session_state.running)
    question = pending or typed
    history: list[RunResult] = st.session_state.history
    if question:
        st.session_state.home = False
    if not question and (not history or st.session_state.home):
        render_empty_state()
        render_history(history, include_latest=True)
        return

    result = history[-1] if history else None
    left, right = st.columns([2, 1], gap="large")
    with right:
        st.subheader("Execution Log")
        if question:
            result = run_with_live_log(question, settings, llm)
        elif result:
            render_log(result.trace.events)
    if result:
        with left:
            render_result(result)
    render_history(st.session_state.history)


if "history" not in st.session_state:
    st.session_state.history = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "running" not in st.session_state:
    st.session_state.running = False
if "home" not in st.session_state:
    st.session_state.home = False

main()
