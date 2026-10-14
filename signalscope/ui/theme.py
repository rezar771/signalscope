"""Shared dark UI styling and safe HTML fragments for SignalScope."""

from __future__ import annotations

from html import escape

import streamlit as st

from signalscope.ui.charts import AXIS_TEXT, BRAND_RED, KIND_COLORS, NEUTRAL, TEXT, VERDICT_COLORS

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {{ color-scheme: dark; }}
html, body, [class*="css"] {{ font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.stApp {{ background: #0b1220; color: {TEXT}; }}
[data-testid="stMainBlockContainer"], .block-container {{ max-width: 1400px; padding-top: 2.25rem !important; }}
[data-testid="stHeader"] {{ background: transparent; }}
/* Hide only the menu and deploy button: the sidebar's expand button lives in the same toolbar. */
#MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stAppDeployButton"] {{ display: none; }}
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] {{ visibility: visible !important; display: flex !important; }}
[data-testid="stMetric"], [data-testid="stPlotlyChart"], [data-testid="stExpander"] {{
  background: #111a2e; border: 1px solid #1e293b; border-radius: 14px; padding: 14px 18px;
  box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.25);
}}
[data-testid="stMetricLabel"] {{ color: {AXIS_TEXT}; text-transform: uppercase; letter-spacing: .06em; font-size: .72rem; }}
[data-testid="stMetricValue"] {{ color: {TEXT}; font-size: 1.45rem; font-weight: 600; }}
/* Stat cards: a grid that adapts to the column width; labels and values never break mid-word. */
.ss-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(96px, 1fr)); gap: .6rem; margin: .75rem 0 1rem; }}
.ss-stat {{ background: #111a2e; border: 1px solid #1e293b; border-radius: 14px; padding: .8rem .95rem;
  box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.25); min-width: 0; }}
.ss-stat-label {{ color: {AXIS_TEXT}; font-size: .68rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: help; }}
.ss-stat-value {{ color: {TEXT}; font-size: clamp(1.15rem, 1.6vw, 1.5rem); font-weight: 650; white-space: nowrap; margin-top: .2rem; }}
.ss-stat-sub {{ color: #64748b; font-size: .72rem; margin-top: .1rem; white-space: nowrap; }}
section[data-testid="stSidebar"] [data-testid="stExpander"] {{ background: transparent; border: none; box-shadow: none; padding: 0; }}
.ss-badge {{ margin-top: .75rem; }}
button[kind="primary"] {{ background: {BRAND_RED}; border-color: {BRAND_RED}; }}
button:focus, input:focus {{ outline-color: {BRAND_RED} !important; }}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ border-bottom-color: {BRAND_RED}; color: {TEXT}; }}
.ss-hero {{ position: relative; overflow: hidden; padding: 1.5rem 1.25rem 1.6rem; margin-bottom: 1rem; border-radius: 16px;
  background: radial-gradient(600px circle at 0% 0%, rgba(224,79,78,0.12), transparent 60%); }}
.ss-caption {{ color: {AXIS_TEXT}; text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; font-weight: 600; }}
.ss-title {{ margin: .45rem 0 .35rem; color: {TEXT}; font-size: 2.4rem; line-height: 1.15; font-weight: 700; }}
.ss-title-mark {{ width: 48px; height: 4px; margin-top: .6rem; border-radius: 4px; background: {BRAND_RED}; }}
.ss-subtitle {{ color: {AXIS_TEXT}; font-size: 1rem; max-width: 780px; }}
.ss-verdict {{ display: inline-flex; align-items: center; border-radius: 999px; padding: .32rem .72rem; font-size: .8rem; font-weight: 600; margin-right: .65rem; }}
.ss-verdict-row {{ display: flex; align-items: center; gap: .3rem; margin: .3rem 0 1rem; }}
.ss-reason {{ color: {AXIS_TEXT}; }}
.ss-log {{ display: flex; align-items: center; gap: .55rem; padding: .48rem .25rem; border-bottom: 1px solid #1e293b; font: .82rem 'JetBrains Mono', monospace; }}
.ss-log-message {{ flex: 1; color: {TEXT}; overflow-wrap: anywhere; }}
.ss-log-duration {{ color: {NEUTRAL}; white-space: nowrap; }}
.ss-log-running {{ color: {BRAND_RED}; animation: ss-pulse 1.2s ease-in-out infinite; }}
.ss-log-error {{ color: #f87171; }} .ss-log-warning {{ color: #fbbf24; }}
.ss-cache {{ color: {KIND_COLORS['api']}; border: 1px solid {KIND_COLORS['api']}; border-radius: 999px; padding: .05rem .42rem; font: .68rem Inter, sans-serif; }}
.ss-health-ok {{ color: #34d399; }} .ss-health-down {{ color: #f87171; }}
.ss-ai-card {{ background: #111a2e; border: 1px solid #1e293b; border-left: 3px solid #a78bfa; border-radius: 12px; padding: 1rem 1.1rem; }}
.ss-ai-label {{ color: #a78bfa; font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }}
.ss-badge {{ display: inline-block; border-radius: 999px; padding: .18rem .55rem; font-size: .75rem; font-weight: 600; }}
@keyframes ss-pulse {{ 50% {{ opacity: .35; }} }}
"""


def inject_css() -> None:
    """Install the shared application styles."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> str:
    """Return the branded header fragment."""
    return (f'<section class="ss-hero">'
            f'<div class="ss-title">{escape(title)}</div><div class="ss-title-mark"></div>'
            f'<div class="ss-subtitle">{escape(subtitle)}</div></section>')


def _rgba(color: str, alpha: float) -> str:
    value = color.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def verdict_pill(verdict: str, reason: str) -> str:
    """Return a verdict and explanatory reason with the shared verdict palette."""
    color = VERDICT_COLORS.get(verdict, NEUTRAL)
    label = escape(verdict.title())
    return (f'<div class="ss-verdict-row"><span class="ss-verdict" style="color:{color};background:{_rgba(color, .15)};'
            f'border:1px solid {color};box-shadow:0 0 24px {_rgba(color, .2)}">{label}</span>'
            f'<span class="ss-reason">{escape(reason)}</span></div>')


def log_line(event) -> str:
    """Render one trace event as a compact, escaped execution-log row."""
    icons = {"llm": "🧠", "api": "🌐", "analysis": "📊", "system": "⚙️"}
    status_class = f" ss-log-{event.status}" if event.status in {"running", "warning", "error"} else ""
    message = event.message
    if event.kind == "llm" and event.details.get("model"):
        message += f" · {event.details['model']}"
    duration = "—" if event.duration_ms is None else f"{event.duration_ms:.0f} ms"
    cached = '<span class="ss-cache">cached</span>' if event.details.get("cache_hit") else ""
    return (f'<div class="ss-log{status_class}"><span>{icons.get(event.kind, "•")}</span>'
            f'<span class="ss-log-message">{escape(message)}</span>{cached}'
            f'<span class="ss-log-duration">{duration}</span></div>')


def badge(text: str, tone: str = "neutral") -> str:
    """Return a small label badge; known verdict tones use the chart palette."""
    color = VERDICT_COLORS.get(tone, AXIS_TEXT if tone == "neutral" else KIND_COLORS.get(tone, NEUTRAL))
    return (f'<span class="ss-badge" style="color:{color};background:{_rgba(color, .15)};'
            f'border:1px solid {color}">{escape(text)}</span>')


def stat_cards(items: list[tuple[str, str, str]]) -> str:
    """Grid of stat cards from (label, value, tooltip) tuples; the tooltip is the card's title."""
    cards = "".join(
        f'<div class="ss-stat" title="{escape(tip)}"><div class="ss-stat-label">{escape(label)}</div>'
        f'<div class="ss-stat-value">{escape(value)}</div></div>'
        for label, value, tip in items
    )
    return f'<div class="ss-stats">{cards}</div>'
