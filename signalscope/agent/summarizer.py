"""Grounded summary: the LLM only phrases facts computed by code, and its text is verified.

Checks: numbers must come from the facts (or the question), the verdict word and the direction
must match, and causal claims are rejected. Any failure, or no LLM at all, falls back to a
deterministic template, and the UI says which one is shown.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from signalscope.agent.llm import CascadeLLM, LLMClient, LLMUnavailable
from signalscope.agent.prompts import SUMMARY_SYSTEM, SUMMARY_USER, VERDICT_WORDS
from signalscope.trace import Tracer


@dataclass(frozen=True)
class Summary:
    text: str
    source: Literal["llm", "template"]
    verified: bool
    model: str | None = None
    issues: list[str] = field(default_factory=list)


_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER = re.compile(r"(?<![\w.])[-−]?\d+(?:[.,]\d+)?")
_NEGATIVE_WORDS = re.compile(r"\b(negative(ly)?|invers(e|ely)|opposite)\b", re.I)
_POSITIVE_WORDS = re.compile(r"\bpositive(ly)?\b", re.I)
_CAUSAL = re.compile(r"\b(causes|caused by|proves?|proof|drives|leads to|results in|effects? on|affects|impacts? on|influences)\b", re.I)
_NEGATION = re.compile(r"\b(not|no|cannot|can't|doesn't|does not|isn't)\b", re.I)


def _to_float(token: str) -> float:
    return float(token.replace("−", "-").replace(",", "."))


def _fact_numbers(value: Any, out: set[float]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        out.add(float(value))
    elif isinstance(value, str):
        for token in _NUMBER.findall(_ISO_DATE.sub(" ", value)):
            out.add(_to_float(token))
        for d in _ISO_DATE.findall(value):           # allow "2026", "07", "16" from dates
            out.update(float(x) for x in d.split("-"))
    elif isinstance(value, dict):
        for v in value.values():
            _fact_numbers(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _fact_numbers(v, out)


def _matches(x: float, allowed: set[float]) -> bool:
    # A fact value or any rounding of it to 0-3 decimals ("0.6234" may be written "0.62").
    # Sign-insensitive: "r = 0.62, negative" is fine; the direction check guards the sign.
    return any(abs(abs(x) - abs(v)) < 1e-9
               for a in allowed for v in (a, *(round(a, d) for d in range(4))))


def check_summary(text: str, facts: dict[str, Any], question: str) -> list[str]:
    issues: list[str] = []
    allowed: set[float] = set()
    _fact_numbers(facts, allowed)
    _fact_numbers(question, allowed)
    cleaned = _ISO_DATE.sub(" ", text)
    invented = sorted({t for t in _NUMBER.findall(cleaned) if not _matches(_to_float(t), allowed)})
    if invented:
        issues.append(f"numbers not in the facts: {', '.join(invented)}")

    verdict = facts.get("verdict")
    word = VERDICT_WORDS.get(verdict, str(verdict))
    if verdict == "none":
        if not re.search(r"\bno\b|\bnone\b|\bnot\b", text, re.I):
            issues.append("verdict 'none' not stated")
    elif not re.search(rf"\b{re.escape(word)}\b", text, re.I):
        issues.append(f"verdict '{word}' not stated")

    direction = facts.get("direction")
    if direction == "positive" and _NEGATIVE_WORDS.search(text):
        issues.append("says negative but the correlation is positive")
    if direction == "negative" and _POSITIVE_WORDS.search(text) and not _NEGATIVE_WORDS.search(text):
        issues.append("says positive but the correlation is negative")

    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if _CAUSAL.search(sentence) and not _NEGATION.search(sentence):
            issues.append(f"causal claim: '{sentence.strip()[:80]}'")
            break
    return issues


def template_summary(facts: dict[str, Any]) -> str:
    """Deterministic fallback, built only from facts."""
    verdict = facts.get("verdict", "none")
    driver, target = facts.get("driver", "the driver"), facts.get("target", "the target")
    if verdict == "insufficient":
        return (f"There is insufficient overlapping data to compare {driver} and {target} "
                f"({facts.get('n', 0)} common days). Try a longer time range or other datasets. "
                "Correlation does not imply causation.")
    lead = {"strong": "There is strong evidence", "moderate": "There is moderate evidence",
            "weak": "There is only weak evidence", "none": "There is no evidence"}[verdict]
    parts = [
        f"{lead} of a relationship between {driver} and {target} over {facts.get('n')} days "
        f"({facts.get('window_start')} to {facts.get('window_end')}).",
        f"The correlation is {facts.get('direction', 'none')} (Pearson r = {facts.get('pearson_r')}, "
        f"p {_p_text(facts.get('pearson_p'))}).",
    ]
    lag = facts.get("lag_days") or 0
    if lag:
        parts.append(f"The relationship is strongest with a lag of {lag} day(s).")
    if facts.get("forecast"):
        last = facts["forecast"][-1]
        parts.append(f"The fitted model forecasts {target} up to {last['date']} "
                     f"(last value {last['yhat']} {facts.get('target_unit', '')}).")
    parts.append("Correlation does not imply causation.")
    return " ".join(parts)


def _p_text(p: Any) -> str:
    return f"{p}" if isinstance(p, str) and p.startswith("<") else f"= {p}"


def summarize(question: str, facts: dict[str, Any], llm: CascadeLLM | None, tracer: Tracer) -> Summary:
    if llm is None or facts.get("verdict") == "insufficient":
        return Summary(template_summary(facts), "template", verified=True)

    word = VERDICT_WORDS.get(facts.get("verdict"), "no")
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM.format(verdict_word=word)},
        {"role": "user", "content": SUMMARY_USER.format(
            question=question, facts=json.dumps(facts, ensure_ascii=False, indent=1))},
    ]

    def on_fallback(failed: LLMClient, nxt: LLMClient, reason: str) -> None:
        tracer.info("llm", f"{reason} → falling back to {nxt.label}", status="warning")

    event = tracer.start("llm", f"Writing summary with {llm.clients[0].label}…")
    try:
        reply = llm.chat(messages, on_fallback=on_fallback)
    except LLMUnavailable as exc:
        tracer.end(event, "warning", message=f"Summary model unavailable ({exc}); using template")
        return Summary(template_summary(facts), "template", verified=True, issues=[str(exc)])

    label = llm.clients[llm.index_of(reply.backend)].label
    issues = check_summary(reply.content, facts, question) if reply.content else ["empty response"]
    tracer.end(event, "ok" if not issues else "warning",
               message=f"Summary by {label}: " + ("verified against facts" if not issues
                                                   else "failed verification, using template"),
               model=reply.model, backend=reply.backend, prompt_tokens=reply.prompt_tokens,
               completion_tokens=reply.completion_tokens, issues=issues)
    if issues:
        return Summary(template_summary(facts), "template", verified=True, model=label, issues=issues)
    return Summary(reply.content, "llm", verified=True, model=label)
