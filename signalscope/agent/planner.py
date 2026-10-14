"""LLM tool selection with validation, one repair round and one escalation.

Bounded by design: at most 2 calls per model (initial + one repair), then one escalation to the
next model in the cascade. The LLM chooses *what* to fetch; everything else is code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from signalscope.agent.llm import CascadeLLM, LLMClient, LLMReply, LLMUnavailable, Message
from signalscope.agent.prompts import EXAMPLE_VALUES, PLANNER_SYSTEM, PLANNER_USER, REPAIR_USER
from signalscope.sources.base import ToolCallError, Window
from signalscope.trace import Tracer

Validator = Callable[[str, Any], BaseModel]

SUGGESTIONS = [
    "Is sunshine in Rome related to solar power generation in Italy?",
    "Is electricity demand in Germany related to temperature in Berlin?",
    "Is wind related to PM10 pollution in Milan?",
]


@dataclass(frozen=True)
class PlannedCall:
    tool: str
    args: BaseModel

    def cache_key(self, window: Window) -> tuple:
        return (self.tool, self.args.model_dump_json(), window.start, window.end)

    def describe(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.args.model_dump().items())
        return f"{self.tool}({params})"


@dataclass(frozen=True)
class Plan:
    calls: list[PlannedCall]
    model: str                       # label of the model that produced the accepted plan
    llm_calls: int
    warnings: list[str] = field(default_factory=list)


class PlanningError(Exception):
    def __init__(self, message: str, suggestions: list[str] | None = None) -> None:
        super().__init__(message)
        self.suggestions = suggestions or SUGGESTIONS


def _grounding_error(args: BaseModel, question: str) -> str | None:
    """Catch valid-but-wrong arguments: values copied from the prompt examples, or a Wikipedia
    topic unrelated to the question. Cities are not checked: a country may imply its capital."""
    q = question.lower()
    values = args.model_dump()
    for value in values.values():
        if isinstance(value, str) and value.lower() in EXAMPLE_VALUES and value.lower() not in q:
            return f"'{value}' comes from the examples, not from the question."
    topic = values.get("topic")
    if isinstance(topic, str):
        words = re.findall(r"[a-z]{3,}", topic.lower())
        if words and not any(w[:4] in q for w in words):
            return f"topic '{topic}' is not mentioned in the question; use the phenomenon the user named."
    return None


def _validate_calls(reply: LLMReply, validate: Validator, question: str) -> tuple[list[PlannedCall], list[str]]:
    calls: list[PlannedCall] = []
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for call in reply.tool_calls:
        try:
            args = validate(call.name, call.arguments)
        except ToolCallError as exc:
            errors.append(f"- {call.name}: {exc}")
            continue
        if problem := _grounding_error(args, question):
            errors.append(f"- {call.name}: {problem}")
            continue
        key = (call.name, args.model_dump_json())
        if key not in seen:
            seen.add(key)
            calls.append(PlannedCall(call.name, args))
    if not reply.tool_calls:
        errors.append("- You answered with text instead of calling tools.")
    elif len(calls) == 1 and not errors:
        errors.append("- Only one phenomenon was fetched; the question needs two.")
    return calls, errors


def plan(question: str, llm: CascadeLLM, tracer: Tracer, tool_schemas: list[dict],
         validate: Validator) -> Plan:
    base: list[Message] = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": PLANNER_USER.format(question=question)},
    ]
    llm_calls = 0
    start = 0
    last_errors: list[str] = []

    def on_fallback(failed: LLMClient, nxt: LLMClient, reason: str) -> None:
        tracer.info("llm", f"{reason} → falling back to {nxt.label}", status="warning")

    def ask(messages: list[Message], purpose: str, start_at: int) -> LLMReply:
        nonlocal llm_calls
        event = tracer.start("llm", f"{purpose} with {llm.clients[start_at].label}…")
        try:
            reply = llm.chat(messages, tools=tool_schemas, start=start_at, on_fallback=on_fallback)
        except LLMUnavailable as exc:
            tracer.end(event, "error", message=f"{purpose} failed: {exc}")
            raise PlanningError(str(exc)) from exc
        llm_calls += 1
        label = llm.clients[llm.index_of(reply.backend)].label
        tracer.end(event, message=f"{purpose} with {label}: {len(reply.tool_calls)} tool call(s)",
                   model=reply.model, backend=reply.backend, prompt_tokens=reply.prompt_tokens,
                   completion_tokens=reply.completion_tokens)
        return reply

    while start < len(llm.clients):
        reply = ask(base, "Planning", start)
        calls, errors = _validate_calls(reply, validate, question)
        if len(calls) < 2:
            kept = ("Valid so far: " + ", ".join(c.describe() for c in calls) + "\n") if calls else ""
            tracer.info("llm", f"Plan rejected ({len(calls)} valid call(s)); asking the model to repair it",
                        status="warning", errors=errors)
            repair = base + [
                {"role": "assistant", "content": json.dumps(
                    [{"tool": c.name, "arguments": c.arguments} for c in reply.tool_calls], default=str)},
                {"role": "user", "content": REPAIR_USER.format(errors="\n".join(errors), kept=kept)},
            ]
            answered_by = llm.index_of(reply.backend)
            reply = ask(repair, "Repairing plan", answered_by)
            calls, errors = _validate_calls(reply, validate, question)
        if len(calls) >= 2:
            warnings = []
            if len(calls) > 2:
                warnings.append(f"The model chose {len(calls)} datasets; using the first two.")
            label = llm.clients[llm.index_of(reply.backend)].label
            return Plan(calls[:2], label, llm_calls, warnings)
        last_errors = errors
        start = llm.index_of(reply.backend) + 1
        if start < len(llm.clients):
            tracer.info("llm", f"Escalating to {llm.clients[start].label} after an invalid plan",
                        status="warning")

    raise PlanningError(
        "I could not work out which two datasets answer this question. "
        "Try naming both phenomena and a place explicitly."
        + (" Details: " + " ".join(e.lstrip("- ") for e in last_errors) if last_errors else "")
    )
