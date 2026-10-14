"""Execution trace: timed, typed events that drive the UI log, the timeline chart and the cost numbers.

The tracer is deliberately single-threaded: only the orchestrator's main thread touches it,
so Streamlit callbacks stay safe. Parallel work reports back to the main thread first.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal

EventKind = Literal["llm", "api", "analysis", "system"]
EventStatus = Literal["running", "ok", "warning", "error"]


@dataclass
class TraceEvent:
    kind: EventKind
    message: str
    status: EventStatus
    start_ms: float                     # ms since the tracer was created
    duration_ms: float | None = None    # None while running and for instant events
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.status != "running"


class Tracer:
    def __init__(self, on_event: Callable[[TraceEvent], None] | None = None) -> None:
        self._t0 = time.perf_counter()
        self._on_event = on_event
        self.events: list[TraceEvent] = []

    def _now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def _emit(self, event: TraceEvent) -> None:
        if self._on_event is not None:
            self._on_event(event)

    def start(self, kind: EventKind, message: str, **details: Any) -> TraceEvent:
        event = TraceEvent(kind, message, "running", self._now_ms(), details=dict(details))
        self.events.append(event)
        self._emit(event)
        return event

    def end(self, event: TraceEvent, status: EventStatus = "ok", message: str | None = None,
            **details: Any) -> TraceEvent:
        event.duration_ms = self._now_ms() - event.start_ms
        event.status = status
        if message is not None:
            event.message = message
        event.details.update(details)
        self._emit(event)
        return event

    def info(self, kind: EventKind, message: str, status: EventStatus = "ok", **details: Any) -> TraceEvent:
        """Instant event (no duration), e.g. a fallback notice."""
        event = TraceEvent(kind, message, status, self._now_ms(), details=dict(details))
        self.events.append(event)
        self._emit(event)
        return event

    @contextmanager
    def step(self, kind: EventKind, message: str, **details: Any) -> Iterator[TraceEvent]:
        """Time a block; marks the event as error and re-raises if the block fails."""
        event = self.start(kind, message, **details)
        try:
            yield event
        except Exception as exc:
            self.end(event, "error", error=str(exc))
            raise
        if not event.finished:
            self.end(event)

    def totals(self) -> dict[str, float | int]:
        done = [e for e in self.events if e.duration_ms is not None]

        def ms(kind: str) -> float:
            return round(sum(e.duration_ms for e in done if e.kind == kind), 1)

        api = [e for e in done if e.kind == "api"]
        llm = [e for e in done if e.kind == "llm"]
        return {
            "total_ms": round(self._now_ms(), 1),
            "llm_ms": ms("llm"),
            "api_ms": ms("api"),
            "analysis_ms": ms("analysis"),
            "llm_calls": len(llm),
            "prompt_tokens": sum(int(e.details.get("prompt_tokens", 0)) for e in llm),
            "completion_tokens": sum(int(e.details.get("completion_tokens", 0)) for e in llm),
            "api_calls": len(api),
            "cache_hits": sum(1 for e in api if e.details.get("cache_hit")),
        }
