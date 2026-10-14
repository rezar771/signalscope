"""LLM access: one OpenAI-compatible client for every backend, plus a stateless fallback cascade.

OpenRouter and Ollama (`/v1`) both speak the OpenAI chat-completions format, so a hosted provider
is a new `OpenAICompatClient` configuration, not new code. Each client bounds its own concurrency
with a semaphore: clients are process-wide (shared by all users), so the semaphore is the
admission control for that backend.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import requests

from signalscope.config import LLMMode, Settings

Message = dict[str, Any]


class LLMUnavailable(Exception):
    """Backend cannot answer right now (busy, rate-limited, down, timed out, bad response)."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] | str   # raw; validated by the registry


@dataclass(frozen=True)
class LLMReply:
    content: str
    tool_calls: list[ToolCall]
    backend: str                      # client name that answered, e.g. "openrouter"
    model: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float


class LLMClient(Protocol):
    name: str
    label: str

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> LLMReply: ...

    def health(self) -> bool: ...

    def status(self) -> tuple[bool, str]: ...


@dataclass
class OpenAICompatClient:
    name: str                          # short id, e.g. "openrouter", "ollama"
    label: str                         # for the UI log, e.g. "nemotron-3-super (OpenRouter)"
    base_url: str
    model: str
    api_key: str | None = None
    timeout_s: float = 60.0
    max_concurrency: int = 2
    acquire_timeout_s: float = 30.0
    health_path: str = "/models"
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)   # provider-specific request options

    def __post_init__(self) -> None:
        self._slots = threading.BoundedSemaphore(self.max_concurrency)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> LLMReply:
        if not self._slots.acquire(timeout=self.acquire_timeout_s):
            raise LLMUnavailable(f"{self.label} is busy ({self.max_concurrency} requests in flight)")
        try:
            return self._post(messages, tools)
        finally:
            self._slots.release()

    def _post(self, messages: list[Message], tools: list[dict] | None) -> LLMReply:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0, **self.extra_body}
        if tools:
            body["tools"] = tools
        started = time.perf_counter()
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", json=body,
                                 headers=self._headers(), timeout=self.timeout_s)
        except requests.Timeout as exc:
            raise LLMUnavailable(f"{self.label} timed out after {self.timeout_s:.0f} s") from exc
        except requests.RequestException as exc:
            raise LLMUnavailable(f"{self.label} is unreachable") from exc
        duration_ms = (time.perf_counter() - started) * 1000
        if resp.status_code == 429:
            raise LLMUnavailable(f"{self.label} is rate-limited (HTTP 429)")
        if resp.status_code >= 400:
            raise LLMUnavailable(f"{self.label} returned HTTP {resp.status_code}")
        try:
            data = resp.json()
            if "error" in data:
                raise LLMUnavailable(f"{self.label} error: {str(data['error'])[:120]}")
            message = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable(f"{self.label} returned a malformed response") from exc
        calls = [
            ToolCall(c["function"]["name"], c["function"].get("arguments") or {})
            for c in (message.get("tool_calls") or [])
            if isinstance(c, dict) and isinstance(c.get("function"), dict) and c["function"].get("name")
        ]
        usage = data.get("usage") or {}
        return LLMReply(
            content=(message.get("content") or "").strip(),
            tool_calls=calls,
            backend=self.name,
            model=data.get("model") or self.model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            duration_ms=duration_ms,
        )

    def health(self) -> bool:
        return self.status()[0]

    def status(self) -> tuple[bool, str]:
        """(available, short reason) for the UI. For OpenRouter free models this also reads the
        daily free-request quota: the key can be valid while every call would get HTTP 429."""
        try:
            resp = requests.get(f"{self.base_url}{self.health_path}", headers=self._headers(), timeout=3)
        except requests.RequestException:
            return False, "unreachable"
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        try:
            data = resp.json()
        except ValueError:
            return False, "malformed response"
        if self.name == "ollama":  # the server can be up without the model being pulled
            if not any(m.get("id") == self.model for m in data.get("data", [])):
                return False, f"model not pulled (ollama pull {self.model})"
            return True, "available"
        quota = (data.get("data") or {}).get("free_model_daily_requests")
        if self.model.endswith(":free") and isinstance(quota, dict):
            if quota.get("remaining", 1) <= 0:
                return False, (f"daily free limit reached ({quota.get('used')}/{quota.get('limit')}), "
                               "resets 00:00 UTC")
            return True, f"available ({quota.get('remaining')} free requests left today)"
        return True, "available"


FallbackHook = Callable[[LLMClient, LLMClient, str], None]


class CascadeLLM:
    """Try clients in order; move to the next one only when a client is unavailable.

    Stateless per call: concurrent users never see each other's fallback decisions.
    """

    def __init__(self, clients: list[LLMClient]) -> None:
        if not clients:
            raise ValueError("CascadeLLM needs at least one client")
        self.clients = clients

    def index_of(self, backend: str) -> int:
        return next(i for i, c in enumerate(self.clients) if c.name == backend)

    def chat(self, messages: list[Message], tools: list[dict] | None = None, start: int = 0,
             on_fallback: FallbackHook | None = None) -> LLMReply:
        reasons: list[str] = []
        candidates = self.clients[start:]
        for i, client in enumerate(candidates):
            try:
                return client.chat(messages, tools)
            except LLMUnavailable as exc:
                reasons.append(str(exc))
                if on_fallback is not None and i + 1 < len(candidates):
                    on_fallback(client, candidates[i + 1], str(exc))
        raise LLMUnavailable("No language model available: " + "; ".join(reasons))


def build_llm(settings: Settings, mode: LLMMode | None = None) -> CascadeLLM:
    """auto = OpenRouter then local Ollama (local only if no key); cloud = OpenRouter; local = Ollama."""
    mode = mode or settings.llm_mode
    cloud = OpenAICompatClient(
        name="openrouter",
        label=f"{settings.openrouter_model.split('/')[-1].removesuffix(':free')} (OpenRouter)",
        base_url=settings.openrouter_base_url,
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        timeout_s=settings.llm_timeout_s,
        max_concurrency=4,
        acquire_timeout_s=2.0,          # busy cloud → fall back quickly instead of queueing
        health_path="/key",             # validates the key, unlike the public /models list
        extra_headers={"X-Title": "SignalScope"},
        # nemotron is a reasoning model: hidden "thinking" made one summary take 60 s+. Measured
        # 2026-10-14: summary 5.5–13.5 s → 1.3–1.7 s with reasoning disabled, same quality.
        extra_body={"reasoning": {"enabled": False}},
    )
    local = OpenAICompatClient(
        name="ollama",
        label=f"{settings.ollama_model} (local)",
        base_url=settings.ollama_base_url.rstrip("/"),
        model=settings.ollama_model,
        timeout_s=settings.llm_timeout_s,
        max_concurrency=2,
        acquire_timeout_s=30.0,         # last resort: wait, then report "busy" instead of hanging
    )
    if mode == "local":
        return CascadeLLM([local])
    if not settings.openrouter_api_key:
        if mode == "cloud":
            raise ValueError("LLM_MODE=cloud requires OPENROUTER_API_KEY")
        return CascadeLLM([local])
    return CascadeLLM([cloud] if mode == "cloud" else [cloud, local])


def parse_arguments(raw: dict[str, Any] | str) -> dict[str, Any] | str:
    """Best-effort JSON decode for display; validation proper happens in the registry."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw
