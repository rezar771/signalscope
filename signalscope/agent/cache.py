"""Thread-safe in-memory LRU + TTL cache for LLM outputs (plans and summaries).

A repeated question costs zero LLM calls. Process-local by design for the POC;
in production the same interface is backed by Redis so all replicas share it.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from typing import Generic, Hashable, TypeVar

V = TypeVar("V")


class TTLCache(Generic[V]):
    def __init__(self, maxsize: int = 256, ttl_s: float = 3600.0) -> None:
        self.maxsize, self.ttl_s = maxsize, ttl_s
        self._data: OrderedDict[Hashable, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> V | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            stored_at, value = item
            if time.monotonic() - stored_at > self.ttl_s:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: Hashable, value: V) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


def normalize_question(question: str) -> str:
    """Case/whitespace/trailing-punctuation insensitive cache key."""
    return re.sub(r"\s+", " ", question).strip().lower().rstrip("?!. ")
