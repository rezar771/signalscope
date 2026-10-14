"""Request coalescing: concurrent callers with the same key share one execution.

Prevents a cache stampede when several users ask for the same data at the same time:
the first caller fetches, the others wait for its result instead of hitting the provider.
In production this becomes a Redis lock shared across replicas.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Callable, Hashable, TypeVar

T = TypeVar("T")


class SingleFlight:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: dict[Hashable, Future] = {}

    def do(self, key: Hashable, fn: Callable[[], T]) -> tuple[T, bool]:
        """Run `fn` once per key at a time. Returns (result, shared): shared=True if this caller
        reused another caller's in-flight execution. Exceptions propagate to every waiter."""
        with self._lock:
            future = self._inflight.get(key)
            leader = future is None
            if leader:
                future = Future()
                self._inflight[key] = future
        if not leader:
            return future.result(), True
        try:
            result = fn()
            future.set_result(result)
            return result, False
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)


FETCHES = SingleFlight()  # process-wide: shared by every user session
