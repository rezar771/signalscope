"""Offline test helpers: a fake requests session fed with recorded API responses.

Fixtures in tests/fixtures/ are real responses recorded on 2026-10-14; see tests/fixtures/README.md
for the exact request and date range of each file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from signalscope.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200, url: str = "", from_cache: bool = False) -> None:
        self._payload = payload
        self.status_code = status_code
        self.url = url
        self.from_cache = from_cache

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error for {self.url}", response=self)  # type: ignore[arg-type]


class FakeSession:
    """Routes GET requests by URL substring.

    routes: {substring: payload | FakeResponse}. The first substring found in the request URL wins
    (insertion order). Unmatched URLs raise requests.ConnectionError, like a dead network.
    Every call is recorded in `calls` as (url, params).
    """

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: float | None = None,
            **kwargs: Any) -> FakeResponse:
        self.calls.append((url, params))
        for key, value in self.routes.items():
            if key in url:
                if isinstance(value, FakeResponse):
                    if not value.url:
                        value.url = url
                    return value
                return FakeResponse(value, url=url)
        raise requests.ConnectionError(f"FakeSession: no route for {url}")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(cache_path=tmp_path / "http", openrouter_api_key=None, llm_mode="local")
