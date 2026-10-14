"""HTTP session factory: SQLite response cache + retries with backoff.

Create one session per fetch task (sessions are not shared across threads); every session
points at the same on-disk cache, so repeated questions from any user hit the cache instead of
the network (Energy-Charts alone is ~6 s / 1.2 MB per 90-day request). WAL mode lets concurrent
runs read while another writes.
"""

from __future__ import annotations

import threading
from http.cookiejar import DefaultCookiePolicy

import requests
import requests_cache
from requests.adapters import HTTPAdapter
from requests_cache.backends.sqlite import SQLiteCache
from urllib3.util.retry import Retry

from signalscope.config import Settings


_BACKENDS: dict[str, SQLiteCache] = {}
_BACKENDS_LOCK = threading.Lock()


def _shared_backend(settings: Settings) -> SQLiteCache:
    """One cache backend per process and path. Creating it per session made parallel threads
    race on SQLite initialisation ("database is locked", caught by a concurrency test)."""
    key = str(settings.cache_path)
    with _BACKENDS_LOCK:
        if key not in _BACKENDS:
            settings.cache_path.parent.mkdir(parents=True, exist_ok=True)
            _BACKENDS[key] = SQLiteCache(key, wal=True, busy_timeout=30_000)
        return _BACKENDS[key]


def make_session(settings: Settings) -> requests.Session:
    session = requests_cache.CachedSession(
        backend=_shared_backend(settings),
        expire_after=settings.cache_ttl_s,
        allowable_codes=(200,),
        stale_if_error=True,    # serve an expired copy if the provider is down
    )
    # No cookies: we only read public data, and Wikipedia's `Vary: Cookie` would otherwise turn
    # every repeated request into a cache miss (measured). Also: nothing user-identifying is stored.
    session.cookies.set_policy(DefaultCookiePolicy(allowed_domains=[]))
    retry = Retry(
        total=3,
        read=0,                 # a slow provider (Energy-Charts: up to ~16 s measured) is not retried
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers["User-Agent"] = settings.user_agent  # required by Wikimedia
    return session
