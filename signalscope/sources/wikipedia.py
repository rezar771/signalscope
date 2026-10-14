"""Wikipedia title resolution and daily pageviews as a proxy for public interest."""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import quote

import pandas as pd
import requests
from pydantic import BaseModel, field_validator

from signalscope.config import Settings
from signalscope.sources.base import (
    FetchResult, SourceError, TimeSeries, Window,
    require_points, slug, split_window, to_daily_series,
)

PAGEVIEWS_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"


class WikipediaArgs(BaseModel):
    """Topic and Wikipedia edition to look up."""

    topic: str
    language: Literal["en", "it", "de", "fr", "es"] = "en"

    @field_validator("topic", mode="before")
    @classmethod
    def normalize_topic(cls, value: Any) -> Any:
        """Trim the topic without changing the article's case."""
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("Topic must not be empty.")
        return value

    @field_validator("language", mode="before")
    @classmethod
    def normalize_language(cls, value: Any) -> Any:
        """Normalize the Wikipedia edition code."""
        return value.strip().lower() if isinstance(value, str) else value


def _resolve_title(
    args: WikipediaArgs, session: requests.Session, settings: Settings,
) -> tuple[str, bool]:
    try:
        response = session.get(
            f"https://{args.language}.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": args.topic, "limit": 1,
                    "namespace": 0, "format": "json"},
            timeout=settings.http_timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) != 4 or not isinstance(payload[1], list):
            raise ValueError("expected an OpenSearch result")
        if not payload[1]:
            raise SourceError(f"No Wikipedia article found for '{args.topic}'.")
        title = payload[1][0]
        if not isinstance(title, str) or not title.strip():
            raise ValueError("missing article title")
    except requests.RequestException as exc:
        raise SourceError(f"Could not resolve Wikipedia topic '{args.topic}': request failed.") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceError(f"Could not resolve Wikipedia topic '{args.topic}': malformed data.") from exc
    return title, bool(getattr(response, "from_cache", False))


def _get_series(
    title: str, language: str, window: Window, session: requests.Session, settings: Settings,
) -> tuple[pd.Series, requests.Response]:
    article = quote(title.replace(" ", "_"), safe="")
    url = (f"{PAGEVIEWS_URL}/{language}.wikipedia/all-access/user/{article}"
           f"/daily/{window.start:%Y%m%d}/{window.end:%Y%m%d}")
    try:
        response = session.get(url, params={}, timeout=settings.http_timeout_s)
        response.raise_for_status()
        items = response.json()["items"]
        if not isinstance(items, list):
            raise ValueError("expected a pageviews list")
        index = pd.DatetimeIndex(pd.to_datetime([item["timestamp"] for item in items], format="%Y%m%d%H"))
        if index.hasnans:
            raise ValueError("missing pageview date")
        values = pd.Series([item["views"] for item in items], index=index, dtype=float)
        values = values.replace([float("inf"), float("-inf")], float("nan"))
    except requests.RequestException as exc:
        raise SourceError(f"Could not fetch Wikipedia pageviews for '{title}': request failed.") from exc
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise SourceError(f"Could not fetch Wikipedia pageviews for '{title}': malformed data.") from exc
    return values, response


def fetch_wikipedia_interest(
    args: WikipediaArgs, window: Window, session: requests.Session, settings: Settings,
) -> FetchResult:
    """Resolve a topic and return daily pageviews, filling omitted days with zero."""
    title, title_cached = _resolve_title(args, session, settings)
    values, response = _get_series(title, args.language, window, session, settings)
    key, label = f"wikipedia_{args.language}_{slug(title)}", f"Wikipedia interest — {title}"
    # Reindex before dropping nulls: an explicit null is missing data, not zero views.
    values = values[~values.index.duplicated(keep="last")].reindex(
        pd.date_range(window.start, window.end), fill_value=0.0,
    )
    history, _ = split_window(to_daily_series(values, key), window)
    require_points(history, label)
    return FetchResult(
        series=TimeSeries(key=key, label=label, unit="views/day", source="Wikimedia Pageviews",
                          url=response.url, license="CC0", history=history),
        cache_hit=title_cached and bool(getattr(response, "from_cache", False)), http_calls=2,
    )
