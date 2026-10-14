"""Energy-Charts quarter-hour power measurements aggregated into daily mean GW."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import requests
from pydantic import BaseModel, field_validator

from signalscope.config import Settings
from signalscope.sources.base import (
    FetchResult, SourceError, TimeSeries, Window,
    require_points, slug, split_window, to_daily_series,
)

ELECTRICITY_URL = "https://api.energy-charts.info/public_power"
METRICS = {
    "load": (("Load",), "Electricity load"),
    "solar": (("Solar",), "Solar generation"),
    "wind": (("Wind onshore", "Wind offshore"), "Wind generation"),
}


class ElectricityArgs(BaseModel):
    """Country and power measurement to retrieve."""

    country: Literal["it", "de", "fr", "es", "nl", "be", "at", "ch", "pl"]
    metric: Literal["load", "solar", "wind"]

    @field_validator("country", mode="before")
    @classmethod
    def normalize_country(cls, value: Any) -> Any:
        """Accept country codes or supported English country names."""
        if not isinstance(value, str):
            return value
        value = value.strip().lower()
        aliases = {
            "italy": "it", "germany": "de", "france": "fr", "spain": "es",
            "netherlands": "nl", "belgium": "be", "austria": "at",
            "switzerland": "ch", "poland": "pl",
        }
        return aliases.get(value, value)

    @field_validator("metric", mode="before")
    @classmethod
    def normalize_metric(cls, value: Any) -> Any:
        """Map common electricity terms to the supported measurements."""
        if not isinstance(value, str):
            return value
        value = value.strip().lower()
        aliases = {"demand": "load", "consumption": "load", "solar power": "solar",
                   "pv": "solar", "wind power": "wind"}
        return aliases.get(value, value)


def _daily_mean(payload: dict, args: ElectricityArgs) -> pd.Series:
    timestamps, production = payload["unix_seconds"], payload["production_types"]
    if not isinstance(timestamps, list) or not isinstance(production, list):
        raise ValueError("expected timestamp and production lists")
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, unit="s", utc=True)).tz_convert("Europe/Berlin")
    if index.hasnans:
        raise ValueError("missing power measurement timestamp")
    names, _ = METRICS[args.metric]
    columns = {}
    for item in production:
        if item["name"] in names:
            data = item["data"]
            if not isinstance(data, list) or len(data) != len(index):
                raise ValueError("expected one power measurement per timestamp")
            columns[item["name"]] = pd.Series(data, index=index, dtype=float)
    if not columns:
        raise SourceError(f"No {args.metric} data for country '{args.country}'.")
    frame = pd.DataFrame(columns).replace([float("inf"), float("-inf")], float("nan"))
    # An absent component contributes nothing, but all-missing measurements remain missing.
    values = frame.sum(axis=1, min_count=1)
    values = values[~values.index.duplicated(keep="last")].sort_index()
    daily = values.resample("D")
    means = daily.mean()
    # Local days may have 23 or 25 hours around daylight-saving transitions.
    expected = ((means.index + pd.DateOffset(days=1)) - means.index) / pd.Timedelta(minutes=15)
    return means.where(daily.count() >= 0.8 * expected) / 1000.0


def _get_series(
    args: ElectricityArgs, window: Window, session: requests.Session, settings: Settings,
) -> tuple[pd.Series, requests.Response]:
    try:
        response = session.get(
            ELECTRICITY_URL,
            params={"country": args.country, "start": window.start.isoformat(), "end": window.end.isoformat()},
            timeout=settings.http_timeout_s,
        )
        response.raise_for_status()
        values = _daily_mean(response.json(), args)
    except requests.RequestException as exc:
        raise SourceError(f"Could not fetch {args.metric} for {args.country.upper()}: Energy-Charts request failed.") from exc
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise SourceError(f"Could not fetch {args.metric} for {args.country.upper()}: Energy-Charts returned malformed data.") from exc
    return values, response


def fetch_electricity(
    args: ElectricityArgs, window: Window, session: requests.Session, settings: Settings,
) -> FetchResult:
    """Fetch daily power means, discarding days with less than 80% coverage."""
    values, response = _get_series(args, window, session, settings)
    key = slug(f"electricity_{args.metric}_{args.country}")
    label = f"{METRICS[args.metric][1]} — {args.country.upper()}"
    history, _ = split_window(to_daily_series(values, key), window)
    require_points(history, label)
    return FetchResult(
        series=TimeSeries(key=key, label=label, unit="GW", source="Energy-Charts (Fraunhofer ISE)",
                          url=response.url, license="CC BY 4.0", history=history),
        cache_hit=bool(getattr(response, "from_cache", False)), http_calls=1,
    )
