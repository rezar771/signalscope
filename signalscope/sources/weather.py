"""Daily weather and air-quality adapters for Open-Meteo."""

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
from signalscope.sources.geocoding import Place, lookup

# The forecast endpoint returns nulls for the oldest ~15 days of a 92-day lookback (verified
# 2026-10-14), so weather history comes from the historical-forecast endpoint for the exact window.
WEATHER_HISTORY_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_FORECAST_DAYS = 7
AIR_FORECAST_DAYS = 5

# API field, output unit, label, divisor for conversion into the output unit.
WEATHER_VARIABLES = {
    "temperature": ("temperature_2m_mean", "°C", "Mean temperature", 1.0),
    "precipitation": ("precipitation_sum", "mm", "Precipitation", 1.0),
    "wind_speed": ("wind_speed_10m_max", "km/h", "Max wind speed", 1.0),
    "sunshine": ("sunshine_duration", "h", "Sunshine duration", 3600.0),
}
POLLUTANT_LABELS = {"pm2_5": "PM2.5", "pm10": "PM10", "nitrogen_dioxide": "NO₂", "ozone": "Ozone"}


class WeatherArgs(BaseModel):
    """Arguments exposed to the weather tool."""

    city: str
    variable: Literal["temperature", "precipitation", "wind_speed", "sunshine"]

    @field_validator("city", mode="before")
    @classmethod
    def normalize_city(cls, value: Any) -> Any:
        """Trim city names and reject empty input."""
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("City must not be empty.")
        return value

    @field_validator("variable", mode="before")
    @classmethod
    def normalize_variable(cls, value: Any) -> Any:
        """Map common weather terms to the supported variables."""
        if not isinstance(value, str):
            return value
        value = value.strip().lower()
        aliases = {
            "temp": "temperature", "temperatures": "temperature", "heat": "temperature",
            "rain": "precipitation", "rainfall": "precipitation", "wind": "wind_speed",
            "sun": "sunshine", "sunlight": "sunshine",
        }
        return aliases.get(value, value)


class AirQualityArgs(BaseModel):
    """Arguments exposed to the air-quality tool."""

    city: str
    pollutant: Literal["pm2_5", "pm10", "nitrogen_dioxide", "ozone"]

    @field_validator("city", mode="before")
    @classmethod
    def normalize_city(cls, value: Any) -> Any:
        """Trim city names and reject empty input."""
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("City must not be empty.")
        return value

    @field_validator("pollutant", mode="before")
    @classmethod
    def normalize_pollutant(cls, value: Any) -> Any:
        """Normalize common pollutant abbreviations."""
        if not isinstance(value, str):
            return value
        value = value.strip().lower()
        aliases = {"pm2.5": "pm2_5", "pm25": "pm2_5", "no2": "nitrogen_dioxide", "o3": "ozone"}
        return aliases.get(value, value)


def _parse_values(payload: dict, frequency: str, field: str) -> pd.Series:
    block = payload[frequency]
    times, values = block["time"], block[field]
    if not isinstance(times, list) or not isinstance(values, list) or len(times) != len(values):
        raise ValueError("expected equally sized time and value lists")
    index = pd.DatetimeIndex(pd.to_datetime(times, format="ISO8601"))
    if index.hasnans:
        raise ValueError("missing timestamps")
    if index.tz is not None:
        index = index.tz_localize(None)
    series = pd.to_numeric(pd.Series(values, index=index, dtype=object), errors="coerce").astype(float)
    series = series.replace([float("inf"), float("-inf")], float("nan"))
    if frequency == "hourly":
        # Duplicate hours must not inflate the daily coverage check.
        series = series[~series.index.duplicated(keep="last")].sort_index()
        daily = series.resample("D")
        series = daily.mean().where(daily.count() >= 18)
    return series


def _get_series(
    url: str, params: dict[str, Any], frequency: Literal["daily", "hourly"], field: str,
    label: str, session: requests.Session, settings: Settings,
) -> tuple[pd.Series, requests.Response]:
    """One Open-Meteo request → raw (not yet windowed) series; errors become SourceError."""
    try:
        response = session.get(url, params={**params, frequency: field}, timeout=settings.http_timeout_s)
        response.raise_for_status()
        return _parse_values(response.json(), frequency, field), response
    except requests.RequestException as exc:
        raise SourceError(f"Could not fetch {label}: Open-Meteo request failed.") from exc
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise SourceError(f"Could not fetch {label}: Open-Meteo returned malformed data.") from exc


def _location(place: Place) -> dict[str, Any]:
    return {"latitude": place.latitude, "longitude": place.longitude, "timezone": place.timezone}


def _build(
    values: pd.Series, key: str, label: str, unit: str, url: str, window: Window,
    responses: list[Any], geocode_cached: bool,
) -> FetchResult:
    history, future = split_window(to_daily_series(values, key), window)
    require_points(history, label)
    return FetchResult(
        series=TimeSeries(
            key=key, label=label, unit=unit, source="Open-Meteo", url=url,
            license="CC BY 4.0", history=history, future=future, exogenous=True,
        ),
        cache_hit=geocode_cached and all(getattr(r, "from_cache", False) for r in responses),
        http_calls=1 + len(responses),
    )


def fetch_weather(
    args: WeatherArgs, window: Window, session: requests.Session, settings: Settings
) -> FetchResult:
    """Daily weather for the window (historical endpoint) plus a 7-day forecast."""
    field, unit, prefix, divisor = WEATHER_VARIABLES[args.variable]
    place, geocode_cached = lookup(args.city, session, settings)
    key, label = f"{args.variable}_{slug(place.name)}", f"{prefix} — {place.name}"
    history, history_resp = _get_series(
        WEATHER_HISTORY_URL,
        {**_location(place), "start_date": window.start.isoformat(), "end_date": window.end.isoformat()},
        "daily", field, label, session, settings,
    )
    forecast, forecast_resp = _get_series(
        WEATHER_FORECAST_URL, {**_location(place), "forecast_days": WEATHER_FORECAST_DAYS},
        "daily", field, label, session, settings,
    )
    # History wins on overlapping dates; the forecast only contributes days after the window.
    combined = pd.concat([history, forecast[~forecast.index.isin(history.index)]]).sort_index() / divisor
    return _build(combined, key, label, unit, history_resp.url, window,
                  [history_resp, forecast_resp], geocode_cached)


def fetch_air_quality(
    args: AirQualityArgs, window: Window, session: requests.Session, settings: Settings
) -> FetchResult:
    """Daily pollutant means (days with ≥ 18 valid hours) for the window plus a 5-day forecast."""
    place, geocode_cached = lookup(args.city, session, settings)
    key, label = f"{args.pollutant}_{slug(place.name)}", f"{POLLUTANT_LABELS[args.pollutant]} — {place.name}"
    values, response = _get_series(
        AIR_QUALITY_URL,
        {**_location(place), "past_days": min(window.days + 1, 92), "forecast_days": AIR_FORECAST_DAYS},
        "hourly", args.pollutant, label, session, settings,
    )
    return _build(values, key, label, "μg/m³", response.url, window, [response], geocode_cached)
