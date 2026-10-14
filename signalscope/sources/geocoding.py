"""Resolve city names through Open-Meteo's geocoding API."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import requests

from signalscope.config import Settings
from signalscope.sources.base import SourceError

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


@dataclass(frozen=True)
class Place:
    """Resolved place and its local timezone."""

    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str


def lookup(
    city: str, session: requests.Session, settings: Settings
) -> tuple[Place, bool]:
    """Resolve a city; also return whether the response came from the HTTP cache."""
    try:
        response = session.get(
            GEOCODING_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=settings.http_timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ValueError("expected a results list")
        if not results:
            raise SourceError(f"Could not find a place called '{city}'.")
        item = results[0]
        name, country, timezone = item["name"], item["country"], item["timezone"]
        if not all(isinstance(value, str) and value.strip() for value in (name, country, timezone)):
            raise ValueError("missing place details")
        latitude, longitude = float(item["latitude"]), float(item["longitude"])
        if not (isfinite(latitude) and -90 <= latitude <= 90
                and isfinite(longitude) and -180 <= longitude <= 180):
            raise ValueError("invalid coordinates")
        place = Place(name, country, latitude, longitude, timezone)
    except requests.RequestException as exc:
        raise SourceError(f"Could not look up '{city}': geocoding service request failed.") from exc
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise SourceError(f"Could not look up '{city}': geocoding service returned malformed data.") from exc
    return place, bool(getattr(response, "from_cache", False))


def geocode(city: str, session: requests.Session, settings: Settings) -> Place:
    """Resolve a city, raising SourceError if lookup fails or no place matches."""
    place, _ = lookup(city, session, settings)
    return place
