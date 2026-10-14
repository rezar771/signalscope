"""Daily reference exchange rates from Frankfurter (ECB)."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import requests
from pydantic import BaseModel, field_validator

from signalscope.config import Settings
from signalscope.sources.base import (
    FetchResult, SourceError, TimeSeries, Window,
    require_points, slug, split_window, to_daily_series,
)

FX_URL = "https://api.frankfurter.dev/v1"
# Currencies published by the ECB via Frankfurter (checked against /v1/currencies on 2026-10-14).
ECB_CURRENCIES = frozenset(
    "AUD BRL CAD CHF CNY CZK DKK EUR GBP HKD HUF IDR ILS INR ISK JPY KRW "
    "MXN MYR NOK NZD PHP PLN RON SEK SGD THB TRY USD ZAR".split()
)


class FxArgs(BaseModel):
    """Supported ECB currency codes, normalized before requests are made."""

    base: str = "EUR"
    quote: str

    @field_validator("base", "quote", mode="before")
    @classmethod
    def normalize_currency(cls, value: Any) -> Any:
        """Normalize currency names before validating their supported codes."""
        if not isinstance(value, str):
            return value
        value = value.strip().upper()
        aliases = {"DOLLAR": "USD", "EURO": "EUR", "POUND": "GBP", "YEN": "JPY", "FRANC": "CHF"}
        return aliases.get(value, value)

    @field_validator("base", "quote")
    @classmethod
    def currency_code(cls, value: str) -> str:
        """Normalize and validate the ISO currency-code format."""
        value = value.strip().upper()
        if re.fullmatch(r"[A-Z]{3}", value) is None:
            raise ValueError("Currency codes must contain three letters.")
        if value not in ECB_CURRENCIES:
            raise ValueError(f"Unsupported currency '{value}'. Allowed codes: {', '.join(sorted(ECB_CURRENCIES))}.")
        return value


def _get_series(
    args: FxArgs, window: Window, session: requests.Session, settings: Settings,
) -> tuple[pd.Series, requests.Response]:
    pair = f"{args.base}/{args.quote}"
    try:
        response = session.get(
            f"{FX_URL}/{window.start.isoformat()}..{window.end.isoformat()}",
            params={"base": args.base, "symbols": args.quote}, timeout=settings.http_timeout_s,
        )
        response.raise_for_status()
        rates = response.json()["rates"]
        if not isinstance(rates, dict) or not all(isinstance(row, dict) for row in rates.values()):
            raise ValueError("expected rates indexed by date")
        if any(args.quote not in row for row in rates.values()):
            raise SourceError(f"Unknown currency pair {pair}.")
        index = pd.DatetimeIndex(pd.to_datetime(list(rates), format="ISO8601"))
        if index.hasnans:
            raise ValueError("missing exchange-rate date")
        values = pd.Series([row[args.quote] for row in rates.values()], index=index, dtype=float)
        values = values.replace([float("inf"), float("-inf")], float("nan"))
    except requests.RequestException as exc:
        if exc.response is not None and exc.response.status_code in (404, 422):
            raise SourceError(f"Unknown currency pair {pair}.") from exc
        raise SourceError(f"Could not fetch exchange rate {pair}: Frankfurter request failed.") from exc
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise SourceError(f"Could not fetch exchange rate {pair}: Frankfurter returned malformed data.") from exc
    return values, response


def fetch_exchange_rate(
    args: FxArgs, window: Window, session: requests.Session, settings: Settings,
) -> FetchResult:
    """Fetch observed rates without filling weekends or other missing dates."""
    if args.base == args.quote:
        raise SourceError("Choose two different currencies for an exchange rate.")
    values, response = _get_series(args, window, session, settings)
    key, label = slug(f"fx_{args.base}_{args.quote}"), f"Exchange rate {args.base}/{args.quote}"
    history, _ = split_window(to_daily_series(values, key), window)
    require_points(history, label)
    return FetchResult(
        series=TimeSeries(key=key, label=label, unit=f"{args.quote} per {args.base}",
                          source="Frankfurter (ECB)", url=response.url,
                          license="ECB data, free reuse", history=history),
        cache_hit=bool(getattr(response, "from_cache", False)), http_calls=1,
    )
