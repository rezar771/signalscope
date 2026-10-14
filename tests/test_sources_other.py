"""Focused offline checks for Wikipedia, Frankfurter and Energy-Charts adapters."""

from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from conftest import FakeResponse, FakeSession, load_fixture
from signalscope.sources.base import SourceError, Window, to_daily_series
from signalscope.sources.electricity import (
    ElectricityArgs, _get_series as get_electricity_series, fetch_electricity,
)
from signalscope.sources.fx import FxArgs, fetch_exchange_rate
from signalscope.sources.wikipedia import WikipediaArgs, fetch_wikipedia_interest

WINDOW = Window(date(2026, 9, 14), date(2026, 10, 13))
ENERGY_WINDOW = Window(date(2026, 10, 10), date(2026, 10, 12))
SEARCH_URL = "https://en.wikipedia.org/w/api.php"
VIEWS_URL = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
             "en.wikipedia/all-access/user/Ice_cream/daily/20260914/20261013")
FX_URL = "https://api.frankfurter.dev/v1/2026-09-14..2026-10-13"
ENERGY_URL = "https://api.energy-charts.info/public_power"


def assert_history(series):
    assert series.history.dtype == float and not series.history.isna().any()
    assert series.history.index.tz is None
    assert series.history.index.equals(series.history.index.normalize())
    assert series.history.index.is_unique and series.history.index.is_monotonic_increasing
    assert series.history.index.name == "date" and series.history.name == series.key
    assert series.future is None and not series.exogenous


def test_wikipedia_resolution_pageviews_and_metadata(settings):
    payload = load_fixture("wiki_pageviews_ice_cream.json")
    session = FakeSession({
        SEARCH_URL: FakeResponse(load_fixture("wiki_opensearch_ice_cream.json"), from_cache=True),
        VIEWS_URL: FakeResponse(payload, from_cache=True, url=VIEWS_URL),
    })
    result = fetch_wikipedia_interest(WikipediaArgs(topic="ice cream"), WINDOW, session, settings)
    series = result.series
    assert_history(series)
    assert series.history.index.equals(pd.date_range(WINDOW.start, WINDOW.end, name="date"))
    assert series.history.iloc[0] == payload["items"][0]["views"]
    assert series.key == "wikipedia_en_ice_cream" and series.label == "Wikipedia interest — Ice cream"
    assert (series.unit, series.source, series.license) == ("views/day", "Wikimedia Pageviews", "CC0")
    assert series.url == VIEWS_URL and result.cache_hit
    assert result.http_calls == len(session.calls) == 2
    assert session.calls[0] == (SEARCH_URL, {"action": "opensearch", "search": "ice cream",
                                            "limit": 1, "namespace": 0, "format": "json"})
    assert session.calls[1][0] == VIEWS_URL
    session.routes[SEARCH_URL].from_cache = False
    assert not fetch_wikipedia_interest(WikipediaArgs(topic="ice cream"), WINDOW, session, settings).cache_hit


def test_wikipedia_missing_day_zero_explicit_null_dropped(settings):
    payload = load_fixture("wiki_pageviews_ice_cream.json")
    payload["items"].pop(0)
    payload["items"][0]["views"] = None
    session = FakeSession({SEARCH_URL: load_fixture("wiki_opensearch_ice_cream.json"), VIEWS_URL: payload})
    series = fetch_wikipedia_interest(WikipediaArgs(topic="ice cream"), WINDOW, session, settings).series
    assert series.history.loc["2026-09-14"] == 0
    assert pd.Timestamp("2026-09-15") not in series.history.index
    assert len(series.history) == 29


def test_wikipedia_not_found(settings):
    session = FakeSession({SEARCH_URL: load_fixture("wiki_opensearch_not_found.json")})
    with pytest.raises(SourceError, match="No Wikipedia article found"):
        fetch_wikipedia_interest(WikipediaArgs(topic="xqzvbnmlkjhgf"), WINDOW, session, settings)
    assert len(session.calls) == 1


def test_wikipedia_title_encoding_and_language(settings):
    search = load_fixture("wiki_opensearch_ice_cream.json")
    search[1][0] = "Ice cream/gelato?"
    url = VIEWS_URL.replace("en.wikipedia", "it.wikipedia").replace("Ice_cream", "Ice_cream%2Fgelato%3F")
    session = FakeSession({"https://it.wikipedia.org/w/api.php": search,
                           url: load_fixture("wiki_pageviews_ice_cream.json")})
    result = fetch_wikipedia_interest(WikipediaArgs(topic="gelato", language="it"), WINDOW, session, settings)
    assert session.calls[1][0] == url
    assert result.series.label == "Wikipedia interest — Ice cream/gelato?"


def test_fx_business_days_and_metadata(settings):
    payload = load_fixture("fx_eur_usd.json")
    provenance = FX_URL + "?base=EUR&symbols=USD"
    session = FakeSession({FX_URL: FakeResponse(payload, from_cache=True, url=provenance)})
    result = fetch_exchange_rate(FxArgs(quote="usd"), WINDOW, session, settings)
    series = result.series
    assert_history(series)
    assert len(series.history) == 22 and (series.history.index.dayofweek < 5).all()
    assert series.history.loc["2026-09-14"] == payload["rates"]["2026-09-14"]["USD"]
    assert series.key == "fx_eur_usd" and series.label == "Exchange rate EUR/USD"
    assert (series.unit, series.source, series.license) == ("USD per EUR", "Frankfurter (ECB)", "ECB data, free reuse")
    assert series.url == provenance and result.cache_hit
    assert result.http_calls == len(session.calls) == 1
    assert session.calls[0] == (FX_URL, {"base": "EUR", "symbols": "USD"})


def test_fx_argument_validation_and_same_currency(settings):
    assert FxArgs(base=" eur ", quote="usd").model_dump() == {"base": "EUR", "quote": "USD"}
    with pytest.raises(ValidationError, match="three letters"):
        FxArgs(quote="US1")
    session = FakeSession({})
    with pytest.raises(SourceError, match="different currencies"):
        fetch_exchange_rate(FxArgs(quote="EUR"), WINDOW, session, settings)
    assert not session.calls


@pytest.mark.parametrize("status", [404, 422])
def test_fx_unknown_pair_http_status(settings, status):
    session = FakeSession({FX_URL: FakeResponse({}, status_code=status)})
    with pytest.raises(SourceError, match="Unknown currency pair EUR/GBP"):
        fetch_exchange_rate(FxArgs(quote="GBP"), WINDOW, session, settings)


def test_fx_unknown_pair_missing_quote(settings):
    session = FakeSession({FX_URL: load_fixture("fx_eur_usd.json")})
    with pytest.raises(SourceError, match="Unknown currency pair EUR/GBP"):
        fetch_exchange_rate(FxArgs(quote="GBP"), WINDOW, session, settings)


def test_fx_nulls_and_window_clipping(settings):
    payload = load_fixture("fx_eur_usd.json")
    payload["rates"]["2026-09-14"]["USD"] = None
    # Returning a wider response must not leak dates outside the requested window.
    window = Window(date(2026, 9, 14), date(2026, 9, 25))
    session = FakeSession({"api.frankfurter.dev": payload})
    series = fetch_exchange_rate(FxArgs(quote="USD"), window, session, settings).series
    assert len(series.history) == 9
    assert series.history.index.min() == pd.Timestamp("2026-09-15")
    assert series.history.index.max() == pd.Timestamp("2026-09-25")
    assert series.future is None


@pytest.mark.parametrize("metric,names", [
    ("load", ["Load"]), ("solar", ["Solar"]), ("wind", ["Wind onshore", "Wind offshore"]),
])
def test_electricity_local_daily_mean_gw(settings, metric, names):
    # The recording has three days: test request/aggregation before the five-day guard.
    payload = load_fixture("energy_it.json")
    provenance = ENERGY_URL + "?country=it&start=2026-10-10&end=2026-10-12"
    session = FakeSession({ENERGY_URL: FakeResponse(payload, from_cache=True, url=provenance)})
    values, response = get_electricity_series(
        ElectricityArgs(country="it", metric=metric), ENERGY_WINDOW, session, settings,
    )
    daily = to_daily_series(values, "power")
    assert daily.index.equals(pd.date_range("2026-10-10", "2026-10-12", name="date"))
    expected = sum(sum(item["data"][:96]) for item in payload["production_types"] if item["name"] in names) / 96 / 1000
    assert daily.iloc[0] == pytest.approx(expected)
    assert response.url == provenance and response.from_cache
    assert session.calls == [(ENERGY_URL, {"country": "it", "start": "2026-10-10", "end": "2026-10-12"})]


def test_electricity_wind_missing_offshore_and_nulls(settings):
    payload = load_fixture("energy_it.json")
    payload["production_types"] = [item for item in payload["production_types"] if item["name"] == "Wind onshore"]
    data = payload["production_types"][0]["data"]
    data[0] = None
    values, _ = get_electricity_series(
        ElectricityArgs(country="it", metric="wind"), ENERGY_WINDOW,
        FakeSession({ENERGY_URL: payload}), settings,
    )
    assert values.iloc[0] == pytest.approx(sum(data[1:96]) / 95 / 1000)


def test_electricity_wind_partial_and_all_missing_measurements(settings):
    payload = load_fixture("energy_it.json")
    types = {item["name"]: item["data"] for item in payload["production_types"]}
    types["Wind onshore"][0] = types["Wind offshore"][0] = None
    types["Wind offshore"][1] = None
    expected = (types["Wind onshore"][1] + sum(types["Wind onshore"][2:96])
                + sum(types["Wind offshore"][2:96])) / 95 / 1000
    values, _ = get_electricity_series(
        ElectricityArgs(country="it", metric="wind"), ENERGY_WINDOW,
        FakeSession({ENERGY_URL: payload}), settings,
    )
    assert values.iloc[0] == pytest.approx(expected)


def test_electricity_coverage_threshold(settings):
    payload = load_fixture("energy_it.json")
    load = next(item["data"] for item in payload["production_types"] if item["name"] == "Load")
    load[:20] = [None] * 20       # 76/96: insufficient
    load[96:115] = [None] * 19   # 77/96: sufficient
    values, _ = get_electricity_series(
        ElectricityArgs(country="it", metric="load"), ENERGY_WINDOW,
        FakeSession({ENERGY_URL: payload}), settings,
    )
    daily = to_daily_series(values, "load")
    assert pd.Timestamp("2026-10-10") not in daily.index
    assert daily.loc["2026-10-11"] == pytest.approx(sum(load[115:192]) / 77 / 1000)


def test_electricity_missing_type(settings):
    payload = load_fixture("energy_it.json")
    payload["production_types"] = [item for item in payload["production_types"] if item["name"] != "Solar"]
    with pytest.raises(SourceError, match="No solar data for country 'it'"):
        fetch_electricity(ElectricityArgs(country="it", metric="solar"), ENERGY_WINDOW,
                          FakeSession({ENERGY_URL: payload}), settings)


@pytest.mark.parametrize("source", ["wikipedia", "fx", "electricity"])
def test_minimum_history_points(settings, source):
    if source == "wikipedia":
        session = FakeSession({SEARCH_URL: load_fixture("wiki_opensearch_ice_cream.json"),
                               "wikimedia.org": load_fixture("wiki_pageviews_ice_cream.json")})
        fetch, args = fetch_wikipedia_interest, WikipediaArgs(topic="ice cream")
        window = Window(date(2026, 9, 14), date(2026, 9, 16))
    elif source == "fx":
        session = FakeSession({"api.frankfurter.dev": load_fixture("fx_eur_usd.json")})
        fetch, args = fetch_exchange_rate, FxArgs(quote="USD")
        window = Window(date(2026, 9, 14), date(2026, 9, 16))
    else:
        session = FakeSession({ENERGY_URL: load_fixture("energy_it.json")})
        fetch, args, window = fetch_electricity, ElectricityArgs(country="it", metric="load"), ENERGY_WINDOW
    with pytest.raises(SourceError, match="only 3 days available"):
        fetch(args, window, session, settings)


@pytest.mark.parametrize("stage", ["wiki_search", "wiki_views", "fx", "electricity"])
def test_request_and_malformed_errors(settings, stage):
    if stage.startswith("wiki"):
        fetch, args, window = fetch_wikipedia_interest, WikipediaArgs(topic="ice cream"), WINDOW
        routes = {SEARCH_URL: load_fixture("wiki_opensearch_ice_cream.json"),
                  VIEWS_URL: load_fixture("wiki_pageviews_ice_cream.json")}
        url = SEARCH_URL if stage == "wiki_search" else VIEWS_URL
    elif stage == "fx":
        fetch, args, window = fetch_exchange_rate, FxArgs(quote="USD"), WINDOW
        url, routes = FX_URL, {}
    else:
        fetch, args, window = fetch_electricity, ElectricityArgs(country="it", metric="load"), ENERGY_WINDOW
        url, routes = ENERGY_URL, {}
    routes[url] = FakeResponse({}, status_code=503)
    with pytest.raises(SourceError, match="request failed"):
        fetch(args, window, FakeSession(routes), settings)
    routes[url] = {}
    with pytest.raises(SourceError, match="malformed data"):
        fetch(args, window, FakeSession(routes), settings)
    del routes[url]
    with pytest.raises(SourceError, match="request failed"):
        fetch(args, window, FakeSession(routes), settings)


def test_electricity_mismatched_array_lengths(settings):
    payload = load_fixture("energy_it.json")
    next(item["data"] for item in payload["production_types"] if item["name"] == "Load").pop()
    with pytest.raises(SourceError, match="malformed data"):
        fetch_electricity(ElectricityArgs(country="it", metric="load"), ENERGY_WINDOW,
                          FakeSession({ENERGY_URL: payload}), settings)
