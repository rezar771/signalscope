"""Offline contract tests using recorded Open-Meteo responses."""

from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from conftest import FakeResponse, FakeSession, load_fixture
from signalscope.sources.base import SourceError, Window
from signalscope.sources.geocoding import Place, geocode
from signalscope.sources.weather import (
    AirQualityArgs, WeatherArgs, fetch_air_quality, fetch_weather,
)

WINDOW = Window(date(2026, 9, 14), date(2026, 10, 13))
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
HISTORY_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def source_case(kind):
    """Return the adapter, args, history URL, geocode payload, history payload, extra routes,
    frequency and field for a source. Payloads are fresh copies, safe to mutate."""
    if kind == "weather":
        return (
            fetch_weather, WeatherArgs(city="Rome", variable="temperature"), HISTORY_URL,
            load_fixture("geocoding_rome.json"), load_fixture("weather_rome_history.json"),
            {FORECAST_URL: load_fixture("weather_rome_forecast.json")}, "daily", "temperature_2m_mean",
        )
    return (
        fetch_air_quality, AirQualityArgs(city="Milan", pollutant="pm10"), AIR_URL,
        load_fixture("geocoding_milan.json"), load_fixture("air_quality_milan.json"),
        {}, "hourly", "pm10",
    )


def weather_session(history=None, forecast=None):
    # HISTORY_URL is listed before FORECAST_URL; FakeSession matches by substring in order.
    return FakeSession({
        GEO_URL: load_fixture("geocoding_rome.json"),
        HISTORY_URL: history if history is not None else load_fixture("weather_rome_history.json"),
        FORECAST_URL: forecast if forecast is not None else load_fixture("weather_rome_forecast.json"),
    })


def assert_daily_contract(series):
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index.tz is None
    assert series.index.equals(series.index.normalize())
    assert series.index.is_unique and series.index.is_monotonic_increasing
    assert series.index.name == "date"
    assert series.dtype == float
    assert not series.isna().any()


def test_geocode_success(settings):
    session = FakeSession({GEO_URL: load_fixture("geocoding_rome.json")})
    assert geocode("Rome", session, settings) == Place("Rome", "Italy", 41.89193, 12.51133, "Europe/Rome")
    assert session.calls == [(GEO_URL, {"name": "Rome", "count": 1, "language": "en", "format": "json"})]


@pytest.mark.parametrize("empty_list", [False, True])
def test_geocode_not_found(settings, empty_list):
    payload = load_fixture("geocoding_not_found.json")
    if empty_list:
        payload["results"] = []
    with pytest.raises(SourceError, match="Could not find a place called 'Xqzvbnmlkj'."):
        geocode("Xqzvbnmlkj", FakeSession({GEO_URL: payload}), settings)


@pytest.mark.parametrize("payload", [None, [], {"results": None}, {"results": [{}]}])
def test_geocode_malformed_response(settings, payload):
    with pytest.raises(SourceError, match="geocoding service returned malformed data"):
        geocode("Rome", FakeSession({GEO_URL: payload}), settings)


@pytest.mark.parametrize("field,value", [("latitude", "NaN"), ("longitude", 181), ("timezone", None)])
def test_geocode_invalid_place(settings, field, value):
    payload = load_fixture("geocoding_rome.json")
    payload["results"][0][field] = value
    with pytest.raises(SourceError, match="malformed data"):
        geocode("Rome", FakeSession({GEO_URL: payload}), settings)


@pytest.mark.parametrize("variable,field,unit,prefix,divisor", [
    ("temperature", "temperature_2m_mean", "°C", "Mean temperature", 1),
    ("precipitation", "precipitation_sum", "mm", "Precipitation", 1),
    ("wind_speed", "wind_speed_10m_max", "km/h", "Max wind speed", 1),
    ("sunshine", "sunshine_duration", "h", "Sunshine duration", 3600),
])
def test_weather_variables_and_window(settings, variable, field, unit, prefix, divisor):
    history, forecast = load_fixture("weather_rome_history.json"), load_fixture("weather_rome_forecast.json")
    session = weather_session(history, forecast)
    result = fetch_weather(WeatherArgs(city="Rome", variable=variable), WINDOW, session, settings)
    series = result.series
    assert series.key == f"{variable}_rome"
    assert series.label == f"{prefix} — Rome"
    assert series.unit == unit
    assert series.source == "Open-Meteo" and series.license == "CC BY 4.0"
    assert series.exogenous and series.history.name == series.key
    assert_daily_contract(series.history)
    assert series.history.index.equals(pd.date_range("2026-09-14", "2026-10-13", name="date"))
    assert series.history.iloc[0] == pytest.approx(history["daily"][field][0] / divisor)
    assert series.future is not None
    assert_daily_contract(series.future)
    assert series.future.index.equals(pd.date_range("2026-10-14", "2026-10-20", name="date"))
    assert series.future.iloc[0] == pytest.approx(forecast["daily"][field][0] / divisor)
    assert result.http_calls == len(session.calls) == 3
    assert not result.cache_hit
    location = {"latitude": 41.89193, "longitude": 12.51133, "timezone": "Europe/Rome"}
    assert session.calls[1] == (HISTORY_URL, {
        **location, "daily": field, "start_date": "2026-09-14", "end_date": "2026-10-13"})
    assert session.calls[2] == (FORECAST_URL, {**location, "daily": field, "forecast_days": 7})


def test_weather_nulls_dropped_and_no_future(settings):
    history, forecast = load_fixture("weather_rome_history.json"), load_fixture("weather_rome_forecast.json")
    history["daily"]["temperature_2m_mean"][0] = None
    forecast["daily"]["temperature_2m_mean"] = [None] * 7
    session = weather_session(history, forecast)
    series = fetch_weather(WeatherArgs(city="Rome", variable="temperature"), WINDOW, session, settings).series
    assert len(series.history) == 29
    assert pd.Timestamp("2026-09-14") not in series.history.index
    assert not series.history.isna().any()
    assert series.future is None


@pytest.mark.parametrize("pollutant,label", [("pm2_5", "PM2.5"), ("pm10", "PM10")])
def test_air_quality_daily_means(settings, pollutant, label):
    payload = load_fixture("air_quality_milan.json")
    session = FakeSession({GEO_URL: load_fixture("geocoding_milan.json"), AIR_URL: payload})
    result = fetch_air_quality(AirQualityArgs(city="Milan", pollutant=pollutant), WINDOW, session, settings)
    series = result.series
    assert series.key == f"{pollutant}_milan" and series.label == f"{label} — Milan"
    assert series.unit == "μg/m³" and series.exogenous
    assert series.source == "Open-Meteo" and series.license == "CC BY 4.0"
    assert_daily_contract(series.history)
    assert series.history.index.equals(pd.date_range("2026-09-29", "2026-10-13", name="date"))
    assert series.history.iloc[0] == pytest.approx(sum(payload["hourly"][pollutant][:24]) / 24)
    assert series.future is not None
    assert_daily_contract(series.future)
    assert series.future.index.equals(pd.date_range("2026-10-14", "2026-10-16", name="date"))
    assert result.http_calls == len(session.calls) == 2
    assert session.calls[1] == (AIR_URL, {
        "latitude": 45.46427, "longitude": 9.18951, "hourly": pollutant,
        "past_days": 31, "forecast_days": 5, "timezone": "Europe/Rome",
    })


@pytest.mark.parametrize("missing_hours,kept", [(6, True), (7, False), (24, False)])
@pytest.mark.parametrize("day", ["2026-09-29", "2026-10-14"])
def test_air_quality_coverage_threshold(settings, missing_hours, kept, day):
    payload = load_fixture("air_quality_milan.json")
    start = payload["hourly"]["time"].index(day + "T00:00")
    expected_values = payload["hourly"]["pm10"][start + missing_hours:start + 24]
    payload["hourly"]["pm10"][start:start + missing_hours] = [None] * missing_hours
    session = FakeSession({GEO_URL: load_fixture("geocoding_milan.json"), AIR_URL: payload})
    series = fetch_air_quality(AirQualityArgs(city="Milan", pollutant="pm10"), WINDOW, session, settings).series
    values = series.history if day == "2026-09-29" else series.future
    assert (pd.Timestamp(day) in values.index) is kept
    if kept:
        assert values.loc[day] == pytest.approx(sum(expected_values) / len(expected_values))


@pytest.mark.parametrize("kind", ["weather", "air"])
@pytest.mark.parametrize("geo_cached,data_cached", [(False, False), (True, False), (False, True), (True, True)])
def test_cache_status_and_exact_provenance(settings, kind, geo_cached, data_cached):
    fetch, args, url, geo, payload, extra, _, _ = source_case(kind)
    request_url = url + "?recorded=provenance"
    session = FakeSession({
        GEO_URL: FakeResponse(geo, from_cache=geo_cached),
        url: FakeResponse(payload, url=request_url, from_cache=data_cached),
        **{u: FakeResponse(p, from_cache=data_cached) for u, p in extra.items()},
    })
    result = fetch(args, WINDOW, session, settings)
    assert result.cache_hit is (geo_cached and data_cached)
    assert result.http_calls == len(session.calls) == 2 + len(extra)
    assert result.series.url == request_url


@pytest.mark.parametrize("kind", ["weather", "air"])
@pytest.mark.parametrize("stage", ["geocode", "data"])
@pytest.mark.parametrize("failure", ["http", "connection"])
def test_request_failure_wrapped(settings, kind, stage, failure):
    fetch, args, url, geo, payload, extra, _, _ = source_case(kind)
    routes = {GEO_URL: geo, url: payload, **extra}
    failing_url = GEO_URL if stage == "geocode" else url
    if failure == "http":
        routes[failing_url] = FakeResponse({}, status_code=503)
    else:
        del routes[failing_url]
    with pytest.raises(SourceError, match="request failed") as error:
        fetch(args, WINDOW, FakeSession(routes), settings)
    assert error.value.__cause__ is not None


@pytest.mark.parametrize("kind", ["weather", "air"])
@pytest.mark.parametrize("fault", ["missing", "length", "date", "null_date", "block", "payload"])
def test_malformed_data_wrapped(settings, kind, fault):
    fetch, args, url, geo, payload, extra, frequency, field = source_case(kind)
    if fault == "missing":
        del payload[frequency][field]
    elif fault == "length":
        payload[frequency][field].pop()
    elif fault == "date":
        payload[frequency]["time"][0] = "not-a-date"
    elif fault == "null_date":
        payload[frequency]["time"][0] = None
    elif fault == "block":
        payload[frequency] = None
    else:
        payload = []
    with pytest.raises(SourceError, match="Open-Meteo returned malformed data"):
        fetch(args, WINDOW, FakeSession({GEO_URL: geo, url: payload, **extra}), settings)


@pytest.mark.parametrize("kind", ["weather", "air"])
@pytest.mark.parametrize("remaining_days", [0, 4, 5])
def test_minimum_history_points(settings, kind, remaining_days):
    fetch, args, url, geo, payload, extra, frequency, field = source_case(kind)
    times = payload[frequency]["time"]
    history_days = sorted({t[:10] for t in times if str(WINDOW.start) <= t[:10] <= str(WINDOW.end)})
    retained = set(history_days[:remaining_days])
    for i, timestamp in enumerate(times):
        if timestamp[:10] not in retained and timestamp[:10] <= str(WINDOW.end):
            payload[frequency][field][i] = None
    session = FakeSession({GEO_URL: geo, url: payload, **extra})
    if remaining_days < 5:
        with pytest.raises(SourceError, match=f"only {remaining_days} days available"):
            fetch(args, WINDOW, session, settings)
    else:
        assert len(fetch(args, WINDOW, session, settings).series.history) == 5


def test_air_quality_past_days_capped(settings):
    fetch, args, url, geo, payload, extra, _, _ = source_case("air")
    session = FakeSession({GEO_URL: geo, url: payload})
    fetch(args, Window(date(2026, 6, 1), WINDOW.end), session, settings)
    assert session.calls[1][1]["past_days"] == 92


def test_weather_history_wins_over_forecast_overlap(settings):
    # If the forecast endpoint also returns a day inside the window, the historical value is kept.
    history, forecast = load_fixture("weather_rome_history.json"), load_fixture("weather_rome_forecast.json")
    forecast["daily"]["time"].insert(0, "2026-10-13")
    forecast["daily"]["temperature_2m_mean"].insert(0, -99.0)
    series = fetch_weather(WeatherArgs(city="Rome", variable="temperature"), WINDOW,
                           weather_session(history, forecast), settings).series
    assert series.history.loc["2026-10-13"] == pytest.approx(history["daily"]["temperature_2m_mean"][-1])


@pytest.mark.parametrize("model,kwargs", [
    (WeatherArgs, {"city": "Rome", "variable": "humidity"}),
    (AirQualityArgs, {"city": "Milan", "pollutant": "co2"}),
])
def test_unsupported_arguments_rejected(model, kwargs):
    with pytest.raises(ValidationError):
        model(**kwargs)
