"""Offline registry contracts, normalization and planner repair integration."""

import json

import pytest

from signalscope.agent.llm import CascadeLLM
from signalscope.agent.planner import plan
from signalscope.sources.base import ToolCallError
from signalscope.sources.electricity import ElectricityArgs, fetch_electricity
from signalscope.sources.fx import ECB_CURRENCIES, FxArgs, fetch_exchange_rate
from signalscope.sources.registry import TOOLS, tool_schemas, validate_call
from signalscope.sources.weather import AirQualityArgs, WeatherArgs, fetch_air_quality, fetch_weather
from signalscope.sources.wikipedia import WikipediaArgs, fetch_wikipedia_interest
from signalscope.trace import Tracer
from test_agent import FakeClient, reply


def test_tool_bindings_and_flat_schemas():
    expected = {
        "get_weather": (WeatherArgs, fetch_weather),
        "get_air_quality": (AirQualityArgs, fetch_air_quality),
        "get_wikipedia_interest": (WikipediaArgs, fetch_wikipedia_interest),
        "get_exchange_rate": (FxArgs, fetch_exchange_rate),
        "get_electricity": (ElectricityArgs, fetch_electricity),
    }
    schemas = tool_schemas()
    assert list(TOOLS) == list(expected)
    assert [item["function"]["name"] for item in schemas] == list(expected)
    for schema, (name, (model, fetch)) in zip(schemas, expected.items()):
        tool = TOOLS[name]
        assert (tool.name, tool.args_model, tool.fetch) == (name, model, fetch)
        assert schema["type"] == "function"
        assert schema["function"]["description"] == tool.description
        assert "example:" in tool.description
        parameters = schema["function"]["parameters"]
        original = model.model_json_schema()
        assert parameters["type"] == "object"
        assert parameters["required"] == original["required"]
        assert set(parameters["properties"]) == set(original["properties"])
        assert not any(token in json.dumps(schema) for token in ('"title"', '"$defs"', '"$ref"'))
        for field, prop in parameters["properties"].items():
            assert prop["type"] == "string"
            if "enum" in original["properties"][field]:
                assert prop["enum"] == original["properties"][field]["enum"]
    assert schemas[2]["function"]["parameters"]["properties"]["language"]["default"] == "en"
    assert schemas[3]["function"]["parameters"]["properties"]["base"]["default"] == "EUR"


def test_valid_call_accepts_dict_or_json_without_mutating_input():
    raw = {"city": "  Rome  ", "variable": " Temperature "}
    direct = validate_call("get_weather", raw)
    encoded = validate_call("get_weather", json.dumps(raw))
    assert isinstance(direct, WeatherArgs)
    assert direct == encoded == WeatherArgs(city="Rome", variable="temperature")
    assert raw == {"city": "  Rome  ", "variable": " Temperature "}
    assert validate_call("get_wikipedia_interest", {"topic": "  Ice cream  "}).language == "en"


@pytest.mark.parametrize("tool,raw,expected", [
    ("get_weather", {"city": "Rome", "variable": " RAIN "}, {"city": "Rome", "variable": "precipitation"}),
    ("get_air_quality", {"city": " Milan ", "pollutant": " PM2.5 "}, {"city": "Milan", "pollutant": "pm2_5"}),
    ("get_electricity", {"country": " Germany ", "metric": " DEMAND "}, {"country": "de", "metric": "load"}),
    ("get_exchange_rate", {"base": " euro ", "quote": " Dollar "}, {"base": "EUR", "quote": "USD"}),
    ("get_wikipedia_interest", {"topic": " Bitcoin ", "language": " EN "}, {"topic": "Bitcoin", "language": "en"}),
])
def test_synonym_groups(tool, raw, expected):
    assert validate_call(tool, raw).model_dump() == expected


def test_unknown_tool_lists_available_tools():
    with pytest.raises(ToolCallError, match="Unknown tool 'get_crypto'") as error:
        validate_call("get_crypto", {})
    assert all(name in str(error.value) for name in TOOLS)


@pytest.mark.parametrize("raw", ['{"city":', '[]', None])
def test_malformed_arguments_raise_shared_error(raw):
    with pytest.raises(ToolCallError, match="JSON"):
        validate_call("get_weather", raw)


def test_enum_and_required_field_errors_are_actionable():
    with pytest.raises(ToolCallError) as error:
        validate_call("get_weather", {"variable": "humidity"})
    message = str(error.value)
    assert "city" in message and "Field required" in message
    assert "variable" in message
    assert all(value in message for value in ("temperature", "precipitation", "wind_speed", "sunshine"))


@pytest.mark.parametrize("tool,raw,field", [
    ("get_weather", {"city": " \t", "variable": "sun"}, "city"),
    ("get_air_quality", {"city": "", "pollutant": "no2"}, "city"),
    ("get_wikipedia_interest", {"topic": "  "}, "topic"),
])
def test_blank_names_rejected(tool, raw, field):
    with pytest.raises(ToolCallError) as error:
        validate_call(tool, raw)
    assert field in str(error.value) and "must not be empty" in str(error.value)


def test_non_string_argument_gets_validation_error():
    with pytest.raises(ToolCallError, match="city"):
        validate_call("get_weather", {"city": 12, "variable": ["temperature"]})


@pytest.mark.parametrize("field", ["base", "quote"])
def test_unsupported_currency_rejected_before_fetch(field):
    raw = {"base": "EUR", "quote": "USD", field: "BTC"}
    with pytest.raises(ToolCallError) as error:
        validate_call("get_exchange_rate", raw)
    message = str(error.value)
    assert field in message and "Unsupported currency 'BTC'" in message
    assert "Allowed codes:" in message
    assert all(code in message for code in ECB_CURRENCIES)


def test_ecb_currency_set_and_default_base():
    assert ECB_CURRENCIES == set(
        "AUD BRL CAD CHF CNY CZK DKK EUR GBP HKD HUF IDR ILS INR ISK JPY KRW "
        "MXN MYR NOK NZD PHP PLN RON SEK SGD THB TRY USD ZAR".split()
    )
    assert validate_call("get_exchange_rate", {"quote": "yen"}).model_dump() == {"base": "EUR", "quote": "JPY"}


def test_real_registry_triggers_planner_currency_repair():
    wiki = ("get_wikipedia_interest", {"topic": "Bitcoin"})
    invalid = ("get_exchange_rate", {"base": "EUR", "quote": "BTC"})
    repaired = ("get_exchange_rate", {"base": "euro", "quote": "dollar"})
    client = FakeClient("local", [reply("local", [invalid, wiki]), reply("local", [repaired, wiki])])
    result = plan(
        "Is the EUR/USD exchange rate related to interest in Bitcoin?",
        CascadeLLM([client]), Tracer(), tool_schemas(), validate_call,
    )
    assert result.llm_calls == 2
    assert [call.tool for call in result.calls] == ["get_exchange_rate", "get_wikipedia_interest"]
    assert result.calls[0].args.model_dump() == {"base": "EUR", "quote": "USD"}
    repair_prompt = client.received[1][-1]["content"]
    assert "quote" in repair_prompt and "Allowed codes:" in repair_prompt
