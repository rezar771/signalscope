"""Tool definitions and argument validation shared by the planner and data fetcher.

Example values in descriptions deliberately reuse the planner prompt's examples (London, Paris,
Umbrella): models sometimes copy examples, and the planner's grounding check rejects exactly those.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import requests
from pydantic import BaseModel, ValidationError

from signalscope.config import Settings
from signalscope.sources.base import FetchResult, ToolCallError, Window
from signalscope.sources.electricity import ElectricityArgs, fetch_electricity
from signalscope.sources.fx import FxArgs, fetch_exchange_rate
from signalscope.sources.weather import AirQualityArgs, WeatherArgs, fetch_air_quality, fetch_weather
from signalscope.sources.wikipedia import WikipediaArgs, fetch_wikipedia_interest


@dataclass(frozen=True)
class Tool:
    """One planner-visible tool and the source adapter that implements it."""

    name: str
    description: str
    args_model: type[BaseModel]
    fetch: Callable[[BaseModel, Window, requests.Session, Settings], FetchResult]

    def schema(self) -> dict:
        """Produce a compact OpenAI-compatible schema from the flat argument model."""
        model_schema = self.args_model.model_json_schema()
        # The argument models use primitives and Literal enums, already inline in Pydantic.
        properties = {
            name: {key: value for key, value in field.items() if key != "title"}
            for name, field in model_schema["properties"].items()
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": properties,
                               "required": model_schema.get("required", [])},
            },
        }


TOOLS: dict[str, Tool] = {
    "get_weather": Tool(
        "get_weather", "Daily weather in a city: temperature, precipitation, wind_speed or sunshine. "
        "Use for weather conditions; example: city='London', variable='precipitation'.",
        WeatherArgs, fetch_weather,
    ),
    "get_air_quality": Tool(
        "get_air_quality", "Daily air pollution in a city: pm2_5, pm10, nitrogen_dioxide or ozone. "
        "Use for pollution questions; example: city='Paris', pollutant='nitrogen_dioxide'.",
        AirQualityArgs, fetch_air_quality,
    ),
    "get_wikipedia_interest": Tool(
        "get_wikipedia_interest", "Daily Wikipedia page views of a topic, a proxy for public interest. "
        "Use for attention to a topic; example: topic='Umbrella', language='en'.",
        WikipediaArgs, fetch_wikipedia_interest,
    ),
    "get_exchange_rate": Tool(
        "get_exchange_rate", "Daily currency exchange rate from ECB reference rates. "
        "Use for supported national currencies, not cryptocurrency; example: base='EUR', quote='USD'.",
        FxArgs, fetch_exchange_rate,
    ),
    "get_electricity": Tool(
        "get_electricity", "Daily electricity load (demand), solar or wind generation of a European country. "
        "Use for power demand or generation; example: country='fr', metric='load'.",
        ElectricityArgs, fetch_electricity,
    ),
}


def tool_schemas() -> list[dict]:
    """Return schemas in the stable registry order used in the planner prompt."""
    return [tool.schema() for tool in TOOLS.values()]


def validate_call(name: str, raw_args: dict | str) -> BaseModel:
    """Parse and normalize arguments, giving the planner actionable repair errors."""
    if name not in TOOLS:
        raise ToolCallError(f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}.")
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise ToolCallError(f"Invalid JSON arguments for '{name}': {exc.msg}.") from exc
    if not isinstance(raw_args, dict):
        raise ToolCallError(f"Arguments for '{name}' must be a JSON object with named fields.")
    tool = TOOLS[name]
    try:
        return tool.args_model.model_validate(raw_args)
    except ValidationError as exc:
        fields = tool.schema()["function"]["parameters"]["properties"]
        errors = []
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            message = f"{field}: {error['msg']}"
            allowed = fields.get(field, {}).get("enum")
            if allowed and error["type"] != "literal_error":
                message += f". Allowed values: {', '.join(allowed)}"
            errors.append(message)
        raise ToolCallError(f"Invalid arguments for '{name}': {'; '.join(errors)}") from exc
