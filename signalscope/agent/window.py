"""Deterministic time-window parsing.

Small LLMs get date arithmetic wrong (measured: "last 60 days" became 30 with llama3.2:3b),
so the window is parsed from the question by code and never left to the model.
"""

from __future__ import annotations

import re
from datetime import date

from signalscope.config import Settings
from signalscope.sources.base import Window

_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "eight": 8, "ten": 10, "twelve": 12, "a": 1}

_COUNT = re.compile(
    r"\b(?:last|past|previous|over the last|over the past|in the last|in the past)\s+"
    r"(\d+|one|two|three|four|five|six|eight|ten|twelve)\s*(day|week|month|year)s?\b", re.I)
_SINGLE = re.compile(r"\b(?:last|past|this)\s+(week|month|quarter|year)\b", re.I)
_SHORT = re.compile(r"\b(today|yesterday|tonight|right now)\b", re.I)


def parse_window(question: str, settings: Settings, today: date | None = None) -> tuple[Window, str | None]:
    """Return the analysis window and an optional note explaining a default or a clamp."""
    days: int | None = None
    if m := _COUNT.search(question):
        count = m.group(1).lower()
        days = (int(count) if count.isdigit() else _WORD_NUMBERS[count]) * _UNIT_DAYS[m.group(2).lower()]
    elif m := _SINGLE.search(question):
        days = {"week": 7, "month": 30, "quarter": 90, "year": 365}[m.group(1).lower()]
    elif _SHORT.search(question):
        days = 1

    note: str | None = None
    if days is None:
        days = settings.default_days
        note = f"No time range in the question: using the last {days} days."
    elif days < settings.min_days:
        note = (f"Requested {days} day(s) is too short for a correlation; "
                f"using the last {settings.min_days} days.")
        days = settings.min_days
    elif days > settings.max_days:
        note = f"Requested {days} days; data sources allow up to {settings.max_days}, so using {settings.max_days}."
        days = settings.max_days
    return Window.last_n_days(days, today=today), note
