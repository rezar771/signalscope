"""Runtime settings, read once from environment variables (and `.env` if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

LLMMode = Literal["auto", "cloud", "local"]


@dataclass(frozen=True)
class Settings:
    # LLM cascade
    llm_mode: LLMMode = "auto"
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.2:3b"
    llm_timeout_s: float = 60.0

    # HTTP / data sources
    http_timeout_s: float = 45.0        # Energy-Charts takes up to ~16 s for 90 days (measured)
    cache_path: Path = Path(".cache/http")
    cache_ttl_s: int = 3600
    user_agent: str = "SignalScope/0.1 (interview POC; https://github.com/)"

    # Analysis window (days, ending yesterday)
    default_days: int = 90
    min_days: int = 14
    max_days: int = 90

    @property
    def cloud_enabled(self) -> bool:
        return self.llm_mode in ("auto", "cloud") and bool(self.openrouter_api_key)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(override=False)
        env = os.environ
        mode = env.get("LLM_MODE", "auto").lower()
        if mode not in ("auto", "cloud", "local"):
            raise ValueError(f"LLM_MODE must be auto, cloud or local, got '{mode}'")
        defaults = cls()
        return cls(
            llm_mode=mode,  # type: ignore[arg-type]
            openrouter_api_key=env.get("OPENROUTER_API_KEY") or None,
            openrouter_model=env.get("OPENROUTER_MODEL", defaults.openrouter_model),
            ollama_base_url=env.get("OLLAMA_URL", defaults.ollama_base_url),
            ollama_model=env.get("OLLAMA_MODEL", defaults.ollama_model),
            llm_timeout_s=float(env.get("LLM_TIMEOUT_S", defaults.llm_timeout_s)),
            http_timeout_s=float(env.get("HTTP_TIMEOUT_S", defaults.http_timeout_s)),
            cache_path=Path(env.get("CACHE_PATH", str(defaults.cache_path))),
            cache_ttl_s=int(env.get("CACHE_TTL_S", defaults.cache_ttl_s)),
        )
