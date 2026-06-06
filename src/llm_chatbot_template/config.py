"""Application settings, loaded from the environment (and an optional ``.env``).

Settings are read once via :func:`get_settings`. By default the app runs with the
offline :class:`~providers.FakeLLM` so no API keys are required. Set
``CHATBOT_PROVIDER=openai`` (plus ``CHATBOT_OPENAI_API_KEY``) to use a real
OpenAI-compatible backend.

Part of LLM Chatbot Template by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the chatbot server."""

    model_config = SettingsConfigDict(
        env_prefix="CHATBOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Literal["fake", "openai"] = "fake"
    system_prompt: str = "You are a helpful, concise assistant."
    max_tokens: int = 1024

    # OpenAI-compatible backend (only used when provider == "openai").
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    openai_base_url: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance built from the environment."""
    return Settings()
