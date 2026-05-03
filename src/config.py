"""Application configuration — loaded from environment via pydantic-settings.

Uses a singleton pattern so the same Settings instance is shared across
the entire application. Environment variables are read from a .env file
at the project root (via python-dotenv) and can be overridden by real
environment variables.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Valura AI microservice.

    All values have sensible defaults except ``openai_api_key``, which
    must be provided via the ``OPENAI_API_KEY`` environment variable or
    ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ------------------------------------------------------------------
    openai_api_key: str = ""
    classifier_model: str = "gpt-4o-mini"
    eval_model: str = "gpt-4.1"

    # --- Persistence -----------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./valura.db"

    # --- Pipeline --------------------------------------------------------------
    pipeline_timeout_seconds: int = 30
    max_session_history_turns: int = 5

    # --- Observability ---------------------------------------------------------
    log_level: str = "INFO"
    environment: str = "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance.

    The first call reads from the environment / ``.env``; subsequent
    calls return the cached object.  Call ``get_settings.cache_clear()``
    in tests if you need a fresh instance.
    """
    return Settings()
