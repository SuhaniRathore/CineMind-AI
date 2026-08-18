"""
Application configuration.

All configuration values are loaded from environment variables (via a .env
file in local development). Nothing sensitive is hardcoded here — see
DEVELOPMENT_RULES.md, rule 13-14.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings, populated from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    app_name: str = "CineMind AI"
    environment: str = "development"
    debug: bool = True

    # API
    api_v1_prefix: str = "/api/v1"

    # CORS (kept permissive-but-explicit for local dev; tightened later)
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Database (placeholder only — no DB connection is made in Milestone 1)
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/cinemind"

    # Auth (placeholder only — no auth logic exists yet)
    jwt_secret_key: str = "changeme-in-env-file"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # External services (placeholder only — not used until later milestones)
    tmdb_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance so env vars are parsed only once."""
    return Settings()
