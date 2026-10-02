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

    # External services — TMDB integration active from Milestone 4.
    # API key loaded exclusively from env var; never hardcoded (DEVELOPMENT_RULES.md rule 13).
    tmdb_api_key: str = ""
    tmdb_base_url: str = "https://api.themoviedb.org/3"
    # Number of TMDB popular-movie pages fetched on each sync (each page ≈ 20 movies)
    tmdb_sync_pages: int = 5

    # ---------------------------------------------------------------------------
    # Milestone 12 — Data Pipeline & ML Foundation settings
    # ---------------------------------------------------------------------------

    # Path to the directory where raw dataset files are stored.
    # In Docker this should be a volume mount (e.g. /app/data).
    # Locally defaults to a 'data' sibling of the backend package root.
    data_dir: str = "data"

    # Path to the directory where trained model artifacts (.pkl) are saved.
    # Subdirectory of data_dir by default; can be overridden independently.
    models_dir: str = "data/models"

    # Secret token required to call /admin/* endpoints.
    # Change this in .env — it is a bearer secret, not a JWT.
    admin_api_secret: str = "changeme-admin-secret"

    # MovieLens dataset variant to download/expect.
    # Defaults to 'ml-latest-small' for fast local development and testing.
    # For actual M14 deep-learning NCF training, override to 'ml-25m' in your .env file
    # to provide sufficient training data.
    movielens_variant: str = "ml-latest-small"

    # Minimum number of in-app ratings a user must have before the
    # collaborative filter is used (below this → content-based only).
    collab_min_ratings: int = 3

    # Hybrid blend weight for content-based score (0.0–1.0).
    # collaborative weight = 1.0 - hybrid_content_weight
    hybrid_content_weight: float = 0.4

    # ---------------------------------------------------------------------------
    # Milestone 14 — NCF (Neural Collaborative Filtering) settings
    # ---------------------------------------------------------------------------

    # Set to True in .env to enable NCF training alongside the SVD model.
    # When True, the NCF model is preferred over SVD in HybridRecommender
    # if its artifact is present.
    use_deep_collab: bool = False

    # NCF training epochs (early-stop may terminate before this limit).
    ncf_epochs: int = 20

    # Mini-batch size for NCF training.
    # 1024 works well for ml-25m on 8 GB RAM; reduce if OOM.
    ncf_batch_size: int = 1024

    # NCF embedding dimensions and MLP hidden layer widths.
    # These match the plan defaults and can be overridden via .env.
    ncf_gmf_dim: int = 64
    ncf_mlp_layers: list[int] = [128, 64, 32, 16]
    ncf_dropout: float = 0.2
    ncf_lr: float = 1e-3
    ncf_weight_decay: float = 1e-5
    ncf_early_stop_patience: int = 3


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance so env vars are parsed only once."""
    return Settings()
