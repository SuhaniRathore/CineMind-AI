"""
Pydantic response schemas for the movies endpoints (Milestone 4).

All API responses use these schemas — no raw ORM objects are ever returned
directly (DEVELOPMENT_RULES.md rule 12).

Design notes:
  - poster_url is a computed field that constructs the full TMDB image URL
    from the stored poster_path.  The base URL is intentionally kept
    simple (w500 size) and can be made configurable later.
  - overview_embedding is absent — that column doesn't exist until V2.
  - All schemas use model_config = ConfigDict(from_attributes=True) so they
    can be constructed directly from SQLAlchemy ORM instances.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, computed_field

# TMDB image base URL — public CDN, not a secret.
_TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


class GenreOut(BaseModel):
    """Genre as returned in API responses."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    tmdb_genre_id: int
    name: str


class MovieOut(BaseModel):
    """
    Full movie record as returned by GET /movies/{id} and list endpoints.

    poster_url is derived from poster_path — it is None when poster_path is
    not available rather than returning a broken URL.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    tmdb_id: int
    title: str
    overview: str | None
    release_date: date | None
    runtime: int | None
    poster_path: str | None
    original_language: str | None
    popularity: float | None
    vote_average: float | None
    vote_count: int | None
    cached_at: datetime
    genres: list[GenreOut] = []

    @computed_field  # type: ignore[misc]
    @property
    def poster_url(self) -> str | None:
        """Full TMDB image URL, or None if no poster is available."""
        if self.poster_path:
            return f"{_TMDB_IMAGE_BASE}{self.poster_path}"
        return None


class MovieListResponse(BaseModel):
    """Paginated list of movies."""

    movies: list[MovieOut]
    total: int
    limit: int
    offset: int


class SyncResult(BaseModel):
    """Result payload returned by POST /movies/sync."""

    genres_synced: int
    movies_synced: int
    errors: int
    message: str
