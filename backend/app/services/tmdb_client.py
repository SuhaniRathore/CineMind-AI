"""
TMDB API client (server-side only).

Architecture rule (ARCHITECTURE.md):
  - TMDB calls originate ONLY from the backend; the API key is never exposed
    to the frontend or included in any response payload.
  - This module has no FastAPI imports — it is a pure service module that
    can be tested independently of the HTTP layer.

Security rule (DEVELOPMENT_RULES.md rule 13):
  - The API key is read from Settings (environment variable); it is never
    hardcoded here or logged.

Usage::

    from app.core.config import get_settings
    from app.services.tmdb_client import TMDBClient

    client = TMDBClient(settings=get_settings())
    genres = client.get_genres()
    movie  = client.get_movie_details(tmdb_id=550)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)


class TMDBError(Exception):
    """
    Raised when a TMDB API call fails (HTTP error or network issue).

    The original httpx exception is chained via ``raise ... from exc`` so
    callers can inspect it if needed, while this class gives a clean API
    boundary — callers never need to import httpx just to catch errors.

    The API key is intentionally NOT included in this exception's message
    (see DEVELOPMENT_RULES.md rule 17 — no secrets in logs).
    """


class TMDBClient:
    """
    Thin wrapper around the TMDB v3 REST API.

    All methods are synchronous (matching the sync SQLAlchemy stack).
    Each method call opens and closes its own httpx connection by default;
    an httpx.Client can be injected via ``http_client`` for testing or
    connection-pool reuse.

    Args:
        settings: Application settings instance — provides base URL and API key.
        http_client: Optional pre-built httpx.Client (use in tests for mocking).
    """

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not settings.tmdb_api_key:
            logger.warning(
                "TMDB_API_KEY is not set — TMDB calls will fail. "
                "Add TMDB_API_KEY to your .env file."
            )
        self._base_url: str = settings.tmdb_base_url.rstrip("/")
        # API key stored as a private attribute; never exposed via __repr__ or logging
        self.__api_key: str = settings.tmdb_api_key
        self._http_client: httpx.Client | None = http_client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """
        Make a GET request to TMDB, merge the api_key param, and return the
        parsed JSON body.

        Raises:
            TMDBError: On any HTTP error (4xx, 5xx) or connection failure.
        """
        merged_params: dict[str, Any] = {"api_key": self.__api_key}
        if params:
            merged_params.update(params)

        url = f"{self._base_url}/{path.lstrip('/')}"

        try:
            if self._http_client is not None:
                response = self._http_client.get(url, params=merged_params)
            else:
                with httpx.Client(timeout=10.0) as client:
                    response = client.get(url, params=merged_params)

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as exc:
            # Log WITHOUT the full URL (which contains the API key as a query
            # param) — log only the status code and clean path.
            logger.error(
                "TMDB request failed: %s %s → HTTP %s",
                exc.request.method,
                path,
                exc.response.status_code,
            )
            raise TMDBError(
                f"TMDB returned HTTP {exc.response.status_code} for {path}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error("TMDB request error on %s: %s", path, type(exc).__name__)
            raise TMDBError(
                f"TMDB request failed for {path}: {type(exc).__name__}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def get_genres(self) -> list[dict[str, Any]]:
        """
        Fetch the official TMDB genre list for movies.

        Returns:
            List of dicts with keys ``id`` (TMDB genre id) and ``name``.

        Example::

            [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, ...]
        """
        data = self._get("/genre/movie/list")
        return data.get("genres", [])

    def get_popular_movies(self, page: int = 1) -> dict[str, Any]:
        """
        Fetch a page of currently popular movies from TMDB.

        Args:
            page: Page number (1-indexed, TMDB max 500).

        Returns:
            TMDB response dict with keys ``results``, ``page``,
            ``total_pages``, ``total_results``.
        """
        return self._get("/movie/popular", params={"page": page})

    def search_movies(self, query: str, page: int = 1) -> dict[str, Any]:
        """
        Search TMDB for movies matching *query*.

        Args:
            query: Search string (title / keywords).
            page: Page number (1-indexed).

        Returns:
            TMDB response dict with keys ``results``, ``page``,
            ``total_pages``, ``total_results``.
        """
        return self._get("/search/movie", params={"query": query, "page": page})

    def get_movie_details(self, tmdb_id: int) -> dict[str, Any]:
        """
        Fetch full details for a single movie by its TMDB id.

        Args:
            tmdb_id: The TMDB integer movie id.

        Returns:
            TMDB movie detail dict (includes ``genres`` list).

        Raises:
            TMDBError: If the movie is not found (404) or any other error.
        """
        return self._get(f"/movie/{tmdb_id}")
