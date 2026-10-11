"""
Movies API router — Milestone 4.

Endpoints:
  GET  /api/v1/movies              — paginated list with optional filters
  GET  /api/v1/movies/search       — title/overview keyword search
  GET  /api/v1/movies/{id}         — single movie detail
  POST /api/v1/movies/sync         — trigger TMDB genre + popular-movie sync

Architecture rules (ARCHITECTURE.md / DEVELOPMENT_RULES.md rule 9):
  - Routes are thin: they validate input, call the service layer, and
    return a Pydantic response model.  No business logic lives here.
  - No authentication yet (Milestone 5) — all endpoints are public for now.
  - TMDB client is constructed from settings injected by FastAPI — the API
    key never appears in route code.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.database import get_db
from app.schemas.movie import MovieListResponse, MovieOut, SyncResult
from app.services import movie_service
from app.services.tmdb_client import TMDBClient, TMDBError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/movies", tags=["movies"])


# ---------------------------------------------------------------------------
# Dependency: build a TMDBClient from settings
# ---------------------------------------------------------------------------


def get_tmdb_client(settings: Settings = Depends(get_settings)) -> TMDBClient:
    """FastAPI dependency — construct a TMDBClient from application settings."""
    return TMDBClient(settings=settings)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=MovieListResponse, summary="List movies")
def list_movies(
    genre_id: int | None = Query(default=None, description="Filter by Genre DB id"),
    sort_by: str = Query(
        default="popularity",
        description="Sort field: popularity | vote_average | release_date | title",
    ),
    limit: int = Query(default=20, ge=1, le=100, description="Results per page"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
) -> MovieListResponse:
    """
    Return a paginated list of movies from our local cache.

    Use `genre_id` to filter by genre (Genre DB primary key) and
    `sort_by` to control ordering.  The DB is seeded via
    `POST /movies/sync`.
    """
    movies, total = movie_service.list_movies(
        db, genre_id=genre_id, sort_by=sort_by, limit=limit, offset=offset
    )
    return MovieListResponse(
        movies=[MovieOut.model_validate(m) for m in movies],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/search", response_model=MovieListResponse, summary="Search movies")
def search_movies(
    q: str = Query(..., min_length=1, description="Search term (title / overview)"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> MovieListResponse:
    """
    Case-insensitive keyword search against titles and overviews in our local
    movie cache.  Semantic search is deferred to V2 (Milestone 15).
    """
    movies, total = movie_service.search_movies_in_db(
        db, q, limit=limit, offset=offset
    )
    return MovieListResponse(
        movies=[MovieOut.model_validate(m) for m in movies],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{movie_id}", response_model=MovieOut, summary="Get movie detail")
def get_movie(
    movie_id: int,
    db: Session = Depends(get_db),
) -> MovieOut:
    """
    Return the full details for a single movie by its **internal DB id**
    (not the TMDB id).

    Returns 404 if the movie has not been synced into our cache yet.
    """
    movie = movie_service.get_movie_by_id(db, movie_id)
    if movie is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Movie with id={movie_id} not found.",
        )
    return MovieOut.model_validate(movie)


@router.post(
    "/sync",
    response_model=SyncResult,
    status_code=status.HTTP_200_OK,
    summary="Sync movies from TMDB",
)
def sync_from_tmdb(
    db: Session = Depends(get_db),
    tmdb_client: TMDBClient = Depends(get_tmdb_client),
    settings: Settings = Depends(get_settings),
) -> SyncResult:
    """
    Trigger a sync of genres and popular movies from TMDB into our local DB.

    - First syncs the genre list (upsert).
    - Then fetches `TMDB_SYNC_PAGES` pages of popular movies (default 5).

    This endpoint is idempotent — calling it multiple times is safe; existing
    records are updated rather than duplicated.

    **Note**: Authentication will be added in Milestone 5.  For now this
    endpoint is public (admin-only protection is a future concern).
    """
    if not settings.tmdb_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TMDB_API_KEY is not configured on the server.",
        )

    try:
        genres_synced = movie_service.sync_genres(db, tmdb_client)
    except TMDBError as exc:
        logger.error("Genre sync failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch genres from TMDB.",
        ) from exc

    movies_synced, errors = movie_service.sync_movies_from_popular(
        db, tmdb_client, pages=settings.tmdb_sync_pages
    )

    return SyncResult(
        genres_synced=genres_synced,
        movies_synced=movies_synced,
        errors=errors,
        message=(
            f"Sync complete. {genres_synced} genre(s), "
            f"{movies_synced} movie(s) upserted, {errors} error(s)."
        ),
    )
