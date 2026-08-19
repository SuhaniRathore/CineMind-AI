"""
Movie service — business logic layer for Milestone 4.

Responsibilities:
  - Upsert genres and movies from TMDB data into our PostgreSQL cache.
  - Provide query functions used by the movies API router.

Architecture rules (ARCHITECTURE.md / DEVELOPMENT_RULES.md rule 9):
  - No FastAPI imports here; this module is purely business logic.
  - Database access goes through the SQLAlchemy Session passed in by the
    caller (dependency injection keeps this testable).
  - TMDB client is also injected, so tests can pass a mock.

All field names from TMDB API responses are mapped here — routers and
schemas never touch raw TMDB dicts directly.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.models.genre import Genre
from app.models.movie import Movie
from app.services.tmdb_client import TMDBClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-mapping helpers (TMDB dict → ORM fields)
# ---------------------------------------------------------------------------


def _parse_release_date(raw: str | None) -> date | None:
    """Parse a TMDB release_date string ('YYYY-MM-DD') to a Python date."""
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        logger.debug("Could not parse TMDB release_date: %r", raw)
        return None


def _tmdb_movie_to_fields(tmdb_data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the fields we store from a TMDB movie dict (either from the
    popular/search results or from a full detail response).

    Returns a dict ready to be used in Movie(...) or setattr() calls.
    """
    return {
        "tmdb_id": tmdb_data["id"],
        "title": tmdb_data.get("title") or tmdb_data.get("name", ""),
        "overview": tmdb_data.get("overview") or None,
        "release_date": _parse_release_date(tmdb_data.get("release_date")),
        "runtime": tmdb_data.get("runtime"),  # only present in detail endpoint
        "poster_path": tmdb_data.get("poster_path") or None,
        "original_language": tmdb_data.get("original_language") or None,
        "popularity": tmdb_data.get("popularity"),
        "vote_average": tmdb_data.get("vote_average"),
        "vote_count": tmdb_data.get("vote_count"),
        "cached_at": datetime.now(tz=timezone.utc),
    }


# ---------------------------------------------------------------------------
# Genre sync
# ---------------------------------------------------------------------------


def sync_genres(db: Session, tmdb_client: TMDBClient) -> int:
    """
    Fetch the TMDB genre list and upsert into our genres table.

    An upsert here means: create if not present (by tmdb_genre_id),
    update name if it changed (TMDB genre names rarely change, but let's
    be correct).

    Returns:
        Number of genres upserted (created or updated).
    """
    tmdb_genres = tmdb_client.get_genres()
    count = 0
    for g in tmdb_genres:
        tmdb_genre_id: int = g["id"]
        name: str = g["name"]

        existing = db.query(Genre).filter_by(tmdb_genre_id=tmdb_genre_id).first()
        if existing is None:
            db.add(Genre(tmdb_genre_id=tmdb_genre_id, name=name))
            count += 1
        elif existing.name != name:
            existing.name = name
            count += 1

    db.commit()
    logger.info("sync_genres: %d genre(s) upserted", count)
    return count


# ---------------------------------------------------------------------------
# Movie sync / upsert
# ---------------------------------------------------------------------------


def upsert_movie_from_tmdb(
    db: Session,
    tmdb_data: dict[str, Any],
) -> Movie:
    """
    Create or update a Movie row from a TMDB movie dict.

    Also links the movie's genres (if the TMDB dict includes a `genres`
    key — present in detail responses, not in list/search responses).

    Args:
        db: Active SQLAlchemy session.
        tmdb_data: Raw dict from TMDB API (popular list item or detail).

    Returns:
        The created or updated Movie ORM instance.
    """
    fields = _tmdb_movie_to_fields(tmdb_data)
    tmdb_id: int = fields["tmdb_id"]

    movie = db.query(Movie).filter_by(tmdb_id=tmdb_id).first()
    if movie is None:
        movie = Movie(**fields)
        db.add(movie)
    else:
        for attr, value in fields.items():
            setattr(movie, attr, value)

    # Link genres only if the TMDB response includes them (detail endpoint)
    if "genres" in tmdb_data:
        linked_genres: list[Genre] = []
        for g in tmdb_data["genres"]:
            genre = db.query(Genre).filter_by(tmdb_genre_id=g["id"]).first()
            if genre:
                linked_genres.append(genre)
        movie.genres = linked_genres

    db.flush()
    return movie


def sync_movies_from_popular(
    db: Session,
    tmdb_client: TMDBClient,
    pages: int = 5,
) -> tuple[int, int]:
    """
    Pull `pages` pages of popular movies from TMDB and upsert them all.

    Each TMDB page contains up to 20 movies.  The popular-list response does
    NOT include genres for each movie, so genre_ids is mapped to whatever
    Genre rows we already have (from sync_genres).  We intentionally do NOT
    make a detail call per movie here — that would mean 100 extra HTTP
    requests and is unnecessarily slow for a bulk seed.

    Args:
        db: Active SQLAlchemy session.
        tmdb_client: TMDB client instance.
        pages: Number of pages to fetch (default 5 = up to 100 movies).

    Returns:
        (movies_upserted, error_count) tuple.
    """
    upserted = 0
    errors = 0

    for page in range(1, pages + 1):
        try:
            data = tmdb_client.get_popular_movies(page=page)
            results: list[dict[str, Any]] = data.get("results", [])
            for item in results:
                try:
                    # Map genre_ids (list of ints) to Genre rows
                    genre_ids: list[int] = item.get("genre_ids", [])
                    movie = upsert_movie_from_tmdb(db, item)
                    # Attach genres by tmdb_genre_id
                    genres = (
                        db.query(Genre)
                        .filter(Genre.tmdb_genre_id.in_(genre_ids))
                        .all()
                        if genre_ids
                        else []
                    )
                    movie.genres = genres
                    upserted += 1
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to upsert movie tmdb_id=%s", item.get("id"))
                    errors += 1
        except Exception:  # noqa: BLE001
            logger.exception("Failed to fetch TMDB popular page %d", page)
            errors += 1

    db.commit()
    logger.info("sync_movies_from_popular: %d upserted, %d errors", upserted, errors)
    return upserted, errors


# ---------------------------------------------------------------------------
# Query functions (used by the movies router)
# ---------------------------------------------------------------------------


def get_movie_by_id(db: Session, movie_id: int) -> Movie | None:
    """Return a Movie by our internal DB primary key, or None if not found."""
    return db.query(Movie).filter(Movie.id == movie_id).first()


def get_movie_by_tmdb_id(db: Session, tmdb_id: int) -> Movie | None:
    """Return a Movie by its TMDB id, or None if not found."""
    return db.query(Movie).filter(Movie.tmdb_id == tmdb_id).first()


def list_movies(
    db: Session,
    *,
    genre_id: int | None = None,
    sort_by: str = "popularity",
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[Movie], int]:
    """
    List movies with optional genre filter and sorting.

    Args:
        db: Active SQLAlchemy session.
        genre_id: If set, only return movies linked to this Genre.id.
        sort_by: Column name to sort by. Accepted values:
                 `popularity`, `vote_average`, `release_date`, `title`.
                 Unknown values fall back to `popularity`.
        limit: Max results to return (1–100).
        offset: Pagination offset.

    Returns:
        (movies, total_count) tuple.
    """
    # Build the base query
    query = db.query(Movie)

    if genre_id is not None:
        query = query.filter(Movie.genres.any(Genre.id == genre_id))

    # Map sort_by to a sortable column expression.
    # We avoid .nulls_last() because SQLite (used in tests) does not support
    # NULLS LAST syntax.  coalesce()-based ordering achieves the same effect
    # on both SQLite and PostgreSQL.
    sort_map = {
        "popularity": func.coalesce(Movie.popularity, 0.0).desc(),
        "vote_average": func.coalesce(Movie.vote_average, 0.0).desc(),
        "release_date": func.coalesce(Movie.release_date, "1900-01-01").desc(),
        "title": Movie.title.asc(),
    }
    order_expr = sort_map.get(sort_by, func.coalesce(Movie.popularity, 0.0).desc())

    total: int = query.count()
    movies = query.order_by(order_expr).offset(offset).limit(limit).all()
    return movies, total


def search_movies_in_db(
    db: Session,
    query_str: str,
    *,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[Movie], int]:
    """
    Case-insensitive title search against our local movie cache.

    Uses PostgreSQL ILIKE for a simple title-substring match.  Semantic /
    vector search is deferred to V2 (Milestone 15).

    Args:
        db: Active SQLAlchemy session.
        query_str: Search term.
        limit: Max results.
        offset: Pagination offset.

    Returns:
        (movies, total_count) tuple.
    """
    pattern = f"%{query_str}%"
    query = db.query(Movie).filter(
        or_(
            Movie.title.ilike(pattern),
            Movie.overview.ilike(pattern),
        )
    )
    total: int = query.count()
    movies = (
        query.order_by(func.coalesce(Movie.popularity, 0.0).desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return movies, total
