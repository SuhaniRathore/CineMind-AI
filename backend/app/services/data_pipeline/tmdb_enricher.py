"""
TMDB enricher — Milestone 12.

Responsibilities:
  - Bulk-enrich existing movie rows with data only available from the TMDB
    detail endpoint (e.g. runtime, imdb_id from external_ids, keywords).
  - Respects TMDB's rate limits (40 requests/second) with configurable
    inter-request delays.
  - Designed to run as an offline batch job (not inside a request handler).

Why this is needed:
  - The existing sync (POST /movies/sync) only fetches the popular-movie LIST
    endpoint, which does NOT include runtime or TMDB's external_ids (imdb_id).
  - For collaborative filtering we need imdb_id to cross-reference MovieLens.
  - Runtime is used by the content-based recommender in future iterations.

Usage::

    from app.services.data_pipeline.tmdb_enricher import TMDBEnricher
    from app.services.tmdb_client import TMDBClient
    from app.core.config import get_settings

    enricher = TMDBEnricher(tmdb_client, settings)
    result = enricher.enrich_all(db, batch_size=50, delay_seconds=0.03)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from app.models.movie import Movie
from app.services.tmdb_client import TMDBClient, TMDBError

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentResult:
    """Summary of a TMDB enrichment run."""
    enriched: int = 0
    skipped: int = 0   # already had imdb_id — no call needed
    errors: int = 0


class TMDBEnricher:
    """
    Bulk-enrich movie rows using the TMDB detail endpoint.

    Only fetches movies where imdb_id IS NULL — already-enriched rows
    are skipped to avoid unnecessary API calls.

    Args:
        tmdb_client: Injected TMDB client instance.
        settings: Application settings (for future per-field enable/disable).
        delay_seconds: Sleep between each TMDB request. Default 0.03 s ≈ 33/s,
                       safely below TMDB's 40 req/s limit.
    """

    def __init__(
        self,
        tmdb_client: TMDBClient,
        settings=None,
        delay_seconds: float = 0.03,
    ) -> None:
        self._client = tmdb_client
        self._delay = delay_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich_all(
        self,
        db: "Session",
        batch_size: int = 50,
    ) -> EnrichmentResult:
        """
        Enrich all movies that are missing imdb_id.

        Fetches movie details from TMDB one-by-one with rate-limit delay.
        Commits after each batch_size movies to avoid holding a long transaction.

        Args:
            db: Active SQLAlchemy session.
            batch_size: Commit every N movies to limit transaction size.

        Returns:
            EnrichmentResult summary.
        """
        result = EnrichmentResult()

        # Only fetch movies where imdb_id is not yet populated
        movies_to_enrich = (
            db.query(Movie)
            .filter(Movie.imdb_id.is_(None))
            .order_by(Movie.id)
            .all()
        )

        total = len(movies_to_enrich)
        logger.info("TMDBEnricher: %d movies to enrich", total)

        for i, movie in enumerate(movies_to_enrich, start=1):
            try:
                detail = self._client.get_movie_details(movie.tmdb_id)

                # Extract imdb_id — TMDB returns it as "tt0114709" string format
                imdb_str: str | None = detail.get("imdb_id")
                if imdb_str and imdb_str.startswith("tt"):
                    movie.imdb_id = imdb_str

                # Also update runtime if missing
                if movie.runtime is None and detail.get("runtime"):
                    movie.runtime = detail["runtime"]

                result.enriched += 1

            except TMDBError as exc:
                logger.warning(
                    "TMDBEnricher: failed to enrich movie id=%d tmdb_id=%d: %s",
                    movie.id,
                    movie.tmdb_id,
                    exc,
                )
                result.errors += 1

            # Batch commit
            if i % batch_size == 0:
                db.flush()
                db.commit()
                logger.info("TMDBEnricher: enriched %d/%d …", i, total)

            # Rate-limit delay
            time.sleep(self._delay)

        # Final commit
        db.commit()
        logger.info(
            "TMDBEnricher complete: %d enriched, %d skipped, %d errors",
            result.enriched,
            result.skipped,
            result.errors,
        )
        return result
