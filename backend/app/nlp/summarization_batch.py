"""
NLP Summarization Batch Runner — Milestone 13.

Processes movies to generate review summaries.
A movie requires summarization if:
1. It has reviews.
2. Its `review_summary_updated_at` is NULL or is older than the latest review for that movie.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import func

from app.core.config import Settings
from app.models.movie import Movie
from app.models.review import Review
from app.nlp.summarization import generate_review_summary

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class NLPSummarizationBatchResult:
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_seen(self) -> int:
        return self.processed + self.skipped + self.failed


class NLPSummarizationRunner:
    """
    Batch-process movies to generate or update review summaries.

    Args:
        db: Active SQLAlchemy Session provided by the caller.
        settings: Application settings.
    """

    def __init__(self, db: "Session", settings: Settings) -> None:
        self._db = db
        self._settings = settings

    def run(self, *, batch_size: int = 50) -> NLPSummarizationBatchResult:
        result = NLPSummarizationBatchResult()

        # Find movies that need summarization:
        # A movie needs summarization if any of its reviews were updated
        # after its review_summary_updated_at, or if it has reviews but
        # review_summary_updated_at is NULL.

        # A subquery to get max updated_at for reviews per movie
        latest_reviews = (
            self._db.query(
                Review.movie_id,
                func.max(Review.updated_at).label("max_review_updated_at"),
            )
            .group_by(Review.movie_id)
            .subquery()
        )

        pending_movies = (
            self._db.query(Movie)
            .join(latest_reviews, Movie.id == latest_reviews.c.movie_id)
            .filter(
                (Movie.review_summary_updated_at.is_(None))
                | (
                    Movie.review_summary_updated_at
                    < latest_reviews.c.max_review_updated_at
                )
            )
        )

        pending_count: int = pending_movies.count()

        if pending_count == 0:
            logger.info(
                "NLP summarization runner: no movies need summarization."
            )
            return result

        logger.info(
            "NLP summarization runner: %d movie(s) to process "
            "(batch_size=%d).",
            pending_count,
            batch_size,
        )

        # Take a snapshot of the pending movie IDs before processing.
        # This prevents pagination from skipping movies after processed
        # movies no longer match the pending query.
        pending_movie_ids = [
            movie_id
            for (movie_id,) in (
                pending_movies
                .with_entities(Movie.id)
                .order_by(Movie.id)
                .all()
            )
        ]

        for start in range(0, len(pending_movie_ids), batch_size):
            batch_ids = pending_movie_ids[start : start + batch_size]

            batch: list[Movie] = (
                self._db.query(Movie)
                .filter(Movie.id.in_(batch_ids))
                .order_by(Movie.id)
                .all()
            )

            if not batch:
                continue

            batch_processed = 0
            batch_failed = 0

            for movie in batch:
                try:
                    self._process_movie(movie)
                    batch_processed += 1

                except Exception as exc:  # noqa: BLE001
                    error_msg = (
                        f"movie id={movie.id}: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    logger.error(
                        "NLP summarization runner failed on %s",
                        error_msg,
                    )

                    result.errors.append(error_msg)
                    batch_failed += 1

            try:
                self._db.commit()

                logger.info(
                    "NLP summarization batch committed: "
                    "processed=%d, failed=%d, batch_start=%d.",
                    batch_processed,
                    batch_failed,
                    start,
                )

            except Exception as exc:  # noqa: BLE001
                self._db.rollback()

                error_msg = (
                    f"batch commit failed for ids={batch_ids}: {exc}"
                )

                logger.error(
                    "NLP summarization runner: %s",
                    error_msg,
                )

                result.errors.append(error_msg)
                result.failed += len(batch)
                continue

            result.processed += batch_processed
            result.failed += batch_failed

        logger.info(
            "NLP summarization runner complete: processed=%d, failed=%d.",
            result.processed,
            result.failed,
        )

        return result

    def _process_movie(self, movie: Movie) -> None:
        """
        Generate summary for a single movie and update the ORM object.
        """

        # Fetch all reviews for this movie
        reviews = (
            self._db.query(Review)
            .filter(Review.movie_id == movie.id)
            .all()
        )

        review_texts = [r.body for r in reviews]

        summary = generate_review_summary(review_texts)

        movie.review_summary = summary
        movie.review_summary_updated_at = datetime.now(timezone.utc)

        logger.debug(
            "Summarized movie id=%d: %d reviews -> %r",
            movie.id,
            len(reviews),
            summary,
        )