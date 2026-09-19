"""
NLP Batch Runner — Milestone 12.

Processes Review rows that have not yet been analysed by the NLP pipeline
(i.e. where nlp_processed_at IS NULL) and writes the derived analysis
fields back to the database.

NLP fields written per review (already defined on the Review model):
  sentiment_label    — overall label: 'positive' | 'negative' | 'neutral'
  sentiment_score    — model confidence in [0.0, 1.0]
  aspect_sentiment   — dict of per-aspect labels/scores (JSON column)
  nlp_processed_at   — UTC timestamp set when processing is complete

Architecture notes (mirrors SeedRunner conventions):
  - No FastAPI imports; pure service / business logic.
  - SQLAlchemy Session is injected by the caller (route handler or CLI).
    This class NEVER calls SessionLocal() internally.
  - Settings is injected for consistency and future configurability
    (e.g. if per-environment batch-size overrides are needed later).
  - Errors in individual reviews are caught, logged, and accumulated
    in NLPBatchResult.errors — a single bad review never aborts the run.
  - db.commit() is called once per batch, not once per review, so the
    database round-trip cost scales with batch_size, not row count.

Usage::

    from app.nlp.batch import NLPBatchRunner
    from app.db.database import SessionLocal

    db = SessionLocal()
    try:
        runner = NLPBatchRunner(db, settings)
        result = runner.run(batch_size=100)
        print(result)
    finally:
        db.close()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from app.core.config import Settings
from app.models.review import Review
from app.nlp.aspect_sentiment import analyze_aspects
from app.nlp.sentiment import analyze_sentiment

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class NLPBatchResult:
    """
    Summary of a single NLP batch-processing run.

    Fields
    ------
    processed:
        Number of reviews successfully analysed and committed.
    skipped:
        Number of reviews skipped because they had no processable body
        (e.g. empty string after stripping whitespace).
    failed:
        Number of reviews that raised an exception during analysis.
        These rows are left with nlp_processed_at = NULL so the next
        run will retry them automatically.
    errors:
        Human-readable descriptions of each failure, suitable for
        inclusion in an API response or log summary.
    """

    processed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_seen(self) -> int:
        """Total review rows fetched from the database in this run."""
        return self.processed + self.skipped + self.failed


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


class NLPBatchRunner:
    """
    Batch-process unanalysed Review rows with the NLP pipeline.

    Args:
        db:       Active SQLAlchemy Session provided by the caller.
                  The caller is responsible for opening and closing it.
        settings: Application settings (injected for consistency with the
                  rest of the service layer; not currently read internally).
    """

    def __init__(self, db: "Session", settings: Settings) -> None:
        self._db = db
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, *, batch_size: int = 100) -> NLPBatchResult:
        """
        Process all pending Review rows in batches.

        A review is considered *pending* if its ``nlp_processed_at``
        column is NULL (i.e. the NLP pipeline has not yet processed it).

        Args:
            batch_size:
                Maximum number of rows fetched from the database per
                iteration.  Tune this based on available memory and
                database connection latency.  Defaults to 100.

        Returns:
            NLPBatchResult with counts and any per-review error messages.
        """
        result = NLPBatchResult()

        # Count pending reviews up-front for informative logging.
        pending_count: int = (
            self._db.query(Review)
            .filter(Review.nlp_processed_at.is_(None))
            .count()
        )

        if pending_count == 0:
            logger.info("NLP batch runner: no pending reviews — nothing to do.")
            return result

        logger.info(
            "NLP batch runner: %d pending review(s) to process (batch_size=%d).",
            pending_count,
            batch_size,
        )

        offset = 0

        while True:
            batch: list[Review] = (
                self._db.query(Review)
                .filter(Review.nlp_processed_at.is_(None))
                .order_by(Review.id)
                .limit(batch_size)
                .offset(offset)
                .all()
            )

            if not batch:
                # No more rows — all pending reviews have been processed.
                break

            batch_processed = 0
            batch_failed = 0

            for review in batch:
                # Guard: skip reviews with no usable text body.
                if not review.body or not review.body.strip():
                    logger.debug(
                        "Skipping review id=%d: empty body.", review.id
                    )
                    result.skipped += 1
                    # Still mark as processed so we do not retry blanks forever.
                    review.nlp_processed_at = datetime.now(timezone.utc)
                    continue

                try:
                    self._process_review(review)
                    batch_processed += 1
                except Exception as exc:  # noqa: BLE001
                    error_msg = (
                        f"review id={review.id}: {type(exc).__name__}: {exc}"
                    )
                    logger.error("NLP batch runner failed on %s", error_msg)
                    result.errors.append(error_msg)
                    batch_failed += 1
                    # Do NOT set nlp_processed_at — leave it NULL so the next
                    # run retries this review.
                    continue

            # Commit the entire batch in a single transaction.
            try:
                self._db.commit()
                logger.info(
                    "NLP batch committed: processed=%d, failed=%d (offset=%d).",
                    batch_processed,
                    batch_failed,
                    offset,
                )
            except Exception as exc:  # noqa: BLE001
                self._db.rollback()
                error_msg = f"batch commit failed at offset={offset}: {exc}"
                logger.error("NLP batch runner: %s", error_msg)
                result.errors.append(error_msg)
                # Count every review in this batch as failed since the
                # commit was rolled back.
                result.failed += len(batch)
                offset += batch_size
                continue

            result.processed += batch_processed
            result.failed += batch_failed
            offset += batch_size

        logger.info(
            "NLP batch runner complete: processed=%d, skipped=%d, failed=%d.",
            result.processed,
            result.skipped,
            result.failed,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_review(self, review: Review) -> None:
        """
        Run both NLP analysers on a single review and write results back
        to the ORM object (does NOT commit — the caller batches commits).

        Args:
            review: The Review ORM instance to update in-place.

        Raises:
            Any exception raised by analyze_sentiment or analyze_aspects
            is propagated to the caller so it can be recorded without
            crashing the entire batch loop.
        """
        sentiment: dict = analyze_sentiment(review.body)
        aspects: dict = analyze_aspects(review.body)

        review.sentiment_label = sentiment["label"]
        review.sentiment_score = sentiment["score"]
        review.aspect_sentiment = aspects
        review.nlp_processed_at = datetime.now(timezone.utc)

        logger.debug(
            "Processed review id=%d: label=%r score=%.4f aspects=%s",
            review.id,
            review.sentiment_label,
            review.sentiment_score,
            list(aspects.keys()),
        )
