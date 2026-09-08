"""
Seed runner — Milestone 12.

Orchestrates the full data pipeline:
  1. Download MovieLens dataset (if not present).
  2. Load MovieLens CSVs → link imdb_ids, build training DataFrame.
  3. Optionally enrich movies with TMDB details (imdb_id, runtime).
  4. Train the collaborative filtering model.
  5. Save the trained model artifact to disk.

Designed to be called from:
  - The admin API endpoint: POST /admin/pipeline/run
  - The CLI script: backend/scripts/run_pipeline.py

Usage::

    from app.services.data_pipeline.seed_runner import SeedRunner

    runner = SeedRunner(db, tmdb_client, settings)
    result = runner.run(enrich=True, train=True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from app.core.config import Settings
from app.ml.collaborative import CollaborativeFilteringModel
from app.ml.model_store import ModelStore
from app.services.data_pipeline.movielens_loader import MovieLensLoader, MovieLensLoadResult
from app.services.data_pipeline.tmdb_enricher import TMDBEnricher, EnrichmentResult
from app.services.tmdb_client import TMDBClient

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Summary of a full pipeline run."""
    movielens: MovieLensLoadResult = field(default_factory=MovieLensLoadResult)
    enrichment: EnrichmentResult = field(default_factory=EnrichmentResult)
    model_trained: bool = False
    model_path: str = ""
    errors: list[str] = field(default_factory=list)


class SeedRunner:
    """
    Orchestrates the complete data expansion + ML training pipeline.

    Args:
        db: Active SQLAlchemy session.
        tmdb_client: TMDB client (used for enrichment step).
        settings: Application settings.
    """

    def __init__(
        self,
        db: "Session",
        tmdb_client: TMDBClient,
        settings: Settings,
    ) -> None:
        self._db = db
        self._tmdb_client = tmdb_client
        self._settings = settings
        self._loader = MovieLensLoader(settings)
        self._enricher = TMDBEnricher(tmdb_client, settings)
        self._store = ModelStore(settings)

    def run(
        self,
        *,
        download: bool = True,
        enrich: bool = True,
        train: bool = True,
    ) -> PipelineResult:
        """
        Run the full pipeline.

        Args:
            download: If True, download MovieLens dataset if not present.
            enrich: If True, enrich movies with TMDB detail calls.
            train: If True, train and save the collaborative model.

        Returns:
            PipelineResult with summary of all steps.
        """
        result = PipelineResult()

        # Step 1: Download MovieLens dataset
        if download:
            try:
                self._loader.download()
            except RuntimeError as exc:
                logger.error("MovieLens download failed: %s", exc)
                result.errors.append(f"download: {exc}")
                # Cannot proceed without data
                return result

        # Step 2: Load MovieLens → link imdb_ids, build training DataFrame
        try:
            result.movielens = self._loader.load(self._db)
            self._db.commit()
            logger.info(
                "MovieLens load complete: %d linked, %d missing",
                result.movielens.movies_linked,
                result.movielens.movies_missing,
            )
        except Exception as exc:
            logger.error("MovieLens load failed: %s", exc)
            result.errors.append(f"movielens_load: {exc}")
            return result

        # Step 3: TMDB enrichment (optional — slow, calls TMDB per movie)
        if enrich:
            try:
                result.enrichment = self._enricher.enrich_all(self._db)
                logger.info(
                    "Enrichment complete: %d enriched, %d errors",
                    result.enrichment.enriched,
                    result.enrichment.errors,
                )
            except Exception as exc:
                logger.error("TMDB enrichment failed: %s", exc)
                result.errors.append(f"enrichment: {exc}")
                # Non-fatal — continue to training with what we have

        # Step 4: Train collaborative model
        if train:
            import pandas as pd
            from app.models.rating import Rating

            train_df = result.movielens.training_ratings_df.copy()

            # Extract existing CineMind in-app ratings to combine with MovieLens dataset.
            # We prefix user IDs to prevent blending incompatible ID spaces.
            in_app_ratings = self._db.query(Rating).all()
            if in_app_ratings:
                app_df = pd.DataFrame([
                    {
                        "ml_user_id": f"app_{r.user_id}",
                        "movie_id": r.movie_id,
                        "rating": r.rating,
                    }
                    for r in in_app_ratings
                ])
                train_df = pd.concat([train_df, app_df], ignore_index=True)

            if train_df.empty:
                logger.warning("Training DataFrame is empty — skipping model training")
                result.errors.append("train: no training data available")
            else:
                try:
                    model = CollaborativeFilteringModel(self._settings)
                    model.fit(train_df)
                    path = self._store.save(model, "collaborative_svd")
                    result.model_trained = True
                    result.model_path = path
                    logger.info("Collaborative model trained and saved to %s", path)
                except Exception as exc:
                    logger.error("Model training failed: %s", exc)
                    result.errors.append(f"train: {exc}")

        return result
