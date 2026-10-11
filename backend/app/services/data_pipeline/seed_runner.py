"""
Seed runner — Milestones 12 & 14.

Supports:
  - MovieLens metadata loading
  - TMDB enrichment
  - SVD collaborative filtering
  - normal NCF training
  - memory-safe streaming NCF training
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from sqlalchemy.orm import Session

from app.core.config import Settings
from app.ml.collaborative import CollaborativeFilteringModel
from app.ml.model_store import ModelStore
from app.services.data_pipeline.movielens_loader import (
    MovieLensLoader,
    MovieLensLoadResult,
)
from app.services.data_pipeline.tmdb_enricher import (
    TMDBEnricher,
    EnrichmentResult,
)
from app.services.tmdb_client import TMDBClient

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Summary of a pipeline run."""

    movielens: MovieLensLoadResult = field(
        default_factory=MovieLensLoadResult
    )

    enrichment: EnrichmentResult = field(
        default_factory=EnrichmentResult
    )

    model_trained: bool = False
    model_path: str = ""

    ncf_model_trained: bool = False
    ncf_model_path: str = ""

    errors: list[str] = field(
        default_factory=list
    )


class SeedRunner:
    """Orchestrates MovieLens loading and ML training."""

    def __init__(
        self,
        db: "Session",
        tmdb_client: TMDBClient,
        settings: Settings,
    ) -> None:
        self._db = db
        self._tmdb_client = tmdb_client
        self._settings = settings

        self._loader = MovieLensLoader(
            settings
        )

        self._enricher = TMDBEnricher(
            tmdb_client,
            settings,
        )

        self._store = ModelStore(
            settings
        )

    def run(
        self,
        *,
        download: bool = True,
        enrich: bool = True,
        train: bool = True,
        train_ncf: bool = False,
        stream_ncf: bool = False,
    ) -> PipelineResult:
        """
        Run the requested pipeline steps.

        For large datasets such as ml-25m:

            train=False
            train_ncf=True
            stream_ncf=True

        This avoids loading the complete ratings.csv into RAM.
        """

        result = PipelineResult()

        # --------------------------------------------------------------
        # Step 1: Download MovieLens
        # --------------------------------------------------------------
        if download:
            try:
                self._loader.download()

            except RuntimeError as exc:
                logger.error(
                    "MovieLens download failed: %s",
                    exc,
                )

                result.errors.append(
                    f"download: {exc}"
                )

                return result

        # --------------------------------------------------------------
        # Step 2: Load MovieLens
        # --------------------------------------------------------------
        try:
            result.movielens = self._loader.load(
                self._db,
                stream_ncf=(
                    stream_ncf
                    and train_ncf
                    and not train
                ),
            )

            self._db.commit()

            logger.info(
                "MovieLens load complete: "
                "%d linked, %d missing",
                result.movielens.movies_linked,
                result.movielens.movies_missing,
            )

        except Exception as exc:
            logger.error(
                "MovieLens load failed: %s",
                exc,
            )

            result.errors.append(
                f"movielens_load: {exc}"
            )

            return result

        # --------------------------------------------------------------
        # Step 3: TMDB enrichment
        # --------------------------------------------------------------
        if enrich:
            try:
                result.enrichment = (
                    self._enricher.enrich_all(
                        self._db
                    )
                )

                logger.info(
                    "Enrichment complete: "
                    "%d enriched, %d errors",
                    result.enrichment.enriched,
                    result.enrichment.errors,
                )

            except Exception as exc:
                logger.error(
                    "TMDB enrichment failed: %s",
                    exc,
                )

                result.errors.append(
                    f"enrichment: {exc}"
                )

        # --------------------------------------------------------------
        # Step 4: Large-scale streaming NCF
        # --------------------------------------------------------------
        if (
            train_ncf
            and stream_ncf
            and not train
            and result.movielens.ncf_streaming
        ):
            try:
                self._run_ncf_streaming(
                    result.movielens,
                    result,
                )

            except Exception as exc:
                logger.error(
                    "Streaming NCF training failed: %s",
                    exc,
                )

                result.errors.append(
                    f"train_ncf: {exc}"
                )

            return result

        # --------------------------------------------------------------
        # Step 5: Traditional DataFrame path
        # --------------------------------------------------------------
        import pandas as pd
        from app.models.rating import Rating

        train_df = (
            result.movielens
            .training_ratings_df
            .copy()
        )

        in_app_ratings = (
            self._db.query(
                Rating
            ).all()
        )

        if in_app_ratings:
            app_df = pd.DataFrame(
                [
                    {
                        "ml_user_id": (
                            f"app_{rating.user_id}"
                        ),
                        "movie_id": rating.movie_id,
                        "rating": rating.rating,
                    }
                    for rating in in_app_ratings
                ]
            )

            train_df = pd.concat(
                [
                    train_df,
                    app_df,
                ],
                ignore_index=True,
            )

        if train_df.empty:
            logger.warning(
                "Training DataFrame is empty."
            )

            if train or train_ncf:
                result.errors.append(
                    "train: no training data available"
                )

            return result

        # --------------------------------------------------------------
        # Step 6: SVD
        # --------------------------------------------------------------
        if train:
            try:
                model = (
                    CollaborativeFilteringModel(
                        self._settings
                    )
                )

                model.fit(
                    train_df
                )

                path = self._store.save(
                    model,
                    "collaborative_svd",
                )

                result.model_trained = True
                result.model_path = path

                logger.info(
                    "Collaborative SVD model trained "
                    "and saved to %s",
                    path,
                )

            except Exception as exc:
                logger.error(
                    "SVD model training failed: %s",
                    exc,
                )

                result.errors.append(
                    f"train_svd: {exc}"
                )

        # --------------------------------------------------------------
        # Step 7: Normal NCF
        # --------------------------------------------------------------
        if train_ncf:
            try:
                self._run_ncf_training(
                    train_df,
                    result,
                )

            except Exception as exc:
                logger.error(
                    "NCF model training failed: %s",
                    exc,
                )

                result.errors.append(
                    f"train_ncf: {exc}"
                )

        return result

    # ------------------------------------------------------------------
    # Normal NCF training
    # ------------------------------------------------------------------

    def _run_ncf_training(
        self,
        train_df: "pd.DataFrame",
        result: PipelineResult,
    ) -> None:
        from app.ml.ncf_data import (
            NCFDataPreprocessor,
        )

        preprocessor = (
            NCFDataPreprocessor(
                use_implicit=False
            )
        )

        train_ds, val_ds = (
            preprocessor.fit_transform(
                train_df
            )
        )

        self._train_ncf_datasets(
            preprocessor,
            train_ds,
            val_ds,
            result,
        )

    # ------------------------------------------------------------------
    # Streaming NCF preparation
    # ------------------------------------------------------------------

    def _run_ncf_streaming(
        self,
        movielens: MovieLensLoadResult,
        result: PipelineResult,
    ) -> None:
        import pandas as pd

        from app.models.rating import Rating
        from app.ml.ncf_streaming import (
            NCFStreamingPreprocessor,
        )

        in_app_ratings = (
            self._db.query(
                Rating
            ).all()
        )

        extra_ratings_df = None

        if in_app_ratings:
            extra_ratings_df = pd.DataFrame(
                [
                    {
                        "ml_user_id": (
                            f"app_{rating.user_id}"
                        ),
                        "movie_id": rating.movie_id,
                        "rating": rating.rating,
                    }
                    for rating in in_app_ratings
                ]
            )

        ratings_csv = Path(
            movielens.ratings_csv_path
        )

        stream_dir = (
            ratings_csv.parent
            / "ncf_stream"
        )

        preprocessor = (
            NCFStreamingPreprocessor(
                use_implicit=False
            )
        )

        logger.info(
            "Preparing streaming NCF dataset..."
        )

        train_ds, val_ds = (
            preprocessor.prepare_from_csv(
                ratings_csv=ratings_csv,
                ml_to_internal=(
                    movielens.ml_to_internal
                ),
                output_dir=stream_dir,
                batch_size=(
                    self._settings
                    .ncf_batch_size
                ),
                chunk_size=250_000,
                val_fraction=0.10,
                seed=42,
                extra_ratings_df=(
                    extra_ratings_df
                ),
            )
        )

        self._train_streaming_ncf(
            preprocessor,
            train_ds,
            val_ds,
            result,
        )

    # ------------------------------------------------------------------
    # Normal NCF training helper
    # ------------------------------------------------------------------

    def _train_ncf_datasets(
        self,
        preprocessor,
        train_ds,
        val_ds,
        result: PipelineResult,
    ) -> None:
        import torch

        from app.ml.ncf_model import (
            NCFModel,
        )

        from app.ml.ncf_trainer import (
            NCFTrainer,
        )

        settings = self._settings

        torch.manual_seed(
            42
        )

        model = NCFModel(
            n_users=(
                preprocessor.n_users
            ),
            n_items=(
                preprocessor.n_items
            ),
            gmf_dim=(
                settings.ncf_gmf_dim
            ),
            mlp_layers=(
                settings.ncf_mlp_layers
            ),
            dropout=(
                settings.ncf_dropout
            ),
            use_implicit=False,
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        trainer = NCFTrainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            preprocessor=preprocessor,
            epochs=settings.ncf_epochs,
            lr=settings.ncf_lr,
            batch_size=settings.ncf_batch_size,
            weight_decay=settings.ncf_weight_decay,
            patience=settings.ncf_early_stop_patience,
            device=device,
        )

        best_rmse = (
            trainer.train()
        )

        path = (
            self._store.save_ncf(
                model,
                preprocessor,
                name="collaborative_ncf",
            )
        )

        result.ncf_model_trained = True
        result.ncf_model_path = path

        logger.info(
            "NCF model trained "
            "(best_val_rmse=%.6f) "
            "and saved to %s",
            best_rmse,
            path,
        )

    # ------------------------------------------------------------------
    # Streaming NCF training helper
    # ------------------------------------------------------------------

    def _train_streaming_ncf(
        self,
        preprocessor,
        train_ds,
        val_ds,
        result: PipelineResult,
    ) -> None:
        import torch

        from app.ml.ncf_model import (
            NCFModel,
        )

        from app.ml.ncf_streaming import (
            NCFStreamingTrainer,
        )

        settings = self._settings

        torch.manual_seed(
            42
        )

        logger.info(
            "Streaming NCF model: "
            "users=%d items=%d",
            preprocessor.n_users,
            preprocessor.n_items,
        )

        model = NCFModel(
            n_users=(
                preprocessor.n_users
            ),
            n_items=(
                preprocessor.n_items
            ),
            gmf_dim=(
                settings.ncf_gmf_dim
            ),
            mlp_layers=(
                settings.ncf_mlp_layers
            ),
            dropout=(
                settings.ncf_dropout
            ),
            use_implicit=False,
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        trainer = NCFStreamingTrainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            preprocessor=preprocessor,
            epochs=settings.ncf_epochs,
            lr=settings.ncf_lr,
            weight_decay=settings.ncf_weight_decay,
            patience=settings.ncf_early_stop_patience,
            device=device,
        )

        best_rmse = (
            trainer.train()
        )

        path = (
            self._store.save_ncf(
                model,
                preprocessor,
                name="collaborative_ncf",
            )
        )

        result.ncf_model_trained = True
        result.ncf_model_path = path

        logger.info(
            "Streaming NCF model trained "
            "(best_val_rmse=%.6f) "
            "and saved to %s",
            best_rmse,
            path,
        )