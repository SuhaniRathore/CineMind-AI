"""
Admin API router - Milestones 12 & 14.

Protected endpoints for triggering the data pipeline and ML model training.
These are NOT for end-users - they are operational/maintenance endpoints.

Authentication: Simple bearer API secret (not a JWT) configured via
ADMIN_API_SECRET in the .env file. This is intentionally simpler than
JWT because these endpoints are called by ops scripts, not by the UI.

Endpoints:
  POST /admin/pipeline/run   - run the full data pipeline (download + load + enrich + train)
  POST /admin/model/train    - train collaborative model from existing data (no re-download)
  GET  /admin/pipeline/status - basic status info (artifact existence, last run)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.database import get_db
from app.ml.model_store import ModelStore
from app.services.tmdb_client import TMDBClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Admin auth dependency
# ---------------------------------------------------------------------------


def _require_admin(
    x_admin_secret: str = Header(
        alias="X-Admin-Secret",
        description="Admin API secret (set via ADMIN_API_SECRET env var)",
    ),
    settings: Settings = Depends(get_settings),
) -> None:
    """
    FastAPI dependency that validates the X-Admin-Secret header.

    Raises HTTP 403 if the secret is missing or incorrect.
    Uses a constant-time comparison via hmac.compare_digest to prevent
    timing-based secret enumeration.
    """
    import hmac
    expected = settings.admin_api_secret
    if not hmac.compare_digest(x_admin_secret.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin secret.",
        )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class PipelineRunRequest(BaseModel):
    download: bool = True    # Download MovieLens dataset if not present
    enrich: bool = True      # Enrich movies with TMDB details
    train: bool = True       # Train SVD collaborative model after loading
    train_ncf: bool = False  # Also train the NCF deep-learning model (M14, slow ~30 min)


class PipelineRunResponse(BaseModel):
    movies_linked: int
    movies_missing: int
    genres_synced: int
    model_trained: bool
    model_path: str
    ncf_model_trained: bool = False   # M14
    ncf_model_path: str = ""          # M14
    errors: list[str]
    message: str


class PipelineStatusResponse(BaseModel):
    collaborative_model_ready: bool
    collaborative_model_path: str
    ncf_model_ready: bool = False     # M14
    ncf_model_path: str = ""          # M14
    movielens_data_present: bool
    movielens_variant: str


class NLPBatchRunResponse(BaseModel):
    processed: int
    skipped: int
    failed: int
    total_seen: int
    errors: list[str]
    message: str


# ---------------------------------------------------------------------------
# TMDB client dependency (reuse existing pattern from movies router)
# ---------------------------------------------------------------------------


def _get_tmdb_client(settings: Settings = Depends(get_settings)) -> TMDBClient:
    return TMDBClient(settings=settings)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/pipeline/run",
    response_model=PipelineRunResponse,
    summary="Run the full data pipeline",
)
def run_pipeline(
    body: PipelineRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    tmdb_client: TMDBClient = Depends(_get_tmdb_client),
    settings: Settings = Depends(get_settings),
    _auth: None = Depends(_require_admin),
) -> PipelineRunResponse:
    """
    Trigger the full data expansion pipeline:
      1. Download MovieLens dataset (if download=True and not present).
      2. Load MovieLens CSVs -> link imdb_ids, build training data.
      3. Enrich movies with TMDB details (if enrich=True).
      4. Train the collaborative filtering model (if train=True).

    This is a synchronous endpoint - it runs the pipeline inline and
    returns when complete. For very large datasets, run this from the
    CLI script (backend/scripts/run_pipeline.py) instead.

    **Requires X-Admin-Secret header.**
    """
    from app.services.data_pipeline.seed_runner import SeedRunner

    logger.info("Admin: pipeline run triggered (download=%s, enrich=%s, train=%s, train_ncf=%s)",
                body.download, body.enrich, body.train, body.train_ncf)

    runner = SeedRunner(db, tmdb_client, settings)
    result = runner.run(
        download=body.download,
        enrich=body.enrich,
        train=body.train,
        train_ncf=body.train_ncf,
    )

    error_count = len(result.errors)
    message = (
        f"Pipeline complete. "
        f"linked={result.movielens.movies_linked}, "
        f"enriched={result.enrichment.enriched}, "
        f"svd_trained={result.model_trained}, "
        f"ncf_trained={result.ncf_model_trained}. "
        f"Errors: {error_count}"
    )

    return PipelineRunResponse(
        movies_linked=result.movielens.movies_linked,
        movies_missing=result.movielens.movies_missing,
        genres_synced=result.movielens.genres_synced,
        model_trained=result.model_trained,
        model_path=result.model_path,
        ncf_model_trained=result.ncf_model_trained,
        ncf_model_path=result.ncf_model_path,
        errors=result.errors,
        message=message,
    )


@router.post(
    "/model/train",
    response_model=PipelineRunResponse,
    summary="Train collaborative model from existing data",
)
def train_model(
    db: Session = Depends(get_db),
    tmdb_client: TMDBClient = Depends(_get_tmdb_client),
    settings: Settings = Depends(get_settings),
    _auth: None = Depends(_require_admin),
) -> PipelineRunResponse:
    """
    Train (or retrain) the collaborative filtering model using existing
    MovieLens data on disk and explicitly including existing CineMind in-app ratings.

    CineMind user IDs and MovieLens user IDs are disambiguated cleanly (e.g. "app_42" vs "ml_1")
    so they do not blend incompatible ID spaces, while still allowing the model to learn
    movie patterns from the external dataset.

    Use this to retrain the model after new in-app ratings accumulate,
    without re-running the full pipeline.

    **Requires X-Admin-Secret header.**
    """
    from app.services.data_pipeline.seed_runner import SeedRunner

    logger.info("Admin: model retrain triggered (train_ncf=%s)", settings.use_deep_collab)

    runner = SeedRunner(db, tmdb_client, settings)
    result = runner.run(
        download=False,
        enrich=False,
        train=False,
        train_ncf=settings.use_deep_collab,
        stream_ncf=True,
    )

    return PipelineRunResponse(
        movies_linked=result.movielens.movies_linked,
        movies_missing=result.movielens.movies_missing,
        genres_synced=result.movielens.genres_synced,
        model_trained=result.model_trained,
        model_path=result.model_path,
        ncf_model_trained=result.ncf_model_trained,
        ncf_model_path=result.ncf_model_path,
        errors=result.errors,
        message=(
            f"Model retrain complete. "
            f"svd_trained={result.model_trained}, "
            f"ncf_trained={result.ncf_model_trained}. "
            f"Errors: {len(result.errors)}"
        ),
    )


@router.get(
    "/pipeline/status",
    response_model=PipelineStatusResponse,
    summary="Check pipeline and model status",
)
def pipeline_status(
    settings: Settings = Depends(get_settings),
    _auth: None = Depends(_require_admin),
) -> PipelineStatusResponse:
    """
    Return the current status of the data pipeline and trained models.

    **Requires X-Admin-Secret header.**
    """
    from pathlib import Path

    store = ModelStore(settings)
    collab_ready = store.exists("collaborative_svd")
    ncf_ready = store.ncf_exists("collaborative_ncf")

    movielens_dir = Path(settings.data_dir) / "movielens" / settings.movielens_variant
    movielens_present = (movielens_dir / "movies.csv").exists()

    return PipelineStatusResponse(
        collaborative_model_ready=collab_ready,
        collaborative_model_path=store.artifact_path("collaborative_svd"),
        ncf_model_ready=ncf_ready,
        ncf_model_path=str((Path(settings.models_dir) / "collaborative_ncf").resolve()),
        movielens_data_present=movielens_present,
        movielens_variant=settings.movielens_variant,
    )


@router.post(
    "/nlp/run",
    response_model=NLPBatchRunResponse,
    summary="Run the NLP batch pipeline over unprocessed reviews",
)
def run_nlp_batch(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _auth: None = Depends(_require_admin),
) -> NLPBatchRunResponse:
    """
    Run sentiment and aspect-sentiment analysis on all Review rows whose
    ``nlp_processed_at`` column is NULL (i.e. not yet processed).

    Processing is synchronous - the response is returned when the batch
    is complete.  For very large review tables, trigger this from the
    CLI instead.

    Fields returned:
      - **processed** - reviews successfully analysed and committed.
      - **skipped**   - reviews with empty/blank body (marked done, not retried).
      - **failed**    - reviews that raised an exception (left NULL, retried next run).
      - **total_seen** - processed + skipped + failed.
      - **errors**    - human-readable description of each per-review failure.

    **Requires X-Admin-Secret header.**
    """
    from app.nlp.batch import NLPBatchRunner

    logger.info("Admin: NLP batch run triggered.")

    runner = NLPBatchRunner(db, settings)
    result = runner.run(batch_size=100)

    message = (
        f"NLP batch complete. "
        f"processed={result.processed}, "
        f"skipped={result.skipped}, "
        f"failed={result.failed}. "
        f"Errors: {len(result.errors)}"
    )
    logger.info(message)

    return NLPBatchRunResponse(
        processed=result.processed,
        skipped=result.skipped,
        failed=result.failed,
        total_seen=result.total_seen,
        errors=result.errors,
        message=message,
    )


@router.post(
    "/nlp/summarize",
    response_model=NLPBatchRunResponse,
    summary="Run the NLP summarization batch pipeline",
)
def run_nlp_summarization_batch(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _auth: None = Depends(_require_admin),
) -> NLPBatchRunResponse:
    """
    Run extractive summarization for all movies that have un-summarized
    or modified reviews.

    Processing is synchronous - the response is returned when the batch
    is complete.

    **Requires X-Admin-Secret header.**
    """
    from app.nlp.summarization_batch import NLPSummarizationRunner

    logger.info("Admin: NLP summarization batch run triggered.")

    runner = NLPSummarizationRunner(db, settings)
    result = runner.run(batch_size=50)

    message = (
        f"NLP summarization batch complete. "
        f"processed={result.processed}, "
        f"skipped={result.skipped}, "
        f"failed={result.failed}. "
        f"Errors: {len(result.errors)}"
    )
    logger.info(message)

    return NLPBatchRunResponse(
        processed=result.processed,
        skipped=result.skipped,
        failed=result.failed,
        total_seen=result.total_seen,
        errors=result.errors,
        message=message,
    )


