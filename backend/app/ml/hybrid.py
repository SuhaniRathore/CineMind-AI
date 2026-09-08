"""
Hybrid recommender — Milestone 12.

Combines content-based and collaborative filtering scores into a single
ranked list. Implements cold-start logic: users with fewer than
settings.collab_min_ratings in-app ratings receive content-based
recommendations only.

Blend formula (weighted linear):
    hybrid_score = w_content * content_score + w_collab * collab_score

where:
    w_content = settings.hybrid_content_weight     (default 0.4)
    w_collab  = 1.0 - settings.hybrid_content_weight  (default 0.6)

Both input score vectors are min-max normalised to [0, 1] before blending
so that different score ranges (TF-IDF cosine vs SVD estimate) don't bias
the blend. If only one source has scores (cold-start or no model), that
source's scores are used directly (weight = 1.0).

Algorithm tag written to recommendation_results.algorithm:
  - "content_based"  → user has < collab_min_ratings or no trained model
  - "collaborative"  → model is trained but content scores are all 0
  - "hybrid"         → both content and collaborative scores contributed

Usage::

    from app.ml.hybrid import HybridRecommender
    from app.ml.model_store import ModelStore
    from app.core.config import get_settings

    settings = get_settings()
    store = ModelStore(settings)
    collab_model = store.load("collaborative_svd")  # None if not trained

    rec = HybridRecommender(settings, collab_model)
    results = rec.recommend(db, user_id=42, limit=10)
    # results: list of (movie_id, score, algorithm_tag, reason_text)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from app.ml.collaborative import CollaborativeFilteringModel
from app.ml.recommender import generate_recommendations
from app.models.movie import Movie
from app.models.rating import Rating
from app.models.recommendation import (
    ALGORITHM_COLLABORATIVE,
    ALGORITHM_CONTENT_BASED,
    ALGORITHM_HYBRID,
)

logger = logging.getLogger(__name__)


def _minmax_normalise(values: list[float]) -> list[float]:
    """Normalise a list of floats to [0, 1]. Returns zeros if range is 0."""
    if not values:
        return []
    arr = np.array(values, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return [0.0] * len(values)
    return ((arr - mn) / (mx - mn)).tolist()


class HybridRecommender:
    """
    Weighted hybrid of content-based + collaborative filtering.

    Args:
        settings: Application settings (blend weights, cold-start threshold).
        collab_model: Trained CollaborativeFilteringModel, or None if not
                      yet trained (triggers pure content-based fallback).
    """

    def __init__(
        self,
        settings,
        collab_model: CollaborativeFilteringModel | None = None,
    ) -> None:
        self._settings = settings
        self._collab = collab_model
        self._w_content: float = getattr(settings, "hybrid_content_weight", 0.4)
        self._w_collab: float = 1.0 - self._w_content
        self._min_ratings: int = getattr(settings, "collab_min_ratings", 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        db: "Session",
        user_id: int,
        limit: int = 10,
    ) -> list[tuple[int, float, str, str | None]]:
        """
        Generate hybrid recommendations for an in-app user.

        Args:
            db: Active SQLAlchemy session.
            user_id: Our internal app user_id.
            limit: Maximum number of results.

        Returns:
            List of (movie_id, score, algorithm_tag, reason_text) tuples,
            sorted descending by score.
        """
        # Count in-app ratings to decide cold-start handling
        in_app_rating_count = (
            db.query(Rating)
            .filter(Rating.user_id == user_id)
            .count()
        )

        use_collab = (
            self._collab is not None
            and self._collab.is_trained
            and in_app_rating_count >= self._min_ratings
        )

        # 1. Content-based scores (existing recommender)
        content_recs: list[tuple[int, float]] = generate_recommendations(
            db, user_id, limit=limit * 3  # fetch extra, collab will rerank
        )

        # 2. Collaborative scores (if model trained and user has enough ratings)
        collab_recs: list[tuple[int, float]] = []
        if use_collab:
            # Gather all candidate movie IDs (union of what content gave us + all movies)
            rated_ids = {
                r.movie_id
                for r in db.query(Rating).filter(Rating.user_id == user_id).all()
            }
            content_ids = {mid for mid, _ in content_recs}
            all_movie_ids = [
                m.id for m in db.query(Movie.id).all()
                if m.id not in rated_ids
            ]
            collab_recs = self._collab.predict_top_n(
                user_id, all_movie_ids, n=limit * 3
            )

        # 3. Determine algorithm tag and blend
        return self._blend(content_recs, collab_recs, limit, use_collab, in_app_rating_count)

    # ------------------------------------------------------------------
    # Internal blend logic
    # ------------------------------------------------------------------

    def _blend(
        self,
        content_recs: list[tuple[int, float]],
        collab_recs: list[tuple[int, float]],
        limit: int,
        use_collab: bool,
        in_app_rating_count: int,
    ) -> list[tuple[int, float, str, str | None]]:
        """Blend content and collaborative scores and attach metadata."""

        # Pure content-based (cold-start or no trained model)
        if not use_collab:
            reason = self._content_reason(in_app_rating_count)
            return [
                (mid, score, ALGORITHM_CONTENT_BASED, reason)
                for mid, score in content_recs[:limit]
            ]

        # Build score lookup dicts
        content_dict = {mid: score for mid, score in content_recs}
        collab_dict = {mid: score for mid, score in collab_recs}

        # Union of all candidate movies
        all_ids = set(content_dict.keys()) | set(collab_dict.keys())

        if not all_ids:
            return []

        # Normalise both score sets
        id_list = list(all_ids)
        raw_content = [content_dict.get(mid, 0.0) for mid in id_list]
        raw_collab = [collab_dict.get(mid, 0.0) for mid in id_list]

        norm_content = _minmax_normalise(raw_content)
        norm_collab = _minmax_normalise(raw_collab)

        # Weighted blend
        blended = []
        for i, mid in enumerate(id_list):
            c = norm_content[i]
            cf = norm_collab[i]
            has_content = c > 0
            has_collab = cf > 0

            if has_content and has_collab:
                score = self._w_content * c + self._w_collab * cf
                tag = ALGORITHM_HYBRID
            elif has_collab:
                score = cf
                tag = ALGORITHM_COLLABORATIVE
            else:
                score = c
                tag = ALGORITHM_CONTENT_BASED

            blended.append((mid, score, tag))

        blended.sort(key=lambda x: x[1], reverse=True)
        blended = blended[:limit]

        # Attach reason text (placeholder for Milestone 16 explainability)
        result = []
        for mid, score, tag in blended:
            reason = self._tag_reason(tag, in_app_rating_count)
            result.append((mid, score, tag, reason))

        return result

    def _content_reason(self, rating_count: int) -> str | None:
        return None

    def _tag_reason(self, tag: str, rating_count: int) -> str | None:
        # Returning None to avoid misleading explanations until Milestone 16
        return None
