"""
Collaborative Filtering recommender — Milestone 12.

Uses scikit-surprise SVD (Singular Value Decomposition) to learn latent
user-item preference factors from a ratings matrix combining:
  - MovieLens ml-latest-small ratings (external training data)
  - In-app user ratings (real user feedback from our DB)

Algorithm choice rationale (per ML_NLP_PLAN.md):
  - SVD: well-understood matrix factorisation, viva-explainable, runs on CPU.
  - scikit-surprise: clean API, full cross-validation support, no GPU needed.
  - Fallback: if the model is not trained (cold start / no artifact), returns [].

Training data format expected by fit():
  DataFrame with columns: ml_user_id (or user_id), movie_id, rating
  - ml_user_id: integer (MovieLens userId or our app user_id — kept separate)
  - movie_id: our internal DB movie id
  - rating: 1–10 scale (MovieLens ratings are pre-normalised by the loader)

Prediction:
  - predict_top_n(user_id, all_movie_ids, n) returns a list of
    (movie_id, predicted_score) tuples sorted descending.
  - user_id here refers to OUR app user_id (integer), NOT a MovieLens userId.
  - The model is trained on MovieLens userId-space; in-app ratings can be
    overlaid by calling fit() with a combined DataFrame (see seed_runner.py).

Cold-start note:
  A new in-app user has no MovieLens userId equivalent. The hybrid.py layer
  handles this by falling back to the content-based recommender when the user
  has fewer than settings.collab_min_ratings in-app ratings.

Usage::

    from app.ml.collaborative import CollaborativeFilteringModel

    model = CollaborativeFilteringModel(settings)
    model.fit(training_df)               # train on MovieLens data
    recs = model.predict_top_n(          # predict for in-app user
        user_id=42,
        candidate_movie_ids=[1, 2, 3, ...],
        n=10,
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import cross_validate

    _SURPRISE_AVAILABLE = True
except ImportError:
    _SURPRISE_AVAILABLE = False
    logger.warning(
        "scikit-surprise is not installed. CollaborativeFilteringModel will not function. "
        "Install it with: pip install scikit-surprise"
    )


# Name under which the trained SVD model is persisted by ModelStore.
MODEL_ARTIFACT_NAME = "collaborative_svd"

# Rating scale used for training (matches our normalised MovieLens ratings).
_RATING_SCALE = (1, 10)


class CollaborativeFilteringModel:
    """
    SVD-based collaborative filtering recommender.

    Args:
        settings: Application settings (provides collab_min_ratings and
                  other tuneable parameters).
    """

    def __init__(self, settings=None) -> None:
        if not _SURPRISE_AVAILABLE:
            raise ImportError(
                "scikit-surprise is required for CollaborativeFilteringModel. "
                "Install it with: pip install scikit-surprise"
            )
        self._settings = settings
        self._algo: SVD | None = None
        self._trained = False
        # Track which movie IDs the model was trained on
        self._trained_movie_ids: set[int] = set()
        self._trained_user_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        ratings_df: pd.DataFrame,
        *,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        verbose: bool = False,
    ) -> "CollaborativeFilteringModel":
        """
        Train the SVD model on a ratings DataFrame.

        Args:
            ratings_df: DataFrame with columns [user_col, 'movie_id', 'rating'].
                        The user column can be 'ml_user_id' or 'user_id'.
                        Ratings should be on a 1–10 scale.
            n_factors: Number of latent factors (default 100 — good balance
                       of accuracy vs training time for ml-latest-small).
            n_epochs: SGD training epochs.
            lr_all: Learning rate for all parameters.
            reg_all: Regularisation for all parameters.
            verbose: If True, print surprise's per-epoch output.

        Returns:
            self (for chaining).
        """
        if ratings_df.empty:
            raise ValueError("ratings_df is empty — cannot train.")

        # Detect user column name
        if "ml_user_id" in ratings_df.columns:
            user_col = "ml_user_id"
        elif "user_id" in ratings_df.columns:
            user_col = "user_id"
        else:
            raise ValueError(
                "ratings_df must have a 'ml_user_id' or 'user_id' column."
            )

        self._trained_movie_ids = set(ratings_df["movie_id"].unique())
        self._trained_user_ids = set(ratings_df[user_col].astype(str).unique())
        logger.info(
            "CollaborativeFilteringModel.fit(): %d rows, %d users, %d movies",
            len(ratings_df),
            len(self._trained_user_ids),
            len(self._trained_movie_ids),
        )

        reader = Reader(rating_scale=_RATING_SCALE)
        data = Dataset.load_from_df(
            ratings_df[[user_col, "movie_id", "rating"]],
            reader,
        )
        trainset = data.build_full_trainset()

        self._algo = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            verbose=verbose,
        )
        self._algo.fit(trainset)
        self._trained = True
        logger.info("CollaborativeFilteringModel: training complete.")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_top_n(
        self,
        user_id: int,
        candidate_movie_ids: list[int],
        n: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Predict top-N movie recommendations for an in-app user.

        Because in-app user IDs are not in the MovieLens training set,
        surprise will use the global mean as the fallback prediction for
        unknown users — this is intentional. The hybrid layer will
        blend this with content-based scores for cold-start users.

        Args:
            user_id: Our internal app user_id (integer).
            candidate_movie_ids: Movie IDs to score (already-rated movies
                                 should be excluded by the caller).
            n: Maximum number of results.

        Returns:
            List of (movie_id, predicted_score) sorted descending by score.
            Empty list if model is not trained.
        """
        if not self._trained or self._algo is None:
            logger.warning(
                "CollaborativeFilteringModel: predict called but model is not trained."
            )
            return []

        user_id_str = f"app_{user_id}"

        # If the user was not in the training set (e.g. they joined or rated after the last batch train),
        # return empty to honestly trigger the content-based fallback instead of returning global means.
        if user_id_str not in self._trained_user_ids:
            return []

        predictions = []
        for movie_id in candidate_movie_ids:
            try:
                pred = self._algo.predict(user_id_str, movie_id)
                predictions.append((movie_id, float(pred.est)))
            except Exception as exc:
                logger.debug("Prediction error for movie_id=%d: %s", movie_id, exc)

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

    @property
    def is_trained(self) -> bool:
        """Return True if the model has been fit."""
        return self._trained

    @property
    def trained_movie_ids(self) -> set[int]:
        """Return the set of movie_ids the model was trained on."""
        return self._trained_movie_ids
