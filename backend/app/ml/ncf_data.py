"""
NCF data preprocessing and PyTorch Dataset — Milestone 14.

Responsibilities:
  - Accept the training_ratings_df produced by MovieLensLoader.load()
    (columns: ml_user_id [str], movie_id [int], rating [float 1–10]).
  - Combine with in-app ratings (same prefix convention as seed_runner).
  - Build dense 0-based index maps:
      user_to_idx : dict[str, int]   — ml_user_id string → embedding row index
      movie_to_idx: dict[int, int]   — internal movie_id → embedding col index
      idx_to_movie: dict[int, int]   — reverse of movie_to_idx (for inference)
  - Perform a stratified 90/10 train/val split by user so every user
    appears in both sets (prevents data leakage from a random split).
  - Expose PyTorch Datasets for both splits.

The preprocessor is serialised (via joblib) alongside the model weights so
the index maps survive across process restarts.

Usage::

    from app.ml.ncf_data import NCFDataPreprocessor

    pre = NCFDataPreprocessor(use_implicit=False)
    train_ds, val_ds = pre.fit_transform(training_df)

    # After loading from disk:
    pre2 = NCFDataPreprocessor.load("path/to/preprocessor.pkl")
    user_idx = pre2.user_to_idx["ml_123"]
    movie_idx = pre2.movie_to_idx[42]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Ratings >= this threshold are considered positive interactions (implicit mode)
_IMPLICIT_POS_THRESHOLD = 7.0

# Fraction of each user's ratings held out for validation
_VAL_FRACTION = 0.10


class NCFDataset(Dataset):
    """
    PyTorch Dataset wrapping arrays of (user_idx, item_idx, label).

    In explicit mode, label is the normalised rating (float, 1–10 → 0–1).
    In implicit mode, label is a binary float (1.0 = positive, 0.0 = negative).
    """

    def __init__(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        assert len(user_indices) == len(item_indices) == len(labels)
        self._users = torch.from_numpy(user_indices.astype(np.int64))
        self._items = torch.from_numpy(item_indices.astype(np.int64))
        self._labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self._users)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._users[idx], self._items[idx], self._labels[idx]


class NCFDataPreprocessor:
    """
    Fit index maps and build train/val Datasets from a ratings DataFrame.

    Args:
        use_implicit: If True, binarise ratings to positive (≥ threshold) /
                      negative (< threshold) and apply negative sampling.
                      If False (default), use raw normalised ratings as labels.
    """

    def __init__(self, use_implicit: bool = False) -> None:
        self.use_implicit = use_implicit

        # Populated by fit_transform()
        self.user_to_idx: dict[str, int] = {}
        self.movie_to_idx: dict[int, int] = {}
        self.idx_to_movie: dict[int, int] = {}
        self.n_users: int = 0
        self.n_items: int = 0
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        ratings_df: pd.DataFrame,
    ) -> tuple[NCFDataset, NCFDataset]:
        """
        Build index maps and return (train_dataset, val_dataset).

        Args:
            ratings_df: DataFrame with columns [ml_user_id (str), movie_id (int),
                        rating (float 1–10)].

        Returns:
            (train_dataset, val_dataset) as NCFDataset instances.
        """
        if ratings_df.empty:
            raise ValueError("ratings_df is empty — cannot fit NCFDataPreprocessor.")

        logger.info(
            "NCFDataPreprocessor.fit_transform(): %d rows, %d users, %d movies",
            len(ratings_df),
            ratings_df["ml_user_id"].nunique(),
            ratings_df["movie_id"].nunique(),
        )

        df = ratings_df[["ml_user_id", "movie_id", "rating"]].copy()
        df = df.dropna(subset=["ml_user_id", "movie_id", "rating"])

        # Build index maps
        unique_users = sorted(df["ml_user_id"].unique())
        unique_movies = sorted(df["movie_id"].unique())

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.movie_to_idx = {m: i for i, m in enumerate(unique_movies)}
        self.idx_to_movie = {i: m for m, i in self.movie_to_idx.items()}
        self.n_users = len(unique_users)
        self.n_items = len(unique_movies)

        logger.info(
            "Index maps built: %d users, %d items",
            self.n_users,
            self.n_items,
        )

        # Map to indices
        df["user_idx"] = df["ml_user_id"].map(self.user_to_idx)
        df["item_idx"] = df["movie_id"].map(self.movie_to_idx)

        # Compute labels
        df["label"] = self._encode_rating_labels(df["rating"])

        # Stratified 90/10 split by user
        train_df, val_df = self._stratified_split(df)

        logger.info(
            "Split: %d train rows, %d val rows",
            len(train_df),
            len(val_df),
        )

        self._is_fitted = True

        train_ds = self._to_dataset(train_df)
        val_ds = self._to_dataset(val_df)
        return train_ds, val_ds

    def transform(self, ratings_df: pd.DataFrame) -> NCFDataset:
        """
        Transform a new DataFrame using already-fitted index maps.
        Rows with unknown users or movies are dropped (cold-start items).
        """
        if not self._is_fitted:
            raise RuntimeError("NCFDataPreprocessor is not fitted. Call fit_transform() first.")

        df = ratings_df[["ml_user_id", "movie_id", "rating"]].copy()
        df["user_idx"] = df["ml_user_id"].map(self.user_to_idx)
        df["item_idx"] = df["movie_id"].map(self.movie_to_idx)
        df = df.dropna(subset=["user_idx", "item_idx"])

        df["label"] = self._encode_rating_labels(df["rating"])

        return self._to_dataset(df)

    def _encode_rating_labels(self, ratings: pd.Series) -> pd.Series:
        if self.use_implicit:
            return (ratings >= _IMPLICIT_POS_THRESHOLD).astype(float)
        return (ratings - 1.0) / 9.0

    def get_all_item_indices(self) -> list[int]:
        """Return all 0-based item indices (useful for batch inference)."""
        return list(range(self.n_items))

    def movie_id_to_idx(self, movie_id: int) -> int | None:
        """Return item index for a given internal movie_id, or None if unknown."""
        return self.movie_to_idx.get(movie_id)

    def idx_to_movie_id(self, idx: int) -> int | None:
        """Return internal movie_id for a given item index, or None if unknown."""
        return self.idx_to_movie.get(idx)

    def save(self, path: str | Path) -> None:
        """Serialise the fitted preprocessor to disk via joblib."""
        joblib.dump(self, path)
        logger.info("NCFDataPreprocessor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "NCFDataPreprocessor":
        """Load a previously saved NCFDataPreprocessor from disk."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected NCFDataPreprocessor, got {type(obj)}")
        logger.info("NCFDataPreprocessor loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stratified_split(
        df: pd.DataFrame,
        val_fraction: float = _VAL_FRACTION,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Stratified split by user.

        Holds out a random `val_fraction` of ratings for validation per user.
        Users with only 1 rating are kept entirely in the training set.
        """
        import numpy as np
        # Count ratings per user
        user_counts = df.groupby("user_idx").size()

        # Give each row a random float to shuffle
        np.random.seed(42)
        df["_rand"] = np.random.rand(len(df))

        # Rank the random floats within each user group
        df["_rank"] = df.groupby("user_idx")["_rand"].rank(method="first", pct=True)

        # Keep users with <=1 rating completely in train.
        # Otherwise, hold out the top `val_fraction` ranks.
        is_single_rating = df["user_idx"].map(user_counts) <= 1
        is_val = (~is_single_rating) & (df["_rank"] > (1.0 - val_fraction))

        val_df = df[is_val].drop(columns=["_rand", "_rank"]).copy()
        train_df = df[~is_val].drop(columns=["_rand", "_rank"]).copy()

        return train_df, val_df

    @staticmethod
    def _to_dataset(df: pd.DataFrame) -> NCFDataset:
        """Convert a preprocessed DataFrame slice to an NCFDataset."""
        return NCFDataset(
            user_indices=df["user_idx"].to_numpy(),
            item_indices=df["item_idx"].to_numpy(),
            labels=df["label"].to_numpy(),
        )
