"""
NCF Collaborative Adapter — Milestone 14.

Wraps a trained NCFModel + NCFDataPreprocessor behind the same interface
as CollaborativeFilteringModel so that HybridRecommender can use either
model without any changes to its own code.

Interface contract (duck-typed — no ABC required):
  predict_top_n(user_id: int, candidate_movie_ids: list[int], n: int)
      → list[tuple[int, float]]
  is_trained: bool (property)
  trained_movie_ids: set[int] (property)

Cold-start behaviour:
  - If the in-app user_id is not in the preprocessor's user_to_idx map
    (i.e. the user was not present in the training data), predict_top_n
    returns [] so the HybridRecommender falls back to content-based only.
  - If a candidate movie_id is not in movie_to_idx, it is skipped silently
    (new movies unseen at training time get a score of 0 by omission).

Inference strategy:
  - We score ALL candidate_movie_ids in a single batched forward pass
    (no per-item loop), which is typically < 100 ms for 10 k candidates on CPU.
  - Scores are raw model outputs (explicit mode) or sigmoid probabilities
    (implicit mode). The HybridRecommender normalises them before blending,
    so the absolute range does not matter.

Usage::

    from app.ml.ncf_adapter import NCFCollaborativeAdapter
    from app.ml.model_store import ModelStore

    store = ModelStore(settings)
    model, preprocessor = store.load_ncf("collaborative_ncf")
    adapter = NCFCollaborativeAdapter(model, preprocessor)

    recs = adapter.predict_top_n(user_id=42, candidate_movie_ids=[1, 2, 3], n=10)
"""

from __future__ import annotations

import logging

import torch

from app.ml.ncf_data import NCFDataPreprocessor
from app.ml.ncf_model import NCFModel

logger = logging.getLogger(__name__)

# Prefix used in seed_runner / movielens_loader for in-app user IDs
_APP_USER_PREFIX = "app_"


class NCFCollaborativeAdapter:
    """
    Drop-in replacement for CollaborativeFilteringModel in HybridRecommender.

    Args:
        model:        Trained NCFModel.
        preprocessor: Fitted NCFDataPreprocessor (holds the index maps).
        device:       Torch device for inference (default: CPU).
    """

    def __init__(
        self,
        model: NCFModel,
        preprocessor: NCFDataPreprocessor,
        device: torch.device | None = None,
    ) -> None:
        self._model = model
        self._preprocessor = preprocessor
        self._device = device or torch.device("cpu")
        self._model.to(self._device)
        self._model.eval()

    # ------------------------------------------------------------------
    # HybridRecommender-compatible interface
    # ------------------------------------------------------------------

    def predict_top_n(
        self,
        user_id: int,
        candidate_movie_ids: list[int],
        n: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Predict top-N movie recommendations for an in-app user.

        Args:
            user_id:             Our internal app user_id (integer).
            candidate_movie_ids: Movie IDs to score (already-rated movies
                                 should be excluded by the caller).
            n:                   Maximum number of results.

        Returns:
            List of (movie_id, predicted_score) sorted descending by score.
            Returns [] if the user was not seen during training (cold-start).
        """
        # Resolve in-app user_id → training user index
        user_key = f"{_APP_USER_PREFIX}{user_id}"
        user_idx = self._preprocessor.user_to_idx.get(user_key)

        if user_idx is None:
            logger.debug(
                "NCFCollaborativeAdapter: user_id=%d (key='%s') not in training set — cold start.",
                user_id,
                user_key,
            )
            return []

        if not candidate_movie_ids:
            return []

        # Map candidate movie_ids → item indices; skip unknowns
        valid_pairs: list[tuple[int, int]] = []   # (movie_id, item_idx)
        for movie_id in candidate_movie_ids:
            item_idx = self._preprocessor.movie_to_idx.get(movie_id)
            if item_idx is not None:
                valid_pairs.append((movie_id, item_idx))

        if not valid_pairs:
            logger.debug(
                "NCFCollaborativeAdapter: no candidate movies found in training index."
            )
            return []

        movie_ids_valid = [p[0] for p in valid_pairs]
        item_indices = [p[1] for p in valid_pairs]

        # Batched forward pass (no per-item loop)
        with torch.no_grad():
            user_tensor = torch.full(
                (len(item_indices),), user_idx, dtype=torch.long, device=self._device
            )
            item_tensor = torch.tensor(item_indices, dtype=torch.long, device=self._device)
            scores = self._model(user_tensor, item_tensor).cpu().numpy()

        results = list(zip(movie_ids_valid, scores.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    @property
    def is_trained(self) -> bool:
        """Always True — the adapter is only constructed around a trained model."""
        return True

    @property
    def trained_movie_ids(self) -> set[int]:
        """Return the set of internal movie_ids the model was trained on."""
        return set(self._preprocessor.movie_to_idx.keys())
