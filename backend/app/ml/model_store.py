"""
ML Model Store — Milestone 12.

Handles serialization and deserialization of trained model artifacts using
joblib (efficient for numpy/scipy objects like scikit-surprise SVD models).

Architecture decisions:
  - Model artifacts are stored in MODELS_DIR (configured via settings).
  - In Docker this should be a named volume so artifacts survive container
    restarts (configured in docker-compose.yml as the 'mldata' volume).
  - Artifact filenames follow the pattern: <name>.pkl
  - save() is atomic — writes to a temp file then renames to prevent corrupt
    partially-written artifacts from being loaded.

Usage::

    from app.ml.model_store import ModelStore
    from app.core.config import get_settings

    store = ModelStore(get_settings())

    # Save
    store.save(trained_model, "collaborative_svd")

    # Load (returns None if not found)
    model = store.load("collaborative_svd")
    if model is None:
        # model not yet trained — use fallback
        ...
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


class ModelStore:
    """
    Persist and retrieve trained ML model objects using joblib.

    Args:
        settings: Application settings (provides models_dir path).
    """

    def __init__(self, settings) -> None:
        self._models_dir = Path(settings.models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)

    def save(self, obj: Any, name: str) -> str:
        """
        Serialize and save a model artifact.

        Uses an atomic write: serializes to a temp file in the same directory,
        then renames to the final path. This prevents partially-written files
        from being loaded if the process is interrupted.

        Args:
            obj: The model object to serialize (any joblib-serializable object).
            name: Logical name for the artifact (without extension).

        Returns:
            Absolute path to the saved artifact file.
        """
        final_path = self._models_dir / f"{name}.pkl"
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._models_dir,
            prefix=f"_{name}_",
            suffix=".tmp",
        )
        try:
            os.close(tmp_fd)
            joblib.dump(obj, tmp_path)
            os.replace(tmp_path, final_path)  # atomic rename
            logger.info("ModelStore: saved artifact '%s' to %s", name, final_path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

        return str(final_path)

    def load(self, name: str) -> Any | None:
        """
        Load a model artifact by name.

        Args:
            name: Logical name for the artifact (without extension).

        Returns:
            The deserialized object, or None if the artifact does not exist.
        """
        path = self._models_dir / f"{name}.pkl"
        if not path.exists():
            logger.info("ModelStore: artifact '%s' not found at %s", name, path)
            return None

        try:
            obj = joblib.load(path)
            logger.info("ModelStore: loaded artifact '%s' from %s", name, path)
            return obj
        except Exception as exc:
            logger.error("ModelStore: failed to load '%s': %s", name, exc)
            return None

    def exists(self, name: str) -> bool:
        """Return True if the artifact file exists on disk."""
        return (self._models_dir / f"{name}.pkl").exists()

    def artifact_path(self, name: str) -> str:
        """Return the expected artifact path (whether or not it exists yet)."""
        return str(self._models_dir / f"{name}.pkl")
