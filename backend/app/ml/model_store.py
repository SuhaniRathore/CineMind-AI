"""
ML Model Store — Milestones 12 & 14.

Handles serialization and deserialization of trained model artifacts:
  - Milestone 12: SVD (scikit-surprise) via joblib — single .pkl file.
  - Milestone 14: NCF (PyTorch) — weights .pt + preprocessor .pkl stored in
    a named subdirectory alongside the SVD artifact.


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

    # Save SVD (existing)
    store.save(trained_model, "collaborative_svd")

    # Load SVD (returns None if not found)
    model = store.load("collaborative_svd")
    if model is None:
        # model not yet trained — use fallback
        ...

    # Save NCF (Milestone 14)
    store.save_ncf(ncf_model, preprocessor, name="collaborative_ncf")

    # Load NCF (returns (NCFModel, NCFDataPreprocessor) | None)
    result = store.load_ncf("collaborative_ncf")
    if result is not None:
        ncf_model, preprocessor = result
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib

if TYPE_CHECKING:
    from app.ml.ncf_data import NCFDataPreprocessor
    from app.ml.ncf_model import NCFModel

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

    # ------------------------------------------------------------------
    # Milestone 14 — NCF (PyTorch) artifact save / load
    # ------------------------------------------------------------------

    def save_ncf(
        self,
        model: "NCFModel",
        preprocessor: "NCFDataPreprocessor",
        name: str = "collaborative_ncf",
    ) -> str:
        """
        Save a trained NCFModel and its NCFDataPreprocessor.

        Writes to:
          <models_dir>/<name>/model_weights.pt   — torch state_dict
          <models_dir>/<name>/model_config.pkl   — NCFModel.get_config()
          <models_dir>/<name>/preprocessor.pkl   — NCFDataPreprocessor

        Returns:
            Path to the artifact directory.
        """
        import torch

        artifact_dir = self._models_dir / name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights atomically
        weights_final = artifact_dir / "model_weights.pt"
        tmp_fd, tmp_weights = tempfile.mkstemp(dir=artifact_dir, prefix="_weights_", suffix=".tmp")
        try:
            os.close(tmp_fd)
            torch.save(model.state_dict(), tmp_weights)
            os.replace(tmp_weights, weights_final)
        except Exception:
            try:
                os.remove(tmp_weights)
            except OSError:
                pass
            raise

        # Save model config atomically
        config_path = artifact_dir / "model_config.pkl"
        tmp_fd, tmp_config = tempfile.mkstemp(dir=artifact_dir, prefix="_config_", suffix=".tmp")
        try:
            os.close(tmp_fd)
            joblib.dump(model.get_config(), tmp_config)
            os.replace(tmp_config, config_path)
        except Exception:
            try:
                os.remove(tmp_config)
            except OSError:
                pass
            raise

        # Save preprocessor atomically
        preprocessor_path = artifact_dir / "preprocessor.pkl"
        tmp_fd, tmp_prep = tempfile.mkstemp(dir=artifact_dir, prefix="_prep_", suffix=".tmp")
        try:
            os.close(tmp_fd)
            preprocessor.save(tmp_prep)
            os.replace(tmp_prep, preprocessor_path)
        except Exception:
            try:
                os.remove(tmp_prep)
            except OSError:
                pass
            raise

        logger.info("ModelStore: NCF artifact '%s' saved to %s", name, artifact_dir)
        return str(artifact_dir)

    def load_ncf(
        self,
        name: str = "collaborative_ncf",
    ) -> "tuple[NCFModel, NCFDataPreprocessor] | None":
        """
        Load a previously saved NCF artifact.

        Returns:
            (NCFModel, NCFDataPreprocessor) tuple, or None if the artifact
            directory or any required file is missing.
        """
        import torch

        from app.ml.ncf_data import NCFDataPreprocessor
        from app.ml.ncf_model import NCFModel

        artifact_dir = self._models_dir / name
        weights_path = artifact_dir / "model_weights.pt"
        config_path = artifact_dir / "model_config.pkl"
        preprocessor_path = artifact_dir / "preprocessor.pkl"

        for p in (weights_path, config_path, preprocessor_path):
            if not p.exists():
                logger.info("ModelStore: NCF artifact '%s' not found at %s", name, p)
                return None

        try:
            config = joblib.load(config_path)
            model = NCFModel.from_config(config)
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            model.eval()

            preprocessor = NCFDataPreprocessor.load(preprocessor_path)

            logger.info("ModelStore: NCF artifact '%s' loaded from %s", name, artifact_dir)
            return model, preprocessor
        except Exception as exc:
            logger.error("ModelStore: failed to load NCF artifact '%s': %s", name, exc)
            return None

    def ncf_exists(self, name: str = "collaborative_ncf") -> bool:
        """Return True if all NCF artifact files exist on disk."""
        artifact_dir = self._models_dir / name
        return all(
            (artifact_dir / f).exists()
            for f in ("model_weights.pt", "model_config.pkl", "preprocessor.pkl")
        )
