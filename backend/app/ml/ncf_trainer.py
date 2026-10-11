"""
NCF training loop — Milestone 14.

Encapsulates the full training lifecycle for the NCFModel:
  - Builds a DataLoader for the training Dataset.
  - Runs Adam optimiser with configurable learning rate and weight decay.
  - Evaluates on the validation Dataset after each epoch.
  - Early-stops on validation RMSE (patience configurable).
  - Keeps the best-validation-loss checkpoint in memory and restores it
    before returning.
  - Logs per-epoch metrics to the Python logger (INFO level) so they appear
    in the API/server log and are observable without tensorboard.

Design decisions:
  - CPU-first: device defaults to CPU; pass device=torch.device("cuda") if
    a GPU is available.
  - Loss:
      Explicit mode → MSELoss (on normalised 0–1 labels)
      Implicit mode → BCELoss (on binary labels)
  - No learning-rate scheduler in the default config; easily added by
    subclassing or passing a scheduler_fn argument.

Usage::

    from app.ml.ncf_trainer import NCFTrainer
    from app.ml.ncf_model import NCFModel, singleton_batchnorm_eval
    from app.ml.ncf_data import NCFDataPreprocessor

    pre = NCFDataPreprocessor(use_implicit=False)
    train_ds, val_ds = pre.fit_transform(ratings_df)

    model = NCFModel(n_users=pre.n_users, n_items=pre.n_items)
    trainer = NCFTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        preprocessor=pre,
        epochs=20,
        lr=1e-3,
        batch_size=1024,
        patience=3,
    )
    best_rmse = trainer.train()
"""

from __future__ import annotations

import copy
import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.ml.ncf_data import NCFDataPreprocessor, NCFDataset
from app.ml.ncf_evaluation import evaluate_ncf
from app.ml.ncf_model import singleton_batchnorm_eval

logger = logging.getLogger(__name__)


class NCFTrainer:
    """
    Training orchestrator for NCFModel.

    Args:
        model:        NCFModel instance (untrained).
        train_dataset: NCFDataset for training.
        val_dataset:   NCFDataset for validation / early-stop.
        preprocessor:  Fitted NCFDataPreprocessor (passed to evaluate_ncf).
        epochs:       Maximum number of training epochs.
        lr:           Adam learning rate.
        batch_size:   DataLoader batch size.
        weight_decay: Adam L2 regularisation weight.
        patience:     Early-stop patience (epochs without val RMSE improvement).
        device:       Torch device. Defaults to CPU.
        eval_k:       Cutoff k for Precision/Recall/NDCG evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: NCFDataset,
        val_dataset: NCFDataset,
        preprocessor: NCFDataPreprocessor,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 1024,
        weight_decay: float = 1e-5,
        patience: int = 3,
        device: torch.device | None = None,
        eval_k: int = 10,
    ) -> None:
        self._model = model
        self._train_ds = train_dataset
        self._val_ds = val_dataset
        self._preprocessor = preprocessor
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._weight_decay = weight_decay
        self._patience = patience
        self._device = device or torch.device("cpu")
        self._eval_k = eval_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> float:
        """
        Run the full training loop.

        Returns:
            Best validation RMSE achieved (lower = better).
            The model's weights are restored to the best checkpoint before
            this method returns.
        """
        model = self._model.to(self._device)

        # Loss function
        if self._preprocessor.use_implicit:
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()

        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        train_loader = DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=0,   # 0 = main process; avoids Windows multiprocessing issues
            pin_memory=(self._device.type == "cuda"),
        )

        best_val_rmse = float("inf")
        best_state: dict | None = None
        epochs_no_improve = 0

        logger.info(
            "NCFTrainer: starting training — %d epochs, lr=%.4f, batch=%d, device=%s",
            self._epochs,
            self._lr,
            self._batch_size,
            self._device,
        )

        for epoch in range(1, self._epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(model, train_loader, criterion, optimiser)
            epoch_time = time.time() - t0

            # Validation
            val_metrics = evaluate_ncf(
                model,
                self._preprocessor,
                self._val_ds,
                self._device,
                k=self._eval_k,
            )
            val_rmse = val_metrics["rmse"]

            logger.info(
                "Epoch %d/%d | train_loss=%.5f | val_rmse=%.5f | val_mae=%.5f | %.1fs",
                epoch,
                self._epochs,
                train_loss,
                val_rmse,
                val_metrics["mae"],
                epoch_time,
            )

            # Early stopping — track best on val RMSE
            if val_rmse < best_val_rmse - 1e-6:
                best_val_rmse = val_rmse
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                logger.info("  ↳ New best val_rmse=%.5f — checkpoint saved.", best_val_rmse)
            else:
                epochs_no_improve += 1
                logger.info(
                    "  ↳ No improvement for %d/%d epochs.",
                    epochs_no_improve,
                    self._patience,
                )
                if epochs_no_improve >= self._patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d).",
                        epoch,
                        self._patience,
                    )
                    break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)
            logger.info("NCFTrainer: restored best checkpoint (val_rmse=%.5f).", best_val_rmse)

        logger.info("NCFTrainer: training complete. Best val_rmse=%.5f", best_val_rmse)
        return best_val_rmse

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimiser: torch.optim.Optimizer,
    ) -> float:
        """Run a single training epoch and return the mean batch loss."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for user_batch, item_batch, label_batch in loader:
            user_batch = user_batch.to(self._device)
            item_batch = item_batch.to(self._device)
            label_batch = label_batch.to(self._device)

            optimiser.zero_grad(set_to_none=True)
            with singleton_batchnorm_eval(model, user_batch.shape[0]):
                preds = model(user_batch, item_batch)
            loss = criterion(preds, label_batch)
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)
