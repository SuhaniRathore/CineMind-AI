"""
NCF evaluation utilities — Milestone 14.

Computes point-wise recommendation metrics on a held-out validation set:
  - RMSE (Root Mean Squared Error)
  - MAE  (Mean Absolute Error)

Evaluation is performed batch-by-batch so validation predictions are not
materialised into large Python lists. This keeps memory usage practical for
large MovieLens datasets such as ml-25m.
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate_ncf(
    model: "torch.nn.Module",
    preprocessor,  # Kept in the public signature for compatibility.
    val_dataset: "torch.utils.data.Dataset",
    device: torch.device,
    k: int = 10,  # Kept for API compatibility; ranking metrics were removed.
    batch_size: int = 4096,
) -> dict[str, float]:
    """
    Evaluate an NCF model on the validation Dataset.

    RMSE and MAE are accumulated batch-by-batch to avoid storing all
    predictions and labels in memory.

    Args:
        model: Trained NCFModel (or compatible Module).
        preprocessor: Fitted preprocessor; retained for API compatibility.
        val_dataset: Validation NCFDataset.
        device: Torch device for inference.
        k: Retained for backward compatibility; unused.
        batch_size: Inference batch size.

    Returns:
        Dict with keys: rmse, mae
    """
    del preprocessor, k

    model.eval()
    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for user_batch, item_batch, label_batch in loader:
            user_batch = user_batch.to(device)
            item_batch = item_batch.to(device)
            label_batch = label_batch.to(device)

            preds = model(user_batch, item_batch)

            errors = preds - label_batch
            squared_error_sum += torch.sum(errors * errors).item()
            absolute_error_sum += torch.sum(torch.abs(errors)).item()
            sample_count += label_batch.numel()

    if sample_count == 0:
        metrics = {
            "rmse": 0.0,
            "mae": 0.0,
        }
        logger.info("NCF Evaluation: empty validation set: %s", metrics)
        return metrics

    rmse = (squared_error_sum / sample_count) ** 0.5
    mae = absolute_error_sum / sample_count

    metrics = {
        "rmse": round(float(rmse), 6),
        "mae": round(float(mae), 6),
    }

    logger.info("NCF Evaluation: %s", metrics)
    return metrics