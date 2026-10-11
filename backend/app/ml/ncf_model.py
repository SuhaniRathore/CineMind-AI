"""
Neural Collaborative Filtering (NCF / NeuMF) model — Milestone 14.

Architecture (He et al. 2017 — "Neural Collaborative Filtering"):
  - GMF branch: element-wise product of user & item embeddings
  - MLP branch: concatenation of separate user & item embeddings fed through
                a stack of fully-connected layers with ReLU + BatchNorm + Dropout
  - NeuMF output: concat(GMF output, MLP output) → FC → 1

Two output modes (controlled by `use_implicit`):
  - Explicit (default): raw scalar output, trained with MSE loss on 1–10 ratings.
  - Implicit: sigmoid output, trained with BCE loss on positive/negative samples.

Design decisions:
  - Pure PyTorch nn.Module — no HuggingFace, no TF, works on CPU.
  - GMF and MLP branches use *separate* embedding tables (standard NeuMF design).
  - BatchNorm applied before ReLU in each MLP layer for training stability.
  - All hyperparameters exposed as __init__ args for easy grid-search.

Usage::

    from app.ml.ncf_model import NCFModel

    model = NCFModel(
        n_users=162_000,
        n_items=62_000,
        gmf_dim=64,
        mlp_dim=128,
        mlp_layers=[128, 64, 32, 16],
        dropout=0.2,
        use_implicit=False,
    )
    # Forward pass
    scores = model(user_idx_tensor, item_idx_tensor)  # shape: [batch_size]
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Sequence
from collections.abc import Iterator

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@contextmanager
def singleton_batchnorm_eval(model: nn.Module, batch_size: int) -> Iterator[None]:
    """Use running BatchNorm statistics for singleton training batches."""
    if batch_size != 1:
        yield
        return

    batch_norm_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    training_states = [module.training for module in batch_norm_layers]
    try:
        for module in batch_norm_layers:
            module.eval()
        yield
    finally:
        for module, was_training in zip(batch_norm_layers, training_states):
            module.train(was_training)


class NCFModel(nn.Module):
    """
    NeuMF: Neural Matrix Factorisation combining GMF + MLP branches.

    Args:
        n_users:       Total number of distinct users in the training set.
        n_items:       Total number of distinct items (movies) in training set.
        gmf_dim:       Embedding dimension for the GMF branch.
        mlp_layers:    List of hidden layer widths for the MLP branch.
                       The first element is also the concat size of the two
                       MLP embeddings (each embedding = mlp_layers[0] // 2).
        dropout:       Dropout probability applied after each MLP hidden layer.
        use_implicit:  If True, final activation is sigmoid (BCE mode).
                       If False, raw output (MSE mode on explicit ratings).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        gmf_dim: int = 64,
        mlp_layers: Sequence[int] = (128, 64, 32, 16),
        dropout: float = 0.2,
        use_implicit: bool = False,
    ) -> None:
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.gmf_dim = gmf_dim
        self.mlp_layers = list(mlp_layers)
        self.dropout = dropout
        self.use_implicit = use_implicit

        # -----------------------------------------------------------------
        # GMF branch embeddings
        # -----------------------------------------------------------------
        self.gmf_user_emb = nn.Embedding(n_users, gmf_dim)
        self.gmf_item_emb = nn.Embedding(n_items, gmf_dim)

        # -----------------------------------------------------------------
        # MLP branch embeddings
        # The input to the first MLP layer is the concatenation of two
        # half-sized embeddings (user + item), so each embedding is
        # mlp_layers[0] // 2 dimensional.
        # -----------------------------------------------------------------
        mlp_emb_dim = mlp_layers[0] // 2
        self.mlp_user_emb = nn.Embedding(n_users, mlp_emb_dim)
        self.mlp_item_emb = nn.Embedding(n_items, mlp_emb_dim)

        # -----------------------------------------------------------------
        # MLP hidden layers
        # Build: mlp_layers[0] → mlp_layers[1] → … → mlp_layers[-1]
        # Each block: Linear → BatchNorm1d → ReLU → Dropout
        # -----------------------------------------------------------------
        mlp_blocks: list[nn.Module] = []
        in_size = mlp_layers[0]
        for out_size in mlp_layers[1:]:
            mlp_blocks.extend([
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ])
            in_size = out_size
        self.mlp_net = nn.Sequential(*mlp_blocks)

        # -----------------------------------------------------------------
        # Output layer: concat(GMF output, MLP last layer) → 1
        # -----------------------------------------------------------------
        output_in = gmf_dim + mlp_layers[-1]
        self.output_layer = nn.Linear(output_in, 1)

        # -----------------------------------------------------------------
        # Weight initialisation
        # -----------------------------------------------------------------
        self._init_weights()

        logger.info(
            "NCFModel created: %d users, %d items, gmf_dim=%d, mlp_layers=%s, implicit=%s",
            n_users,
            n_items,
            gmf_dim,
            mlp_layers,
            use_implicit,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_idx: LongTensor of shape [batch_size] — 0-based user indices.
            item_idx: LongTensor of shape [batch_size] — 0-based item indices.

        Returns:
            FloatTensor of shape [batch_size] — predicted scores.
            Range: raw real value (explicit mode) or [0, 1] (implicit mode).
        """
        # GMF branch: element-wise product
        gmf_u = self.gmf_user_emb(user_idx)   # [B, gmf_dim]
        gmf_i = self.gmf_item_emb(item_idx)   # [B, gmf_dim]
        gmf_out = gmf_u * gmf_i               # [B, gmf_dim]

        # MLP branch: concatenation → MLP stack
        mlp_u = self.mlp_user_emb(user_idx)   # [B, mlp_emb_dim]
        mlp_i = self.mlp_item_emb(item_idx)   # [B, mlp_emb_dim]
        mlp_cat = torch.cat([mlp_u, mlp_i], dim=1)  # [B, mlp_layers[0]]
        mlp_out = self.mlp_net(mlp_cat)              # [B, mlp_layers[-1]]

        # Fuse and output
        fused = torch.cat([gmf_out, mlp_out], dim=1)   # [B, gmf_dim + mlp_layers[-1]]
        logit = self.output_layer(fused).squeeze(1)     # [B]

        if self.use_implicit:
            return torch.sigmoid(logit)
        return logit

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier-uniform for linear layers; normal(0, 0.01) for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def get_config(self) -> dict:
        """Return a serialisable config dict (saved alongside weights)."""
        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "gmf_dim": self.gmf_dim,
            "mlp_layers": self.mlp_layers,
            "dropout": self.dropout,
            "use_implicit": self.use_implicit,
        }

    @classmethod
    def from_config(cls, config: dict) -> "NCFModel":
        """Reconstruct model architecture from a saved config dict."""
        return cls(**config)
