"""
Memory-safe streaming NCF utilities â€” Milestone 14.

Large MovieLens datasets are processed from CSV chunks into on-disk NPZ
shards. Training and validation are then streamed in batches without
materialising the complete ratings dataset in RAM.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

from app.ml.ncf_data import NCFDataPreprocessor

logger = logging.getLogger(__name__)

_VAL_FRACTION = 0.10
_DEFAULT_CHUNK_SIZE = 250_000


class NCFStreamingDataset(IterableDataset):
    """Stream train/validation batches from on-disk NPZ shards."""

    def __init__(
        self,
        shard_dir: str | Path,
        *,
        split: str,
        batch_size: int,
        num_rows: int,
        shuffle: bool,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self._shard_dir = Path(shard_dir)
        self._split = split
        self._batch_size = int(batch_size)
        self._num_rows = int(num_rows)
        self._shuffle = bool(shuffle)
        self._seed = int(seed)
        self._epoch = 0

        self._refresh_shards()

    def _refresh_shards(self) -> None:
        self._shards = sorted(
            self._shard_dir.glob(
                f"{self._split}_*.npz"
            )
        )

    def __len__(self) -> int:
        return self._num_rows

    def __iter__(self):
        self._refresh_shards()

        if not self._shards:
            return

        rng = np.random.default_rng(
            self._seed + self._epoch
        )

        self._epoch += 1

        shard_order = np.arange(
            len(self._shards)
        )

        if self._shuffle:
            rng.shuffle(shard_order)

        # Small carry-over buffer so batches can cross shard boundaries.
        carry_users = np.empty(
            0,
            dtype=np.int32,
        )
        carry_items = np.empty(
            0,
            dtype=np.int32,
        )
        carry_labels = np.empty(
            0,
            dtype=np.float32,
        )

        for shard_index in shard_order:
            shard_path = self._shards[
                shard_index
            ]

            with np.load(
                shard_path,
                allow_pickle=False,
            ) as data:
                users = np.asarray(
                    data["users"],
                    dtype=np.int32,
                )

                items = np.asarray(
                    data["items"],
                    dtype=np.int32,
                )

                labels = np.asarray(
                    data["labels"],
                    dtype=np.float32,
                )

            if self._shuffle:
                order = rng.permutation(
                    len(users)
                )

                users = users[order]
                items = items[order]
                labels = labels[order]

            if len(carry_users) > 0:
                users = np.concatenate(
                    [
                        carry_users,
                        users,
                    ]
                )

                items = np.concatenate(
                    [
                        carry_items,
                        items,
                    ]
                )

                labels = np.concatenate(
                    [
                        carry_labels,
                        labels,
                    ]
                )

                carry_users = np.empty(
                    0,
                    dtype=np.int32,
                )
                carry_items = np.empty(
                    0,
                    dtype=np.int32,
                )
                carry_labels = np.empty(
                    0,
                    dtype=np.float32,
                )

            full_rows = (
                len(users)
                // self._batch_size
            ) * self._batch_size

            for start in range(
                0,
                full_rows,
                self._batch_size,
            ):
                end = (
                    start
                    + self._batch_size
                )

                yield (
                    torch.from_numpy(
                        users[start:end]
                        .astype(
                            np.int64,
                            copy=False,
                        )
                    ),
                    torch.from_numpy(
                        items[start:end]
                        .astype(
                            np.int64,
                            copy=False,
                        )
                    ),
                    torch.from_numpy(
                        labels[start:end]
                        .astype(
                            np.float32,
                            copy=False,
                        )
                    ),
                )

            if full_rows < len(users):
                carry_users = users[
                    full_rows:
                ].copy()

                carry_items = items[
                    full_rows:
                ].copy()

                carry_labels = labels[
                    full_rows:
                ].copy()

        if len(carry_users) > 1:
            yield (
                torch.from_numpy(
                    carry_users.astype(
                        np.int64,
                        copy=False,
                    )
                ),
                torch.from_numpy(
                    carry_items.astype(
                        np.int64,
                        copy=False,
                    )
                ),
                torch.from_numpy(
                    carry_labels.astype(
                        np.float32,
                        copy=False,
                    )
                ),
            )


class NCFStreamingPreprocessor(
    NCFDataPreprocessor
):
    """
    Build compact user/movie maps and on-disk NCF shards from ratings.csv.
    """

    def prepare_from_csv(
        self,
        ratings_csv: str | Path,
        ml_to_internal: dict[int, int],
        output_dir: str | Path,
        *,
        batch_size: int = 1024,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        val_fraction: float = _VAL_FRACTION,
        seed: int = 42,
        extra_ratings_df: pd.DataFrame | None = None,
    ) -> tuple[
        NCFStreamingDataset,
        NCFStreamingDataset,
    ]:
        ratings_csv = Path(
            ratings_csv
        )

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        has_extra = (
            extra_ratings_df is not None
            and not extra_ratings_df.empty
        )

        fingerprint = self._fingerprint(
            ratings_csv,
            ml_to_internal,
        )

        metadata_path = (
            output_dir / "metadata.json"
        )

        if (
            not has_extra
            and metadata_path.exists()
        ):
            try:
                metadata = json.loads(
                    metadata_path.read_text(
                        encoding="utf-8"
                    )
                )

                cache_valid = (
                    metadata.get(
                        "fingerprint"
                    )
                    == fingerprint
                    and bool(
                        list(
                            output_dir.glob(
                                "train_*.npz"
                            )
                        )
                    )
                    and (
                        int(
                            metadata.get(
                                "n_val",
                                0,
                            )
                        )
                        == 0
                        or bool(
                            list(
                                output_dir.glob(
                                    "val_*.npz"
                                )
                            )
                        )
                    )
                )

                if cache_valid:
                    self._restore_metadata(
                        metadata
                    )

                    logger.info(
                        "Using cached NCF streaming shards: %s",
                        output_dir,
                    )

                    return (
                        NCFStreamingDataset(
                            output_dir,
                            split="train",
                            batch_size=batch_size,
                            num_rows=metadata[
                                "n_train"
                            ],
                            shuffle=True,
                            seed=seed,
                        ),
                        NCFStreamingDataset(
                            output_dir,
                            split="val",
                            batch_size=batch_size,
                            num_rows=metadata[
                                "n_val"
                            ],
                            shuffle=False,
                            seed=seed,
                        ),
                    )

            except Exception:
                logger.warning(
                    "Invalid NCF streaming cache; rebuilding."
                )

        self._clear_output(
            output_dir
        )

        # --------------------------------------------------------------
        # PASS 1
        # Discover all valid users and internal movies.
        # --------------------------------------------------------------
        user_ids: set[int] = set()
        movie_ids: set[int] = set()

        logger.info(
            "NCF streaming pass 1 starting..."
        )

        for chunk in pd.read_csv(
            ratings_csv,
            usecols=[
                "userId",
                "movieId",
                "rating",
            ],
            dtype={
                "userId": np.int32,
                "movieId": np.int32,
                "rating": np.float32,
            },
            chunksize=chunk_size,
        ):
            internal_ids = (
                chunk["movieId"]
                .map(ml_to_internal)
            )

            valid = internal_ids.notna()

            if not valid.any():
                continue

            user_ids.update(
                chunk.loc[
                    valid,
                    "userId",
                ]
                .astype(int)
                .unique()
                .tolist()
            )

            movie_ids.update(
                internal_ids.loc[
                    valid
                ]
                .astype(int)
                .unique()
                .tolist()
            )

        # --------------------------------------------------------------
        # Add in-app users and movies.
        # --------------------------------------------------------------
        extra_user_keys: list[str] = []

        if has_extra:
            extra = extra_ratings_df[
                [
                    "ml_user_id",
                    "movie_id",
                    "rating",
                ]
            ].dropna()

            extra_user_keys = sorted(
                extra[
                    "ml_user_id"
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            movie_ids.update(
                extra[
                    "movie_id"
                ]
                .astype(int)
                .unique()
                .tolist()
            )

        self.user_to_idx = {
            f"ml_{user_id}": idx
            for idx, user_id
            in enumerate(
                sorted(user_ids)
            )
        }

        next_user_idx = len(
            self.user_to_idx
        )

        for user_key in extra_user_keys:
            if user_key not in self.user_to_idx:
                self.user_to_idx[
                    user_key
                ] = next_user_idx

                next_user_idx += 1

        self.movie_to_idx = {
            movie_id: idx
            for idx, movie_id
            in enumerate(
                sorted(movie_ids)
            )
        }

        self.idx_to_movie = {
            idx: movie_id
            for movie_id, idx
            in self.movie_to_idx.items()
        }

        self.n_users = len(
            self.user_to_idx
        )

        self.n_items = len(
            self.movie_to_idx
        )

        # --------------------------------------------------------------
        # PASS 2
        # Create train and validation shards.
        # --------------------------------------------------------------
        logger.info(
            "NCF streaming pass 2 starting..."
        )

        rng = np.random.default_rng(
            seed
        )

        seen_users: set[int] = set()

        train_index = 0
        val_index = 0

        n_train = 0
        n_val = 0

        for chunk in pd.read_csv(
            ratings_csv,
            usecols=[
                "userId",
                "movieId",
                "rating",
            ],
            dtype={
                "userId": np.int32,
                "movieId": np.int32,
                "rating": np.float32,
            },
            chunksize=chunk_size,
        ):
            chunk[
                "internal_movie_id"
            ] = (
                chunk["movieId"]
                .map(ml_to_internal)
            )

            chunk = chunk.dropna(
                subset=[
                    "internal_movie_id"
                ]
            )

            if chunk.empty:
                continue

            already_seen = (
                chunk["userId"].isin(
                    seen_users
                )
            )

            first_new_user_row = (
                ~chunk["userId"].duplicated(
                    keep="first"
                )
            )

            first_global_interaction = (
                (~already_seen)
                & first_new_user_row
            )

            seen_users.update(
                chunk[
                    "userId"
                ]
                .astype(int)
                .unique()
                .tolist()
            )

            random_values = rng.random(
                len(chunk)
            )

            is_val = (
                ~first_global_interaction.to_numpy()
            ) & (
                random_values
                >= (
                    1.0
                    - val_fraction
                )
            )

            users = (
                chunk["userId"]
                .astype(int)
                .map(
                    lambda user_id:
                    self.user_to_idx[
                        f"ml_{user_id}"
                    ]
                )
                .to_numpy(
                    dtype=np.int32
                )
            )

            items = (
                chunk[
                    "internal_movie_id"
                ]
                .astype(int)
                .map(
                    self.movie_to_idx
                )
                .to_numpy(
                    dtype=np.int32
                )
            )

            ratings = (
                chunk["rating"]
                .to_numpy(
                    dtype=np.float32
                )
            )

            if self.use_implicit:
                labels = (
                    ratings
                    >= 7.0
                ).astype(
                    np.float32
                )
            else:
                labels = (
                    (ratings * 2.0)
                    - 1.0
                ) / 9.0

            train_mask = ~is_val

            if train_mask.any():
                np.savez(
                    output_dir
                    / f"train_{train_index:06d}.npz",
                    users=users[
                        train_mask
                    ],
                    items=items[
                        train_mask
                    ],
                    labels=labels[
                        train_mask
                    ],
                )

                train_index += 1

                n_train += int(
                    train_mask.sum()
                )

            if is_val.any():
                np.savez(
                    output_dir
                    / f"val_{val_index:06d}.npz",
                    users=users[
                        is_val
                    ],
                    items=items[
                        is_val
                    ],
                    labels=labels[
                        is_val
                    ],
                )

                val_index += 1

                n_val += int(
                    is_val.sum()
                )

        # --------------------------------------------------------------
        # Add in-app ratings to training.
        # --------------------------------------------------------------
        if has_extra:
            extra = extra_ratings_df[
                [
                    "ml_user_id",
                    "movie_id",
                    "rating",
                ]
            ].dropna()

            if not extra.empty:
                extra_users = (
                    extra[
                        "ml_user_id"
                    ]
                    .astype(str)
                    .map(
                        self.user_to_idx
                    )
                    .to_numpy(
                        dtype=np.int32
                    )
                )

                extra_items = (
                    extra[
                        "movie_id"
                    ]
                    .astype(int)
                    .map(
                        self.movie_to_idx
                    )
                    .to_numpy(
                        dtype=np.int32
                    )
                )

                extra_labels = (
                    (
                        extra[
                            "rating"
                        ].astype(float)
                        - 1.0
                    ) / 9.0
                ).to_numpy(
                    dtype=np.float32
                )

                np.savez(
                    output_dir
                    / "train_extra_000000.npz",
                    users=extra_users,
                    items=extra_items,
                    labels=extra_labels,
                )

                n_train += len(
                    extra
                )

        self._is_fitted = True

        metadata = {
            "version": 2,
            "fingerprint": fingerprint,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_train": n_train,
            "n_val": n_val,
            "user_to_idx": self.user_to_idx,
            "movie_to_idx": {
                str(movie_id): idx
                for movie_id, idx
                in self.movie_to_idx.items()
            },
        }

        metadata_path.write_text(
            json.dumps(
                metadata
            ),
            encoding="utf-8",
        )

        logger.info(
            "NCF streaming dataset ready: "
            "users=%d items=%d train=%d val=%d",
            self.n_users,
            self.n_items,
            n_train,
            n_val,
        )

        return (
            NCFStreamingDataset(
                output_dir,
                split="train",
                batch_size=batch_size,
                num_rows=n_train,
                shuffle=True,
                seed=seed,
            ),
            NCFStreamingDataset(
                output_dir,
                split="val",
                batch_size=batch_size,
                num_rows=n_val,
                shuffle=False,
                seed=seed,
            ),
        )

    @staticmethod
    def _fingerprint(
        ratings_csv: Path,
        ml_to_internal: dict[int, int],
    ) -> str:
        payload = {
            "size": ratings_csv.stat().st_size,
            "mtime_ns": ratings_csv.stat().st_mtime_ns,
            "mapping": sorted(
                (
                    int(key),
                    int(value),
                )
                for key, value
                in ml_to_internal.items()
            ),
        }

        return hashlib.sha256(
            json.dumps(
                payload,
                separators=(",", ":"),
            ).encode(
                "utf-8"
            )
        ).hexdigest()

    def _restore_metadata(
        self,
        metadata: dict,
    ) -> None:
        self.user_to_idx = {
            str(key): int(value)
            for key, value
            in metadata[
                "user_to_idx"
            ].items()
        }

        self.movie_to_idx = {
            int(key): int(value)
            for key, value
            in metadata[
                "movie_to_idx"
            ].items()
        }

        self.idx_to_movie = {
            idx: movie_id
            for movie_id, idx
            in self.movie_to_idx.items()
        }

        self.n_users = int(
            metadata["n_users"]
        )

        self.n_items = int(
            metadata["n_items"]
        )

        self._is_fitted = True

    @staticmethod
    def _clear_output(
        output_dir: Path,
    ) -> None:
        for pattern in (
            "train_*.npz",
            "val_*.npz",
        ):
            for path in output_dir.glob(
                pattern
            ):
                try:
                    path.unlink()
                except OSError:
                    pass

        metadata = (
            output_dir
            / "metadata.json"
        )

        try:
            metadata.unlink()
        except OSError:
            pass


class NCFStreamingTrainer:
    """Train NCF directly from streaming datasets."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: NCFStreamingDataset,
        val_dataset: NCFStreamingDataset,
        preprocessor: NCFDataPreprocessor,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 3,
        device: torch.device | None = None,
    ) -> None:
        self._model = model
        self._train_ds = train_dataset
        self._val_ds = val_dataset
        self._preprocessor = preprocessor
        self._epochs = epochs
        self._lr = lr
        self._weight_decay = weight_decay
        self._patience = patience
        self._device = (
            device
            or torch.device("cpu")
        )

    def train(self) -> float:
        model = self._model.to(
            self._device
        )

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        best_rmse = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(
            1,
            self._epochs + 1,
        ):
            model.train()

            total_loss = 0.0
            batch_count = 0

            for (
                user_batch,
                item_batch,
                label_batch,
            ) in self._train_ds:
                user_batch = user_batch.to(
                    self._device
                )

                item_batch = item_batch.to(
                    self._device
                )

                label_batch = label_batch.to(
                    self._device
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                predictions = model(
                    user_batch,
                    item_batch,
                )

                loss = criterion(
                    predictions,
                    label_batch,
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            train_loss = (
                total_loss
                / max(
                    batch_count,
                    1,
                )
            )

            rmse, mae = self._evaluate()

            logger.info(
                "Epoch %d/%d | train_loss=%.6f "
                "| val_rmse=%.6f | val_mae=%.6f",
                epoch,
                self._epochs,
                train_loss,
                rmse,
                mae,
            )

            if rmse < (
                best_rmse
                - 1e-6
            ):
                best_rmse = rmse

                best_state = copy.deepcopy(
                    model.state_dict()
                )

                no_improve = 0

            else:
                no_improve += 1

                if (
                    no_improve
                    >= self._patience
                ):
                    logger.info(
                        "NCF early stopping at epoch %d",
                        epoch,
                    )
                    break

        if best_state is not None:
            model.load_state_dict(
                best_state
            )

        return float(
            best_rmse
        )

    def _evaluate(
        self,
    ) -> tuple[float, float]:
        self._model.eval()

        squared_sum = 0.0
        absolute_sum = 0.0
        count = 0

        with torch.no_grad():
            for (
                user_batch,
                item_batch,
                label_batch,
            ) in self._val_ds:
                user_batch = user_batch.to(
                    self._device
                )

                item_batch = item_batch.to(
                    self._device
                )

                label_batch = label_batch.to(
                    self._device
                )

                predictions = self._model(
                    user_batch,
                    item_batch,
                )

                errors = (
                    predictions
                    - label_batch
                )

                squared_sum += torch.sum(
                    errors * errors
                ).item()

                absolute_sum += torch.sum(
                    torch.abs(errors)
                ).item()

                count += label_batch.numel()

        if count == 0:
            return 0.0, 0.0

        rmse = (
            squared_sum / count
        ) ** 0.5

        mae = (
            absolute_sum / count
        )

        return (
            float(rmse),
            float(mae),
        )
