"""
Memory-safe streaming NCF utilities — Milestone 14.

Large MovieLens datasets are processed from CSV chunks into on-disk NPZ
shards. Training and validation are then streamed in batches without
materialising the complete ratings dataset in RAM.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

from app.ml.ncf_data import NCFDataPreprocessor
from app.ml.ncf_model import NCFModel, singleton_batchnorm_eval

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

    @property
    def shard_dir(self) -> Path:
        return self._shard_dir

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def split(self) -> str:
        return self._split

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = int(epoch)

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

        if len(carry_users) > 0:
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
        extra_fingerprint = self._extra_ratings_fingerprint(
            extra_ratings_df
        )

        # Only fingerprint a real ratings dataset. In some deployed environments,
        # a header-only placeholder CSV may exist while the saved shards are the
        # only available copy of the prepared MovieLens data.
        source_has_rows = self._has_ratings_rows(ratings_csv)
        fingerprint = (
            self._fingerprint(
                ratings_csv,
                ml_to_internal,
                use_implicit=self.use_implicit,
                val_fraction=val_fraction,
                seed=seed,
            )
            if source_has_rows
            else None
        )
        preprocessing = {
            "use_implicit": self.use_implicit,
            "val_fraction": float(val_fraction),
            "seed": int(seed),
        }

        metadata_path = (
            output_dir / "metadata.json"
        )

        metadata: dict | None = None
        cache_valid = False
        if metadata_path.exists():
            try:
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
                train_shards = list(output_dir.glob("train_*.npz"))
                val_shards = list(output_dir.glob("val_*.npz"))
                cache_version = metadata.get("version")
                cache_valid = (
                    cache_version in (2, 3)
                    and isinstance(metadata.get("fingerprint"), str)
                    and (
                        cache_version == 2
                        or metadata.get("preprocessing") == preprocessing
                    )
                    and self._has_contiguous_indices(
                        metadata.get("user_to_idx"),
                        metadata.get("n_users"),
                    )
                    and self._has_contiguous_indices(
                        metadata.get("movie_to_idx"),
                        metadata.get("n_items"),
                    )
                    and bool(train_shards)
                    and int(metadata.get("n_train", 0)) > 0
                )

                # With genuine source data, stale cache must not be reused.
                # With no rows (missing/header-only placeholder), retain a
                # structurally valid cache instead of invalidating saved shards.
                if cache_valid and source_has_rows:
                    cache_valid = metadata.get("fingerprint") == fingerprint
                elif cache_valid and cache_version == 2:
                    cache_valid = (
                        not source_has_rows
                        and not self.use_implicit
                        and val_fraction == _VAL_FRACTION
                        and seed == 42
                    )

                # App ratings are part of the effective dataset. Do not reuse
                # a cache if their fingerprint differs from the saved one.
                if cache_valid:
                    if "extra_ratings_fingerprint" in metadata:
                        cache_valid = (
                            metadata.get("extra_ratings_fingerprint")
                            == extra_fingerprint
                        )
                    else:
                        # Legacy caches had no app-rating fingerprint. Only
                        # reuse one if there is no current or cached app data.
                        cache_valid = (
                            extra_fingerprint is None
                            and not list(output_dir.glob("train_extra_*.npz"))
                        )

                if cache_valid:
                    self._restore_metadata(metadata)
                    cache_valid = (
                        len(self.user_to_idx) == self.n_users
                        and len(self.movie_to_idx) == self.n_items
                    )
                if cache_valid:
                    n_train = self._count_shard_rows(
                        train_shards,
                        n_users=self.n_users,
                        n_items=self.n_items,
                    )
                    n_val = self._count_shard_rows(
                        val_shards,
                        n_users=self.n_users,
                        n_items=self.n_items,
                    )
                    cache_valid = (
                        n_train == int(metadata["n_train"])
                        and n_val == int(metadata.get("n_val", 0))
                    )
                if cache_valid:
                    logger.info(
                        "Using cached NCF streaming shards: %s",
                        output_dir,
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
            except Exception as exc:
                cache_valid = False
                logger.warning(
                    "Invalid NCF streaming cache; cannot reuse it: %s",
                    exc,
                )

        # Never clear shards unless a real ratings source is available to
        # rebuild them. This protects the only prepared copy of MovieLens data.
        if not cache_valid and not source_has_rows:
            raise ValueError(
                "Cannot rebuild NCF streaming shards because ratings.csv is "
                "missing or has no data rows and no valid cache is available. "
                "Existing shard files have been left untouched."
            )

        if not cache_valid and (
            metadata_path.exists()
            or any(output_dir.glob("train_*.npz"))
            or any(output_dir.glob("train_extra_*.npz"))
            or any(output_dir.glob("val_*.npz"))
        ):
            raise ValueError(
                "Existing NCF cache is stale or invalid. Its files have been "
                "left untouched; choose a new output directory or explicitly "
                "archive the existing cache before rebuilding."
            )

        if not cache_valid:
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

            if self.n_users == 0 or self.n_items == 0:
                raise ValueError(
                    "The ratings CSV and internal movie mapping produced no "
                    "usable NCF interactions. Existing shards were left untouched."
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
                next_user_idx = len(self.user_to_idx)
                for u in extra["ml_user_id"].astype(str).unique():
                    if u not in self.user_to_idx:
                        self.user_to_idx[u] = next_user_idx
                        next_user_idx += 1
                self.n_users = len(self.user_to_idx)

                next_item_idx = len(self.movie_to_idx)
                for i in extra["movie_id"].astype(int).unique():
                    if i not in self.movie_to_idx:
                        self.movie_to_idx[i] = next_item_idx
                        next_item_idx += 1
                self.n_items = len(self.movie_to_idx)
                self.idx_to_movie = {
                    idx: movie_id
                    for movie_id, idx in self.movie_to_idx.items()
                }

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

                extra_labels = self._encode_rating_labels(
                    extra["rating"].astype(float)
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

        # Recount rows from the files on disk. In particular, do not add
        # len(extra) to a cached n_train value: the previous extra shard may
        # already have been included in that metadata.
        n_train = self._count_shard_rows(
            list(output_dir.glob("train_*.npz")),
            n_users=self.n_users,
            n_items=self.n_items,
        )
        n_val = self._count_shard_rows(
            list(output_dir.glob("val_*.npz")),
            n_users=self.n_users,
            n_items=self.n_items,
        )

        self._is_fitted = True

        metadata = {
            "version": 3,
            "fingerprint": fingerprint,
            "extra_ratings_fingerprint": extra_fingerprint,
            "preprocessing": preprocessing,
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

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_dir,
                prefix="_metadata_",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name
                json.dump(metadata, temp_file)
            os.replace(temp_path, metadata_path)
        except Exception:
            if temp_path is not None:
                try:
                    os.remove(temp_path)
                except OSError:
                    logger.warning(
                        "Could not remove temporary NCF metadata file: %s",
                        temp_path,
                    )
            raise

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
        *,
        use_implicit: bool,
        val_fraction: float,
        seed: int,
    ) -> str:
        source_hash = hashlib.sha256()
        with ratings_csv.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                source_hash.update(chunk)

        payload = {
            "source_sha256": source_hash.hexdigest(),
            "mapping": sorted(
                (
                    int(key),
                    int(value),
                )
                for key, value
                in ml_to_internal.items()
            ),
            "use_implicit": use_implicit,
            "val_fraction": float(val_fraction),
            "seed": int(seed),
        }

        return hashlib.sha256(
            json.dumps(
                payload,
                separators=(",", ":"),
            ).encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _extra_ratings_fingerprint(
        extra_ratings_df: pd.DataFrame | None,
    ) -> str | None:
        """Fingerprint normalized app ratings for safe cache reuse."""
        if extra_ratings_df is None or extra_ratings_df.empty:
            return None

        required = {"ml_user_id", "movie_id", "rating"}
        if not required.issubset(extra_ratings_df.columns):
            raise ValueError(
                "Extra ratings must contain columns: "
                f"{sorted(required)}"
            )

        extra = extra_ratings_df[
            ["ml_user_id", "movie_id", "rating"]
        ].dropna()
        rows = sorted(
            (
                str(user_id),
                int(movie_id),
                float(rating),
            )
            for user_id, movie_id, rating
            in extra.itertuples(index=False, name=None)
        )
        payload = json.dumps(rows, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _has_ratings_rows(
        ratings_csv: Path,
    ) -> bool:
        """Return whether ratings.csv contains at least one data row.

        A valid header-only placeholder is treated as unavailable source data.
        An unrelated/malformed CSV raises instead of silently triggering a rebuild.
        """
        if not ratings_csv.is_file():
            return False

        try:
            columns = pd.read_csv(ratings_csv, nrows=0).columns
        except pd.errors.EmptyDataError:
            return False

        required = {"userId", "movieId", "rating"}
        if not required.issubset(set(columns)):
            raise ValueError(
                f"Ratings CSV must contain columns: {sorted(required)}"
            )

        try:
            first_row = pd.read_csv(
                ratings_csv,
                usecols=["userId", "movieId", "rating"],
                nrows=1,
            )
        except pd.errors.EmptyDataError:
            return False

        return not first_row.empty

    @staticmethod
    def _count_shard_rows(
        shards: list[Path],
        *,
        n_users: int,
        n_items: int,
    ) -> int:
        """Validate shard arrays and count their rows instead of trusting metadata."""
        total = 0
        for shard_path in shards:
            with np.load(shard_path, allow_pickle=False) as data:
                users = data["users"]
                items = data["items"]
                labels = data["labels"]
                arrays = (users, items, labels)
                if any(array.ndim != 1 for array in arrays):
                    raise ValueError(
                        f"NCF shard arrays must be one-dimensional: {shard_path}"
                    )
                lengths = tuple(len(array) for array in arrays)
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"Inconsistent array lengths in NCF shard: {shard_path}"
                )
            if not np.issubdtype(users.dtype, np.integer) or not np.issubdtype(
                items.dtype, np.integer
            ):
                raise ValueError(
                    f"NCF shard indices must be integers: {shard_path}"
                )
            if lengths[0] and (
                users.min() < 0
                or users.max() >= n_users
                or items.min() < 0
                or items.max() >= n_items
            ):
                raise ValueError(
                    f"NCF shard index is outside the metadata maps: {shard_path}"
                )
            if not np.issubdtype(labels.dtype, np.number) or (
                not np.isfinite(labels).all()
                or (lengths[0] and (labels.min() < 0 or labels.max() > 1))
            ):
                raise ValueError(
                    f"NCF shard labels must be finite values in [0, 1]: {shard_path}"
                )
            total += lengths[0]
        return total

    @staticmethod
    def _has_contiguous_indices(mapping: object, count: object) -> bool:
        if (
            not isinstance(mapping, dict)
            or not isinstance(count, int)
            or isinstance(count, bool)
        ):
            return False
        indices = list(mapping.values())
        if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
            return False
        expected_count = count
        indices.sort()
        return (
            expected_count > 0
            and len(mapping) == expected_count
            and indices == list(range(expected_count))
        )

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
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
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
        self._checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else None
        )
        self._resume = resume
        self.history: list[dict[str, float | int]] = []
        self.elapsed_seconds = 0.0

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
        start_epoch = 0
        completed = False

        if self._resume:
            if self._checkpoint_path is None:
                raise ValueError("resume requires checkpoint_path")
            if not self._checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"NCF checkpoint does not exist: {self._checkpoint_path}"
                )
            checkpoint = self._load_checkpoint()
            if checkpoint.get("config") != self._checkpoint_config():
                raise ValueError(
                    "NCF checkpoint configuration does not match this run"
                )
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            best_rmse = float(checkpoint["best_rmse"])
            best_state = checkpoint["best_state"]
            no_improve = int(checkpoint["no_improve"])
            start_epoch = int(checkpoint["epoch"])
            self.history = list(checkpoint["history"])
            self.elapsed_seconds = float(checkpoint["elapsed_seconds"])
            completed = bool(checkpoint["completed"])
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if checkpoint["cuda_rng_states"] is not None:
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "Checkpoint contains CUDA RNG state but CUDA is unavailable"
                    )
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in checkpoint["cuda_rng_states"]]
                )
            self._train_ds.set_epoch(start_epoch)
        elif self._checkpoint_path is not None and self._checkpoint_path.exists():
            raise FileExistsError(
                f"NCF checkpoint already exists; pass resume=True to continue: "
                f"{self._checkpoint_path}"
            )

        if completed:
            if best_state is not None:
                model.load_state_dict(best_state)
            return float(best_rmse)

        run_started = time.perf_counter()
        for epoch in range(start_epoch + 1, self._epochs + 1):
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

                with singleton_batchnorm_eval(model, user_batch.shape[0]):
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
            epoch_elapsed = time.perf_counter() - run_started
            elapsed_seconds = self.elapsed_seconds + epoch_elapsed

            logger.info(
                "Epoch %d/%d | train_loss=%.6f "
                "| val_rmse=%.6f | val_mae=%.6f",
                epoch,
                self._epochs,
                train_loss,
                rmse,
                mae,
            )
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_rmse": float(rmse),
                    "val_mae": float(mae),
                    "elapsed_seconds": float(elapsed_seconds),
                }
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

            stopped_early = no_improve >= self._patience
            completed = stopped_early or epoch == self._epochs
            if self._checkpoint_path is not None:
                self.elapsed_seconds = elapsed_seconds
                self._save_checkpoint(
                    optimizer=optimizer,
                    best_rmse=best_rmse,
                    best_state=best_state,
                    no_improve=no_improve,
                    epoch=epoch,
                    completed=completed,
                )

            if stopped_early:
                logger.info(
                    "NCF early stopping at epoch %d",
                    epoch,
                )
                break

        if best_state is not None:
            model.load_state_dict(
                best_state
            )

        if self._checkpoint_path is None:
            self.elapsed_seconds += time.perf_counter() - run_started

        return float(
            best_rmse
        )

    def _checkpoint_config(self) -> dict:
        if not isinstance(self._model, NCFModel):
            raise TypeError("Resumable NCF checkpoints require an NCFModel")
        metadata_path = self._train_ds.shard_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {
            "model": self._model.get_config(),
            "epochs": self._epochs,
            "lr": self._lr,
            "weight_decay": self._weight_decay,
            "patience": self._patience,
            "batch_size": self._train_ds.batch_size,
            "seed": self._train_ds.seed,
            "split": self._train_ds.split,
            "shuffle": self._train_ds.shuffle,
            "fingerprint": metadata.get("fingerprint"),
            "n_train": int(metadata["n_train"]),
            "n_val": int(metadata["n_val"]),
        }

    def _load_checkpoint(self) -> dict:
        try:
            checkpoint = torch.load(
                self._checkpoint_path,
                map_location=self._device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                self._checkpoint_path,
                map_location=self._device,
            )
        if checkpoint.get("version") != 1:
            raise ValueError("Unsupported NCF checkpoint version")
        return checkpoint

    def _save_checkpoint(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        best_rmse: float,
        best_state: dict | None,
        no_improve: int,
        epoch: int,
        completed: bool,
    ) -> None:
        if self._checkpoint_path is None:
            return
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "version": 1,
            "config": self._checkpoint_config(),
            "model_state": self._model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_rmse": best_rmse,
            "best_state": best_state,
            "no_improve": no_improve,
            "epoch": epoch,
            "history": self.history,
            "elapsed_seconds": self.elapsed_seconds,
            "completed": completed,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }
        fd, temp_path = tempfile.mkstemp(
            dir=self._checkpoint_path.parent,
            prefix="_ncf_checkpoint_",
            suffix=".tmp",
        )
        try:
            os.close(fd)
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, self._checkpoint_path)
        except Exception:
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Could not remove temporary NCF checkpoint: %s", temp_path)
            raise

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
