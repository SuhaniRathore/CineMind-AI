import json
import os

import pandas as pd
import pytest
import torch

from app.ml.ncf_streaming import NCFStreamingPreprocessor


def _write_sample_ratings_csv(path):
    pd.DataFrame(
        [
            [1, 10, 5.0, 1],
            [1, 20, 4.0, 2],
            [1, 30, 3.0, 3],
            [2, 10, 4.5, 4],
            [2, 20, 2.5, 5],
            [2, 30, 5.0, 6],
            [3, 10, 3.5, 7],
            [3, 20, 4.0, 8],
            [3, 30, 2.0, 9],
        ],
        columns=["userId", "movieId", "rating", "timestamp"],
    ).to_csv(path, index=False)


def test_ncf_streaming_creates_train_and_val_shards(tmp_path):
    ratings_csv = tmp_path / "ratings.csv"

    pd.DataFrame(
        [
            [1, 10, 5.0, 1],
            [1, 20, 4.0, 2],
            [1, 30, 3.0, 3],
            [2, 10, 4.5, 4],
            [2, 20, 2.5, 5],
            [2, 30, 5.0, 6],
            [3, 10, 3.5, 7],
            [3, 20, 4.0, 8],
            [3, 30, 2.0, 9],
        ],
        columns=[
            "userId",
            "movieId",
            "rating",
            "timestamp",
        ],
    ).to_csv(
        ratings_csv,
        index=False,
    )

    preprocessor = NCFStreamingPreprocessor(
        use_implicit=False
    )

    train_ds, val_ds = preprocessor.prepare_from_csv(
        ratings_csv=ratings_csv,
        ml_to_internal={
            10: 101,
            20: 102,
            30: 103,
        },
        output_dir=tmp_path / "ncf_stream",
        batch_size=2,
        chunk_size=4,
        val_fraction=0.20,
        seed=42,
    )

    assert preprocessor.n_users == 3
    assert preprocessor.n_items == 3

    assert len(train_ds) > 0
    assert len(val_ds) > 0

    assert list(
        (tmp_path / "ncf_stream").glob("train_*.npz")
    )

    assert list(
        (tmp_path / "ncf_stream").glob("val_*.npz")
    )

    train_batches = list(iter(train_ds))
    val_batches = list(iter(val_ds))

    assert len(train_batches) > 0
    assert len(val_batches) > 0

    users, items, labels = train_batches[0]

    assert users.dtype == torch.int64
    assert items.dtype == torch.int64
    assert labels.dtype == torch.float32


def test_streaming_fingerprint_tracks_content_even_when_size_and_mtime_do_not_change(
    tmp_path,
):
    ratings_csv = tmp_path / "ratings.csv"
    _write_sample_ratings_csv(ratings_csv)
    output_dir = tmp_path / "ncf_stream"
    mapping = {10: 101, 20: 102, 30: 103}

    NCFStreamingPreprocessor().prepare_from_csv(
        ratings_csv,
        mapping,
        output_dir,
        batch_size=2,
    )
    metadata_path = output_dir / "metadata.json"
    original_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    original_shards = {
        path.name: path.read_bytes()
        for path in output_dir.glob("*.npz")
    }
    original_stat = ratings_csv.stat()

    content = ratings_csv.read_text(encoding="utf-8")
    ratings_csv.write_text(content.replace("5.0", "4.0", 1), encoding="utf-8")
    os.utime(
        ratings_csv,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    changed_fingerprint = NCFStreamingPreprocessor._fingerprint(
        ratings_csv,
        mapping,
        use_implicit=False,
        val_fraction=0.1,
        seed=42,
    )

    assert ratings_csv.stat().st_size == original_stat.st_size
    assert ratings_csv.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert changed_fingerprint != original_metadata["fingerprint"]
    with pytest.raises(ValueError, match="stale or invalid"):
        NCFStreamingPreprocessor().prepare_from_csv(
            ratings_csv,
            mapping,
            output_dir,
            batch_size=2,
        )
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == original_metadata
    assert {
        path.name: path.read_bytes()
        for path in output_dir.glob("*.npz")
    } == original_shards


def test_streaming_cache_tracks_preprocessing_and_app_rating_labels(tmp_path):
    ratings_csv = tmp_path / "ratings.csv"
    _write_sample_ratings_csv(ratings_csv)
    output_dir = tmp_path / "ncf_stream"
    extra_ratings = pd.DataFrame(
        [
            ["app_17", 104, 8.0],
            ["app_17", 105, 4.0],
        ],
        columns=["ml_user_id", "movie_id", "rating"],
    )

    preprocessor = NCFStreamingPreprocessor(use_implicit=True)
    train_ds, _ = preprocessor.prepare_from_csv(
        ratings_csv,
        {10: 101, 20: 102, 30: 103},
        output_dir,
        batch_size=3,
        extra_ratings_df=extra_ratings,
    )

    assert "app_17" in preprocessor.user_to_idx
    assert preprocessor.movie_to_idx[104] >= 0
    assert preprocessor.movie_to_idx[105] >= 0
    assert json.loads((output_dir / "metadata.json").read_text())["preprocessing"] == {
        "use_implicit": True,
        "val_fraction": 0.1,
        "seed": 42,
    }

    labels_by_movie = {}
    app_user_idx = preprocessor.user_to_idx["app_17"]
    for users, items, labels in train_ds:
        for user_idx, item_idx, label in zip(
            users.tolist(),
            items.tolist(),
            labels.tolist(),
        ):
            if user_idx == app_user_idx:
                labels_by_movie[preprocessor.idx_to_movie[item_idx]] = label

    assert labels_by_movie == {104: 1.0, 105: 0.0}


def test_header_only_source_reuses_cache_without_modifying_shards(tmp_path):
    ratings_csv = tmp_path / "ratings.csv"
    _write_sample_ratings_csv(ratings_csv)
    output_dir = tmp_path / "ncf_stream"
    mapping = {10: 101, 20: 102, 30: 103}

    NCFStreamingPreprocessor().prepare_from_csv(
        ratings_csv,
        mapping,
        output_dir,
        batch_size=2,
    )
    existing_files = {
        path.name: path.read_bytes()
        for path in output_dir.glob("*.npz")
    }
    ratings_csv.write_text(
        "userId,movieId,rating,timestamp\n",
        encoding="utf-8",
    )

    train_ds, val_ds = NCFStreamingPreprocessor().prepare_from_csv(
        ratings_csv,
        mapping,
        output_dir,
        batch_size=2,
    )

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert {
        path.name: path.read_bytes()
        for path in output_dir.glob("*.npz")
    } == existing_files


def test_invalid_header_only_cache_is_preserved_and_not_rebuilt(tmp_path):
    ratings_csv = tmp_path / "ratings.csv"
    _write_sample_ratings_csv(ratings_csv)
    output_dir = tmp_path / "ncf_stream"

    NCFStreamingPreprocessor().prepare_from_csv(
        ratings_csv,
        {10: 101, 20: 102, 30: 103},
        output_dir,
        batch_size=2,
    )
    shard_files = list(output_dir.glob("*.npz"))
    existing_files = {path.name: path.read_bytes() for path in shard_files}
    metadata = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    metadata["n_train"] += 1
    (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    ratings_csv.write_text(
        "userId,movieId,rating,timestamp\n",
        encoding="utf-8",
    )

    try:
        NCFStreamingPreprocessor().prepare_from_csv(
            ratings_csv,
            {10: 101, 20: 102, 30: 103},
            output_dir,
            batch_size=2,
        )
    except ValueError as exc:
        assert "cannot rebuild" in str(exc).lower()
    else:
        raise AssertionError("Invalid cache must not be reused without source rows")

    assert {
        path.name: path.read_bytes()
        for path in output_dir.glob("*.npz")
    } == existing_files