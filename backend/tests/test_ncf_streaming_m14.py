import pandas as pd
import torch

from app.ml.ncf_streaming import NCFStreamingPreprocessor


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