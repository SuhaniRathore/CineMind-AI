import pandas as pd
import torch

from app.ml.ncf_model import NCFModel
from app.ml.ncf_streaming import (
    NCFStreamingPreprocessor,
    NCFStreamingTrainer,
)


def test_streaming_ncf_trainer_runs_one_epoch(tmp_path):
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
            [4, 10, 4.0, 10],
            [4, 20, 3.0, 11],
            [4, 30, 5.0, 12],
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

    train_ds, val_ds = (
        preprocessor.prepare_from_csv(
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
    )

    model = NCFModel(
        n_users=preprocessor.n_users,
        n_items=preprocessor.n_items,
        gmf_dim=8,
        mlp_layers=[8, 4],
        dropout=0.0,
        use_implicit=False,
    )

    trainer = NCFStreamingTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        preprocessor=preprocessor,
        epochs=1,
        lr=0.001,
        weight_decay=1e-5,
        patience=1,
        device=torch.device("cpu"),
    )

    best_rmse = trainer.train()

    assert model.training is False
    assert isinstance(best_rmse, float)
    assert best_rmse >= 0.0