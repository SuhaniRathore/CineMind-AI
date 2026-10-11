from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from app.ml.model_store import ModelStore
from app.ml.ncf_model import NCFModel
from app.ml.ncf_streaming import (
    NCFStreamingPreprocessor,
    NCFStreamingTrainer,
)


def test_streaming_ncf_trainer_resumes_from_epoch_checkpoint(
    tmp_path,
    monkeypatch,
):
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

    checkpoint_path = tmp_path / "checkpoints" / "candidate.pt"
    trainer = NCFStreamingTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        preprocessor=preprocessor,
        epochs=2,
        lr=0.001,
        weight_decay=1e-5,
        patience=1,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
    )

    save_checkpoint = trainer._save_checkpoint

    def save_then_interrupt(**kwargs):
        save_checkpoint(**kwargs)
        raise RuntimeError("simulated runtime interruption")

    monkeypatch.setattr(trainer, "_save_checkpoint", save_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated runtime interruption"):
        trainer.train()

    assert checkpoint_path.is_file()
    assert trainer.history[0]["epoch"] == 1

    protected_trainer = NCFStreamingTrainer(
        model=NCFModel(
            n_users=preprocessor.n_users,
            n_items=preprocessor.n_items,
            gmf_dim=8,
            mlp_layers=[8, 4],
            dropout=0.0,
        ),
        train_dataset=train_ds,
        val_dataset=val_ds,
        preprocessor=preprocessor,
        epochs=2,
        lr=0.001,
        weight_decay=1e-5,
        patience=1,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
    )
    with pytest.raises(FileExistsError, match="pass resume=True"):
        protected_trainer.train()

    resumed_model = NCFModel(
        n_users=preprocessor.n_users,
        n_items=preprocessor.n_items,
        gmf_dim=8,
        mlp_layers=[8, 4],
        dropout=0.0,
        use_implicit=False,
    )
    resumed_trainer = NCFStreamingTrainer(
        model=resumed_model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        preprocessor=preprocessor,
        epochs=2,
        lr=0.001,
        weight_decay=1e-5,
        patience=1,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        resume=True,
    )
    best_rmse = resumed_trainer.train()

    assert resumed_model.training is False
    assert isinstance(best_rmse, float)
    assert best_rmse >= 0.0
    assert [row["epoch"] for row in resumed_trainer.history] == [1, 2]
    assert all(
        row["val_rmse"] >= 0.0 and row["val_mae"] >= 0.0
        for row in resumed_trainer.history
    )
    assert resumed_trainer.elapsed_seconds > 0
    store = ModelStore(SimpleNamespace(models_dir=str(tmp_path / "models")))
    artifact_name = "collaborative_ncf_candidate_test"
    store.save_ncf(resumed_model, preprocessor, name=artifact_name)
    loaded = store.load_ncf(name=artifact_name)
    assert loaded is not None
    loaded_model, loaded_preprocessor = loaded
    with torch.no_grad():
        prediction = loaded_model(
            torch.tensor([0]),
            torch.tensor([0]),
        )
    assert loaded_preprocessor.n_users == preprocessor.n_users
    assert loaded_preprocessor.n_items == preprocessor.n_items
    assert torch.isfinite(prediction).all()


def test_streaming_ncf_trainer_rejects_mismatched_checkpoint(tmp_path):
    ratings_csv = tmp_path / "ratings.csv"
    pd.DataFrame(
        [
            [1, 10, 5.0, 1],
            [1, 20, 4.0, 2],
            [1, 30, 3.0, 3],
            [2, 10, 4.5, 4],
            [2, 20, 2.5, 5],
            [2, 30, 5.0, 6],
        ],
        columns=["userId", "movieId", "rating", "timestamp"],
    ).to_csv(ratings_csv, index=False)
    preprocessor = NCFStreamingPreprocessor(use_implicit=False)
    train_ds, val_ds = preprocessor.prepare_from_csv(
        ratings_csv,
        {10: 101, 20: 102, 30: 103},
        tmp_path / "ncf_stream",
        batch_size=2,
        chunk_size=4,
        val_fraction=0.2,
        seed=42,
    )
    checkpoint_path = tmp_path / "candidate.pt"
    NCFStreamingTrainer(
        model=NCFModel(preprocessor.n_users, preprocessor.n_items, gmf_dim=8, mlp_layers=[8, 4]),
        train_dataset=train_ds,
        val_dataset=val_ds,
        preprocessor=preprocessor,
        epochs=1,
        patience=1,
        checkpoint_path=checkpoint_path,
    ).train()

    mismatched = NCFStreamingTrainer(
        model=NCFModel(preprocessor.n_users, preprocessor.n_items, gmf_dim=8, mlp_layers=[8, 4]),
        train_dataset=train_ds,
        val_dataset=val_ds,
        preprocessor=preprocessor,
        epochs=2,
        patience=1,
        checkpoint_path=checkpoint_path,
        resume=True,
    )
    with pytest.raises(ValueError, match="configuration does not match"):
        mismatched.train()