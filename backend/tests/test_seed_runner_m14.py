from unittest.mock import Mock

import pandas as pd

from app.core.config import Settings
from app.services.data_pipeline.movielens_loader import (
    MovieLensLoadResult,
)
from app.services.data_pipeline.seed_runner import SeedRunner


def test_seed_runner_trains_ncf_without_svd(monkeypatch):
    settings = Settings(
        jwt_secret_key="test-secret",
        jwt_algorithm="HS256",
        database_url="sqlite:///:memory:",
        tmdb_api_key="test-fake-key",
    )

    class FakeQuery:
        def all(self):
            return []

    class FakeDB:
        def query(self, model):
            return FakeQuery()

        def commit(self):
            pass

    runner = SeedRunner(
        db=FakeDB(),
        tmdb_client=Mock(),
        settings=settings,
    )

    training_df = pd.DataFrame(
        [
            {
                "ml_user_id": "ml_1",
                "movie_id": 1,
                "rating": 8.0,
            },
            {
                "ml_user_id": "ml_1",
                "movie_id": 2,
                "rating": 9.0,
            },
            {
                "ml_user_id": "ml_2",
                "movie_id": 1,
                "rating": 7.0,
            },
        ]
    )

    def fake_load(_db, **kwargs):
        return MovieLensLoadResult(
            movies_linked=2,
            training_ratings_df=training_df,
        )

    monkeypatch.setattr(
        runner._loader,
        "load",
        fake_load,
    )

    def fail_svd(*args, **kwargs):
        raise AssertionError(
            "SVD training must not run"
        )

    monkeypatch.setattr(
        "app.services.data_pipeline.seed_runner.CollaborativeFilteringModel",
        fail_svd,
    )

    ncf_called = {}

    def fake_ncf_training(df, result):
        ncf_called["rows"] = len(df)
        result.ncf_model_trained = True
        result.ncf_model_path = "fake_ncf_model"

    monkeypatch.setattr(
        runner,
        "_run_ncf_training",
        fake_ncf_training,
    )

    result = runner.run(
        download=False,
        enrich=False,
        train=False,
        train_ncf=True,
    )

    assert ncf_called["rows"] == 3
    assert result.ncf_model_trained is True
    assert result.ncf_model_path == "fake_ncf_model"
    assert result.model_trained is False
    assert result.model_path == ""
    assert result.errors == []