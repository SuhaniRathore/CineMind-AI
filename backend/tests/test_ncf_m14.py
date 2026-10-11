import pytest
import pandas as pd
import numpy as np
import torch

from app.ml.ncf_data import NCFDataPreprocessor
from app.ml.ncf_model import NCFModel
from app.ml.ncf_adapter import NCFCollaborativeAdapter

def test_ncf_model_forward_explicit():
    model = NCFModel(n_users=10, n_items=10, gmf_dim=8, mlp_layers=[16, 8], use_implicit=False)
    user_idx = torch.tensor([0, 1, 2])
    item_idx = torch.tensor([3, 4, 5])
    
    out = model(user_idx, item_idx)
    assert out.shape == (3,), "Output shape should match batch size"
    # explicit mode returns logits, no sigmoid bound checks needed

def test_ncf_model_forward_implicit():
    model = NCFModel(n_users=10, n_items=10, gmf_dim=8, mlp_layers=[16, 8], use_implicit=True)
    user_idx = torch.tensor([0, 1])
    item_idx = torch.tensor([3, 4])
    
    out = model(user_idx, item_idx)
    assert out.shape == (2,), "Output shape should match batch size"
    assert torch.all(out >= 0) and torch.all(out <= 1), "Implicit mode should return probabilities in [0, 1]"


def test_ncf_trainer_handles_singleton_training_batch():
    from app.ml.ncf_trainer import NCFTrainer

    ratings = pd.DataFrame(
        {
            "ml_user_id": ["user_1", "user_1"],
            "movie_id": [101, 102],
            "rating": [8.0, 6.0],
        }
    )
    preprocessor = NCFDataPreprocessor()
    train_dataset, val_dataset = preprocessor.fit_transform(ratings)
    model = NCFModel(
        n_users=preprocessor.n_users,
        n_items=preprocessor.n_items,
        gmf_dim=4,
        mlp_layers=[8, 4],
        dropout=0.0,
    )
    trainer = NCFTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        preprocessor=preprocessor,
        epochs=1,
        batch_size=8,
        patience=1,
    )

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    assert trainer.train() >= 0


def test_ncf_data_preprocessor_stratified_split():
    # 2 users, user1 has 10 ratings, user2 has 1 rating
    df = pd.DataFrame({
        "ml_user_id": ["u1"] * 10 + ["u2"],
        "movie_id": list(range(10)) + [99],
        "rating": [8.0] * 10 + [5.0]
    })
    
    pre = NCFDataPreprocessor()
    train_ds, val_ds = pre.fit_transform(df)
    
    # User 1 should have 90% (9) in train, 10% (1) in val
    # User 2 should have 100% (1) in train, 0 in val
    assert len(train_ds) == 10
    assert len(val_ds) == 1
    
    # Check that val_ds only contains user u1 (which maps to index 0)
    # user_to_idx is {'u1': 0, 'u2': 1}
    for i in range(len(val_ds)):
        u, item, label = val_ds[i]
        assert u.item() == pre.user_to_idx["u1"], "Validation should only contain users with >1 rating"

def test_ncf_adapter_cold_start():
    # Mock preprocessor and model
    pre = NCFDataPreprocessor()
    pre.user_to_idx = {"app_1": 0}
    pre.movie_to_idx = {100: 0, 101: 1}
    
    model = NCFModel(n_users=1, n_items=2)
    adapter = NCFCollaborativeAdapter(model, pre)
    
    # Cold start user (not in user_to_idx)
    recs = adapter.predict_top_n(user_id=999, candidate_movie_ids=[100, 101], n=10)
    assert recs == [], "Cold start user should return empty recommendations"
    
    # Known user, unknown movie
    recs = adapter.predict_top_n(user_id=1, candidate_movie_ids=[100, 999], n=10)
    assert len(recs) == 1, "Unknown movies should be skipped"
    assert recs[0][0] == 100, "Only known movies should be scored"

def test_movielens_loader_skips_invalid_tmdb_id(monkeypatch):
    from unittest.mock import MagicMock

    from app.core.config import Settings
    from app.services.data_pipeline.movielens_loader import MovieLensLoader

    settings = Settings(
        data_dir="data",
        movielens_variant="ml-latest-small",
        tmdb_api_key="test-key",
    )

    loader = MovieLensLoader(settings)

    movies_df = pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["Valid Movie", "Missing TMDB Movie"],
            "genres": ["Drama", "Comedy"],
        }
    )

    links_df = pd.DataFrame(
        {
            "movieId": [1, 2],
            "imdbId": ["1234567", np.nan],
            "tmdbId": ["123", np.nan],
        }
    )

    ratings_df = pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [1, 2],
            "rating": [4.0, 5.0],
            "timestamp": [100, 200],
        }
    )

    monkeypatch.setattr(loader, "_load_movies_csv", lambda: movies_df)
    monkeypatch.setattr(loader, "_load_links_csv", lambda: links_df)
    monkeypatch.setattr(loader, "_load_ratings_csv", lambda: ratings_df)
    monkeypatch.setattr(loader, "_sync_genres", lambda db, df: 0)
    monkeypatch.setattr(loader, "dataset_exists", lambda: True)

    db = MagicMock()

    db = MagicMock()

    def assign_movie_ids(movies):
        for index, movie in enumerate(movies, start=1):
            movie.id = index

    db.add_all.side_effect = assign_movie_ids

    query_mock = MagicMock()
    query_mock.filter.return_value.all.return_value = []
    db.query.return_value = query_mock

    result = loader.load(db)

    db.add_all.assert_called_once()

    inserted_movies = db.add_all.call_args.args[0]
    inserted_tmdb_ids = [movie.tmdb_id for movie in inserted_movies]

    assert inserted_tmdb_ids == [123]
    assert 0 not in inserted_tmdb_ids

    assert list(result.training_ratings_df["movie_id"]) == [1]