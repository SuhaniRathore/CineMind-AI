from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.db.database import Base
from app.main import create_app
from app.models.genre import Genre
from app.models.movie import Movie
from app.models.rating import Rating
from app.models.recommendation import RecommendationResult
from app.models.user import User

# Test secrets
_TEST_SECRET = "test-jwt-secret-do-not-use-in-production-ever"
_TEST_ALGORITHM = "HS256"


def _test_settings() -> Settings:
    return Settings(
        jwt_secret_key=_TEST_SECRET,
        jwt_algorithm=_TEST_ALGORITHM,
        access_token_expire_minutes=30,
        database_url="sqlite:///:memory:",
        tmdb_api_key="test-fake-key",
    )


def _make_sqlite_client() -> TestClient:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    settings = _test_settings()
    app = create_app()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    from app.db.database import get_db
    from app.core.config import get_settings

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = lambda: settings

    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def client() -> TestClient:
    return _make_sqlite_client()


@pytest.fixture()
def auth_header(client: TestClient) -> dict[str, str]:
    payload = {
        "email": "recuser@example.com",
        "password": "supersecret123",
        "display_name": "RecUser",
    }
    client.post("/api/v1/auth/register", json=payload)
    resp = client.post("/api/v1/auth/login", json=payload)
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def seed_movies(client: TestClient) -> dict[str, int]:
    """
    Seed movies with specific genres to test content similarity.
    Action: Movie A, Movie B
    Comedy: Movie C
    Action/Comedy: Movie D
    """
    from app.db.database import get_db
    generator = client.app.dependency_overrides[get_db]()
    db = next(generator)

    genre_action = Genre(tmdb_genre_id=28, name="Action")
    genre_comedy = Genre(tmdb_genre_id=35, name="Comedy")
    db.add_all([genre_action, genre_comedy])
    db.commit()

    movie_a = Movie(tmdb_id=201, title="Action Movie A", original_language="en")
    movie_a.genres.append(genre_action)

    movie_b = Movie(tmdb_id=202, title="Action Movie B", original_language="en")
    movie_b.genres.append(genre_action)

    movie_c = Movie(tmdb_id=203, title="Comedy Movie C", original_language="en")
    movie_c.genres.append(genre_comedy)

    movie_d = Movie(tmdb_id=204, title="Action Comedy D", original_language="en")
    movie_d.genres.append(genre_action)
    movie_d.genres.append(genre_comedy)

    db.add_all([movie_a, movie_b, movie_c, movie_d])
    db.commit()
    db.refresh(movie_a)
    db.refresh(movie_b)
    db.refresh(movie_c)
    db.refresh(movie_d)

    m_ids = {
        "A": movie_a.id,
        "B": movie_b.id,
        "C": movie_c.id,
        "D": movie_d.id,
    }

    try:
        next(generator)
    except StopIteration:
        pass
    return m_ids


class TestRecommendations:
    def test_unauthorized(self, client: TestClient):
        resp = client.get("/api/v1/recommendations")
        assert resp.status_code == 401

    def test_empty_state_no_ratings(self, client: TestClient, auth_header: dict[str, str], seed_movies):
        # User has no ratings
        resp = client.get("/api/v1/recommendations", headers=auth_header)
        assert resp.status_code == 200
        assert resp.json()["total"] == 0
        assert len(resp.json()["recommendations"]) == 0

    def test_insufficient_movie_data(self, client: TestClient, auth_header: dict[str, str]):
        # No movies in DB at all, but user exists.
        # Let's mock the user rating a movie, but since we didn't seed_movies, there are none.
        resp = client.get("/api/v1/recommendations", headers=auth_header)
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_content_similarity_ranking(self, client: TestClient, auth_header: dict[str, str], seed_movies):
        # Rate Action Movie A highly (9/10)
        client.post(
            "/api/v1/ratings",
            json={"movie_id": seed_movies["A"], "rating": 9},
            headers=auth_header,
        )

        resp = client.get("/api/v1/recommendations", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        
        # Movie A is already rated, should be excluded.
        rec_movie_ids = [r["movie_id"] for r in data["recommendations"]]
        assert seed_movies["A"] not in rec_movie_ids

        # Should recommend B and D (they have Action). C (pure Comedy) should be lowest or excluded if 0 similarity.
        # B is pure Action, D is Action/Comedy. B should be higher or equal to D in similarity to A (pure Action).
        assert seed_movies["B"] in rec_movie_ids
        
        # Verify persistence
        from app.db.database import get_db
        generator = client.app.dependency_overrides[get_db]()
        db = next(generator)
        user = db.query(User).filter(User.email == "recuser@example.com").first()
        persisted = db.query(RecommendationResult).filter(RecommendationResult.user_id == user.id).all()
        assert len(persisted) == len(data["recommendations"])

        try:
            next(generator)
        except StopIteration:
            pass

    def test_low_rating_ignored(self, client: TestClient, auth_header: dict[str, str], seed_movies):
        # Rate Movie A with a low rating (4/10)
        client.post(
            "/api/v1/ratings",
            json={"movie_id": seed_movies["A"], "rating": 4},
            headers=auth_header,
        )

        resp = client.get("/api/v1/recommendations", headers=auth_header)
        assert resp.status_code == 200
        # Should return 0 since there are no HIGH ratings (>=7)
        assert resp.json()["total"] == 0
