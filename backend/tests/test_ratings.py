"""
Tests for Milestone 6 — Ratings.
"""

from __future__ import annotations

from datetime import datetime
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.db.database import Base
from app.main import create_app
from app.models.movie import Movie
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
        "email": "rater@example.com",
        "password": "supersecret123",
        "display_name": "Rater"
    }
    client.post("/api/v1/auth/register", json=payload)
    resp = client.post("/api/v1/auth/login", json=payload)
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture()
def movie_id(client: TestClient) -> int:
    # We must access the DB directly to seed a movie
    from app.db.database import get_db
    generator = client.app.dependency_overrides[get_db]()
    db = next(generator)
    
    movie = Movie(
        tmdb_id=101,
        title="Test Movie",
    )
    db.add(movie)
    db.commit()
    db.refresh(movie)
    m_id = movie.id
    
    try:
        next(generator)
    except StopIteration:
        pass

    return m_id

class TestCreateRating:
    def test_create_rating_success(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        payload = {"movie_id": movie_id, "rating": 8}
        resp = client.post("/api/v1/ratings", json=payload, headers=auth_header)
        assert resp.status_code == 201
        body = resp.json()
        assert body["rating"] == 8
        assert body["movie_id"] == movie_id
        assert "id" in body

    def test_create_rating_unauthenticated(self, client: TestClient, movie_id: int):
        payload = {"movie_id": movie_id, "rating": 8}
        resp = client.post("/api/v1/ratings", json=payload)
        assert resp.status_code == 401

    def test_create_rating_duplicate_fails(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        payload = {"movie_id": movie_id, "rating": 8}
        client.post("/api/v1/ratings", json=payload, headers=auth_header)
        resp = client.post("/api/v1/ratings", json=payload, headers=auth_header)
        assert resp.status_code == 409

    def test_create_rating_invalid_movie(self, client: TestClient, auth_header: dict[str, str]):
        payload = {"movie_id": 9999, "rating": 8}
        resp = client.post("/api/v1/ratings", json=payload, headers=auth_header)
        assert resp.status_code == 404

    def test_create_rating_invalid_scale(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        resp = client.post("/api/v1/ratings", json={"movie_id": movie_id, "rating": 11}, headers=auth_header)
        assert resp.status_code == 422
        resp = client.post("/api/v1/ratings", json={"movie_id": movie_id, "rating": 0}, headers=auth_header)
        assert resp.status_code == 422

class TestGetRatings:
    def test_list_ratings(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        payload = {"movie_id": movie_id, "rating": 8}
        client.post("/api/v1/ratings", json=payload, headers=auth_header)
        
        resp = client.get("/api/v1/ratings")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert len(body["ratings"]) == 1
        assert body["ratings"][0]["rating"] == 8

    def test_list_ratings_filter_by_movie(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        payload = {"movie_id": movie_id, "rating": 8}
        client.post("/api/v1/ratings", json=payload, headers=auth_header)
        
        resp = client.get(f"/api/v1/ratings?movie_id={movie_id}")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1
        
        resp2 = client.get("/api/v1/ratings?movie_id=9999")
        assert resp2.status_code == 200
        assert resp2.json()["total"] == 0

class TestUpdateRating:
    def test_update_rating_success(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        payload = {"movie_id": movie_id, "rating": 5}
        create_resp = client.post("/api/v1/ratings", json=payload, headers=auth_header)
        rating_id = create_resp.json()["id"]

        update_resp = client.put(f"/api/v1/ratings/{rating_id}", json={"rating": 9}, headers=auth_header)
        assert update_resp.status_code == 200
        assert update_resp.json()["rating"] == 9

    def test_update_rating_not_found(self, client: TestClient, auth_header: dict[str, str]):
        update_resp = client.put("/api/v1/ratings/9999", json={"rating": 9}, headers=auth_header)
        assert update_resp.status_code == 404

    def test_update_rating_wrong_user(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        payload = {"movie_id": movie_id, "rating": 5}
        create_resp = client.post("/api/v1/ratings", json=payload, headers=auth_header)
        rating_id = create_resp.json()["id"]

        # create second user
        payload2 = {"email": "other@example.com", "password": "supersecret123", "display_name": "Other"}
        client.post("/api/v1/auth/register", json=payload2)
        login_resp = client.post("/api/v1/auth/login", json=payload2)
        token = login_resp.json()["access_token"]
        auth_header2 = {"Authorization": f"Bearer {token}"}

        update_resp = client.put(f"/api/v1/ratings/{rating_id}", json={"rating": 9}, headers=auth_header2)
        assert update_resp.status_code == 403

class TestDeleteRating:
    def test_delete_rating_success(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        payload = {"movie_id": movie_id, "rating": 5}
        create_resp = client.post("/api/v1/ratings", json=payload, headers=auth_header)
        rating_id = create_resp.json()["id"]

        delete_resp = client.delete(f"/api/v1/ratings/{rating_id}", headers=auth_header)
        assert delete_resp.status_code == 204

        get_resp = client.get("/api/v1/ratings")
        assert get_resp.json()["total"] == 0

    def test_delete_rating_wrong_user(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        payload = {"movie_id": movie_id, "rating": 5}
        create_resp = client.post("/api/v1/ratings", json=payload, headers=auth_header)
        rating_id = create_resp.json()["id"]

        # create second user
        payload2 = {"email": "other2@example.com", "password": "supersecret123", "display_name": "Other2"}
        client.post("/api/v1/auth/register", json=payload2)
        login_resp = client.post("/api/v1/auth/login", json=payload2)
        token = login_resp.json()["access_token"]
        auth_header2 = {"Authorization": f"Bearer {token}"}

        delete_resp = client.delete(f"/api/v1/ratings/{rating_id}", headers=auth_header2)
        assert delete_resp.status_code == 403
