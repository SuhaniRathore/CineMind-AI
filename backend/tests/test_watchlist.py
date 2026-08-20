from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.db.database import Base
from app.main import create_app
from app.models.movie import Movie
from app.models.watchlist import WatchlistItem

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
        "email": "watchlistuser@example.com",
        "password": "supersecret123",
        "display_name": "WatchlistUser"
    }
    client.post("/api/v1/auth/register", json=payload)
    resp = client.post("/api/v1/auth/login", json=payload)
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture()
def movie_id(client: TestClient) -> int:
    from app.db.database import get_db
    generator = client.app.dependency_overrides[get_db]()
    db = next(generator)
    
    movie = Movie(tmdb_id=103, title="Test Movie for Watchlist")
    db.add(movie)
    db.commit()
    db.refresh(movie)
    m_id = movie.id
    
    try:
        next(generator)
    except StopIteration:
        pass
    return m_id

class TestWatchlist:
    def test_add_to_watchlist(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        resp = client.post("/api/v1/watchlist", json={"movie_id": movie_id}, headers=auth_header)
        assert resp.status_code == 201
        assert resp.json()["movie_id"] == movie_id

    def test_add_duplicate(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        client.post("/api/v1/watchlist", json={"movie_id": movie_id}, headers=auth_header)
        resp = client.post("/api/v1/watchlist", json={"movie_id": movie_id}, headers=auth_header)
        assert resp.status_code == 409

    def test_list_watchlist(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        client.post("/api/v1/watchlist", json={"movie_id": movie_id}, headers=auth_header)
        resp = client.get("/api/v1/watchlist", headers=auth_header)
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_remove_watchlist(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        create_resp = client.post("/api/v1/watchlist", json={"movie_id": movie_id}, headers=auth_header)
        item_id = create_resp.json()["id"]

        del_resp = client.delete(f"/api/v1/watchlist/{item_id}", headers=auth_header)
        assert del_resp.status_code == 204

        resp = client.get("/api/v1/watchlist", headers=auth_header)
        assert resp.json()["total"] == 0

    def test_remove_unauthorized(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        create_resp = client.post("/api/v1/watchlist", json={"movie_id": movie_id}, headers=auth_header)
        item_id = create_resp.json()["id"]

        # create second user
        payload2 = {"email": "other@example.com", "password": "supersecret123", "display_name": "Other"}
        client.post("/api/v1/auth/register", json=payload2)
        login_resp = client.post("/api/v1/auth/login", json=payload2)
        token = login_resp.json()["access_token"]
        auth_header2 = {"Authorization": f"Bearer {token}"}

        del_resp = client.delete(f"/api/v1/watchlist/{item_id}", headers=auth_header2)
        assert del_resp.status_code == 403
