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
from app.models.review import Review

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
        "email": "reviewer@example.com",
        "password": "supersecret123",
        "display_name": "Reviewer"
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
    
    movie = Movie(tmdb_id=102, title="Test Movie for Reviews")
    db.add(movie)
    db.commit()
    db.refresh(movie)
    m_id = movie.id
    
    try:
        next(generator)
    except StopIteration:
        pass
    return m_id

class TestCreateReview:
    def test_create_review_success(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        payload = {"movie_id": movie_id, "title": "A great movie", "body": "I really enjoyed watching this."}
        resp = client.post("/api/v1/reviews", json=payload, headers=auth_header)
        assert resp.status_code == 201
        body = resp.json()
        assert body["title"] == "A great movie"
        assert body["movie_id"] == movie_id

    def test_create_review_duplicate(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        payload = {"movie_id": movie_id, "title": "First Review", "body": "First body is quite long."}
        client.post("/api/v1/reviews", json=payload, headers=auth_header)
        resp = client.post("/api/v1/reviews", json=payload, headers=auth_header)
        assert resp.status_code == 409

class TestListReviews:
    def test_list_reviews(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        payload = {"movie_id": movie_id, "title": "Review 1", "body": "Body 1 is long enough."}
        client.post("/api/v1/reviews", json=payload, headers=auth_header)
        
        resp = client.get("/api/v1/reviews")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["reviews"][0]["title"] == "Review 1"

class TestUpdateReview:
    def test_update_review(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        payload = {"movie_id": movie_id, "title": "Old Title", "body": "Old Body is long enough."}
        create_resp = client.post("/api/v1/reviews", json=payload, headers=auth_header)
        review_id = create_resp.json()["id"]

        update_resp = client.put(
            f"/api/v1/reviews/{review_id}",
            json={"title": "New Title", "body": "New Body is long enough."},
            headers=auth_header
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["title"] == "New Title"

    def test_update_unauthorized(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        payload = {"movie_id": movie_id, "title": "Old Title", "body": "Old Body is long enough."}
        create_resp = client.post("/api/v1/reviews", json=payload, headers=auth_header)
        review_id = create_resp.json()["id"]

        # create second user
        payload2 = {"email": "other@example.com", "password": "supersecret123", "display_name": "Other"}
        client.post("/api/v1/auth/register", json=payload2)
        login_resp = client.post("/api/v1/auth/login", json=payload2)
        token = login_resp.json()["access_token"]
        auth_header2 = {"Authorization": f"Bearer {token}"}

        update_resp = client.put(
            f"/api/v1/reviews/{review_id}",
            json={"title": "Hacked Title", "body": "Hacked Body is long enough."},
            headers=auth_header2
        )
        assert update_resp.status_code == 403

class TestDeleteReview:
    def test_delete_review(self, client: TestClient, auth_header: dict[str, str], movie_id: int):
        payload = {"movie_id": movie_id, "title": "To delete", "body": "To delete this one."}
        create_resp = client.post("/api/v1/reviews", json=payload, headers=auth_header)
        review_id = create_resp.json()["id"]

        delete_resp = client.delete(f"/api/v1/reviews/{review_id}", headers=auth_header)
        assert delete_resp.status_code == 204

        get_resp = client.get("/api/v1/reviews")
        assert get_resp.json()["total"] == 0
