"""
Tests for Milestone 5 — Authentication.

Test strategy:
  - All tests use FastAPI TestClient with an in-memory SQLite database
    (same pattern as test_movies.py) — no live PostgreSQL required.
  - JWT tokens are created with a deterministic test secret so tests are
    hermetic and never touch the real .env JWT_SECRET_KEY.
  - No real bcrypt hashing is mocked — bcrypt is fast enough for test
    passwords (work-factor defaults apply, but test passwords are short
    and bcrypt still completes in milliseconds for test suites).

Test classes:
  TestRegister      — POST /api/v1/auth/register
  TestLogin         — POST /api/v1/auth/login
  TestGetMe         — GET  /api/v1/users/me  (protected route example)
  TestSecurity      — unit tests for security.py primitives
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.db.database import Base, get_db
from app.main import create_app


# ---------------------------------------------------------------------------
# Shared test Settings — deterministic secrets, never reads real .env
# ---------------------------------------------------------------------------

_TEST_SECRET = "test-jwt-secret-do-not-use-in-production-ever"
_TEST_ALGORITHM = "HS256"
_TEST_EXPIRE_MINUTES = 30


def _test_settings() -> Settings:
    """Return a Settings instance with fixed test values (no .env loading)."""
    return Settings(
        jwt_secret_key=_TEST_SECRET,
        jwt_algorithm=_TEST_ALGORITHM,
        access_token_expire_minutes=_TEST_EXPIRE_MINUTES,
        database_url="sqlite:///:memory:",  # unused — overridden by override_get_db
        tmdb_api_key="test-fake-key",
    )


# ---------------------------------------------------------------------------
# SQLite in-memory database fixture (per-test isolation)
# ---------------------------------------------------------------------------

def _make_sqlite_client() -> TestClient:
    """
    Build a TestClient backed by a fresh in-memory SQLite database.

    Each call returns an independent client so tests do not share state.
    """
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
    """Per-test isolated TestClient with fresh in-memory SQLite DB."""
    return _make_sqlite_client()


# ---------------------------------------------------------------------------
# Helper: register a user and return the response JSON
# ---------------------------------------------------------------------------

_VALID_PAYLOAD = {
    "email": "alice@example.com",
    "password": "supersecret123",
    "display_name": "Alice",
}


def _register(client: TestClient, payload: dict | None = None) -> dict:
    payload = payload or _VALID_PAYLOAD
    resp = client.post("/api/v1/auth/register", json=payload)
    return resp


# ===========================================================================
# TestSecurity — unit tests for app.core.security primitives
# ===========================================================================


class TestSecurity:
    """Pure-unit tests for security.py — no HTTP, no database."""

    def test_hash_password_returns_string(self):
        h = hash_password("mypassword")
        assert isinstance(h, str)
        assert len(h) > 0

    def test_hash_password_is_not_plaintext(self):
        h = hash_password("mypassword")
        assert "mypassword" not in h

    def test_verify_password_correct(self):
        h = hash_password("correct-horse-battery-staple")
        assert verify_password("correct-horse-battery-staple", h) is True

    def test_verify_password_wrong(self):
        h = hash_password("correct-horse-battery-staple")
        assert verify_password("wrong-password", h) is False

    def test_two_hashes_of_same_password_differ(self):
        """bcrypt uses a random salt — same input must never produce the same hash."""
        h1 = hash_password("same-password")
        h2 = hash_password("same-password")
        assert h1 != h2

    def test_create_access_token_returns_string(self):
        settings = _test_settings()
        token = create_access_token(42, settings)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_decode_access_token_roundtrip(self):
        settings = _test_settings()
        token = create_access_token(99, settings)
        user_id = decode_access_token(token, settings)
        assert user_id == 99

    def test_decode_access_token_wrong_secret_raises(self):
        settings = _test_settings()
        token = create_access_token(1, settings)

        bad_settings = Settings(
            jwt_secret_key="totally-wrong-secret",
            jwt_algorithm=_TEST_ALGORITHM,
            database_url="sqlite:///:memory:",
            tmdb_api_key="x",
        )
        with pytest.raises(jwt.PyJWTError):
            decode_access_token(token, bad_settings)

    def test_decode_expired_token_raises(self):
        """A token with exp in the past must be rejected."""
        settings = _test_settings()
        payload = {
            "sub": "7",
            "iat": datetime.now(tz=timezone.utc) - timedelta(hours=2),
            "exp": datetime.now(tz=timezone.utc) - timedelta(hours=1),
        }
        expired_token = jwt.encode(payload, _TEST_SECRET, algorithm=_TEST_ALGORITHM)
        with pytest.raises(jwt.ExpiredSignatureError):
            decode_access_token(expired_token, settings)

    def test_decode_tampered_token_raises(self):
        settings = _test_settings()
        token = create_access_token(1, settings)
        # Flip the first character in the signature segment to guarantee a data bit change
        parts = token.split(".")
        parts[2] = ("A" if parts[2][0] != "A" else "B") + parts[2][1:]
        tampered = ".".join(parts)
        with pytest.raises(jwt.PyJWTError):
            decode_access_token(tampered, settings)


# ===========================================================================
# TestRegister — POST /api/v1/auth/register
# ===========================================================================


class TestRegister:
    def test_register_success_returns_201(self, client: TestClient):
        resp = _register(client)
        assert resp.status_code == 201

    def test_register_returns_user_fields(self, client: TestClient):
        resp = _register(client)
        body = resp.json()
        assert body["email"] == _VALID_PAYLOAD["email"]
        assert body["display_name"] == _VALID_PAYLOAD["display_name"]
        assert body["is_active"] is True
        assert "id" in body
        assert "created_at" in body

    def test_register_does_not_return_password(self, client: TestClient):
        resp = _register(client)
        body = resp.json()
        assert "password" not in body
        assert "hashed_password" not in body

    def test_register_duplicate_email_returns_409(self, client: TestClient):
        _register(client)  # first registration
        resp = _register(client)  # same email again
        assert resp.status_code == 409

    def test_register_invalid_email_returns_422(self, client: TestClient):
        payload = {**_VALID_PAYLOAD, "email": "not-an-email"}
        resp = _register(client, payload)
        assert resp.status_code == 422

    def test_register_short_password_returns_422(self, client: TestClient):
        """Pydantic min_length=8 on password triggers 422 before hitting service."""
        payload = {**_VALID_PAYLOAD, "password": "short"}
        resp = _register(client, payload)
        assert resp.status_code == 422

    def test_register_missing_display_name_returns_422(self, client: TestClient):
        payload = {"email": "bob@example.com", "password": "longpassword"}
        resp = _register(client, payload)
        assert resp.status_code == 422

    def test_register_email_is_normalised(self, client: TestClient):
        """email should be lowercased and stripped."""
        payload = {**_VALID_PAYLOAD, "email": "  Alice@EXAMPLE.COM  "}
        resp = _register(client, payload)
        # Pydantic EmailStr normalises to lowercase
        assert resp.status_code == 201
        assert "@example.com" in resp.json()["email"]


# ===========================================================================
# TestLogin — POST /api/v1/auth/login
# ===========================================================================


class TestLogin:
    def _login(self, client: TestClient, email: str, password: str) -> dict:
        return client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": password},
        )

    def test_login_success_returns_200(self, client: TestClient):
        _register(client)
        resp = self._login(client, _VALID_PAYLOAD["email"], _VALID_PAYLOAD["password"])
        assert resp.status_code == 200

    def test_login_returns_access_token(self, client: TestClient):
        _register(client)
        resp = self._login(client, _VALID_PAYLOAD["email"], _VALID_PAYLOAD["password"])
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    def test_login_token_is_valid_jwt(self, client: TestClient):
        """The returned token must decode correctly with the test secret."""
        _register(client)
        resp = self._login(client, _VALID_PAYLOAD["email"], _VALID_PAYLOAD["password"])
        token = resp.json()["access_token"]
        payload = jwt.decode(token, _TEST_SECRET, algorithms=[_TEST_ALGORITHM])
        assert "sub" in payload
        assert int(payload["sub"]) > 0

    def test_login_returns_user_in_body(self, client: TestClient):
        _register(client)
        resp = self._login(client, _VALID_PAYLOAD["email"], _VALID_PAYLOAD["password"])
        body = resp.json()
        assert "user" in body
        assert body["user"]["email"] == _VALID_PAYLOAD["email"]

    def test_login_wrong_password_returns_401(self, client: TestClient):
        _register(client)
        resp = self._login(client, _VALID_PAYLOAD["email"], "wrong-password-xyz")
        assert resp.status_code == 401

    def test_login_unknown_email_returns_401(self, client: TestClient):
        resp = self._login(client, "nobody@example.com", "doesntmatter123")
        assert resp.status_code == 401

    def test_login_does_not_return_password(self, client: TestClient):
        _register(client)
        resp = self._login(client, _VALID_PAYLOAD["email"], _VALID_PAYLOAD["password"])
        body = resp.json()
        assert "password" not in body
        assert "hashed_password" not in str(body)

    def test_login_missing_fields_returns_422(self, client: TestClient):
        """Verify the dedicated login schema rejects missing email or password."""
        # Missing password
        resp1 = client.post("/api/v1/auth/login", json={"email": "alice@example.com"})
        assert resp1.status_code == 422
        
        # Missing email
        resp2 = client.post("/api/v1/auth/login", json={"password": "password123"})
        assert resp2.status_code == 422

    def test_login_missing_credentials(self, client: TestClient):
        resp = client.post("/api/v1/auth/login", json={})
        assert resp.status_code == 422


# ===========================================================================
# TestGetMe — GET /api/v1/users/me  (protected route)
# ===========================================================================


class TestGetMe:
    def _auth_header(self, client: TestClient) -> dict:
        """Register, login, and return a valid Authorization header."""
        _register(client)
        resp = client.post(
            "/api/v1/auth/login",
            json={"email": _VALID_PAYLOAD["email"], "password": _VALID_PAYLOAD["password"]},
        )
        token = resp.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_get_me_with_valid_token_returns_200(self, client: TestClient):
        headers = self._auth_header(client)
        resp = client.get("/api/v1/users/me", headers=headers)
        assert resp.status_code == 200

    def test_get_me_returns_correct_user(self, client: TestClient):
        headers = self._auth_header(client)
        resp = client.get("/api/v1/users/me", headers=headers)
        body = resp.json()
        assert body["email"] == _VALID_PAYLOAD["email"]
        assert body["display_name"] == _VALID_PAYLOAD["display_name"]

    def test_get_me_no_token_returns_401(self, client: TestClient):
        resp = client.get("/api/v1/users/me")
        assert resp.status_code == 401

    def test_get_me_invalid_token_returns_401(self, client: TestClient):
        resp = client.get(
            "/api/v1/users/me",
            headers={"Authorization": "Bearer this.is.not.a.valid.token"},
        )
        assert resp.status_code == 401

    def test_get_me_expired_token_returns_401(self, client: TestClient):
        """A token signed with the test secret but already expired must be rejected."""
        settings = _test_settings()
        payload = {
            "sub": "999",
            "iat": datetime.now(tz=timezone.utc) - timedelta(hours=2),
            "exp": datetime.now(tz=timezone.utc) - timedelta(hours=1),
        }
        expired_token = jwt.encode(payload, _TEST_SECRET, algorithm=_TEST_ALGORITHM)
        resp = client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert resp.status_code == 401

    def test_get_me_wrong_secret_returns_401(self, client: TestClient):
        """Token signed with a different secret must be rejected."""
        payload = {
            "sub": "1",
            "iat": datetime.now(tz=timezone.utc),
            "exp": datetime.now(tz=timezone.utc) + timedelta(hours=1),
        }
        bad_token = jwt.encode(payload, "wrong-secret", algorithm=_TEST_ALGORITHM)
        resp = client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {bad_token}"},
        )
        assert resp.status_code == 401

    def test_get_me_does_not_return_password(self, client: TestClient):
        headers = self._auth_header(client)
        resp = client.get("/api/v1/users/me", headers=headers)
        body = resp.json()
        assert "password" not in body
        assert "hashed_password" not in body

    def test_get_me_malformed_auth_header(self, client: TestClient):
        resp = client.get(
            "/api/v1/users/me",
            headers={"Authorization": "Basic not.a.jwt"},
        )
        assert resp.status_code == 401
