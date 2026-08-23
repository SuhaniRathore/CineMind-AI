import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.limiter import limiter

@pytest.fixture(autouse=True)
def enable_limiter():
    limiter.enabled = True
    yield
    limiter.enabled = False

client = TestClient(app)

def test_global_rate_limit_health_check_exempt():
    # Health should not be rate limited
    headers = {"X-Forwarded-For": "10.0.0.1"}
    for _ in range(65):
        resp = client.get("/health", headers=headers)
        assert resp.status_code == 200

def test_auth_login_rate_limit():
    # /api/v1/auth/login is limited to 5/minute
    payload = {"email": "test@example.com", "password": "password123"}
    headers = {"X-Forwarded-For": "10.0.0.2"}
    for i in range(5):
        resp = client.post("/api/v1/auth/login", json=payload, headers=headers)
        # We don't care if it's 401, just that it's not 429 yet
        assert resp.status_code != 429
    
    # 6th request should be rate limited
    resp = client.post("/api/v1/auth/login", json=payload, headers=headers)
    assert resp.status_code == 429

def test_movies_global_rate_limit():
    # /api/v1/movies has a default limit of 60/minute
    headers = {"X-Forwarded-For": "10.0.0.3"}
    for i in range(60):
        resp = client.get("/api/v1/movies", headers=headers)
        assert resp.status_code != 429

    # 61st request should be rate limited
    resp = client.get("/api/v1/movies", headers=headers)
    assert resp.status_code == 429
