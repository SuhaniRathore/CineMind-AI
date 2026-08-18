"""Tests for the health-check endpoint.

Updated in Milestone 2 to assert the new db_status field.
"""

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_name"] == "CineMind AI"
    assert "environment" in body
    # db_status is always present; its value depends on whether PostgreSQL
    # is running in the test environment — both outcomes are acceptable here.
    assert body["db_status"] in ("connected", "unavailable")
