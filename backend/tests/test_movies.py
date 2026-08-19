"""
Tests for Milestone 4 — TMDB integration and movie endpoints.

Test strategy:
  - TMDBClient tests use httpx mocking (pytest-mock / unittest.mock) to
    verify parsing logic without making real outbound HTTP calls.
    No real TMDB API key is ever used or exposed in tests.
  - Movie endpoint tests use FastAPI TestClient with an in-memory SQLite
    database (same approach as Milestone 3 model tests) so no live
    PostgreSQL is required.
  - Sync endpoint tests mock the TMDBClient at the service layer.

Running all tests (including pre-existing ones) must still pass after
Milestone 4 changes — the health, db, and model tests are not modified.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.db.database import Base, get_db
from app.main import create_app
from app.models.genre import Genre
from app.models.movie import Movie
from app.services.tmdb_client import TMDBClient, TMDBError


# ---------------------------------------------------------------------------
# Shared TMDB fixture data
# ---------------------------------------------------------------------------

FAKE_GENRES = [
    {"id": 28, "name": "Action"},
    {"id": 12, "name": "Adventure"},
    {"id": 16, "name": "Animation"},
]

FAKE_MOVIE_ITEM = {
    "id": 550,
    "title": "Fight Club",
    "overview": "An insomniac office worker forms an underground fight club.",
    "release_date": "1999-10-15",
    "runtime": None,  # not in list responses
    "poster_path": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
    "original_language": "en",
    "popularity": 61.416,
    "vote_average": 8.433,
    "vote_count": 26280,
    "genre_ids": [18, 53],
}

FAKE_MOVIE_DETAIL = {
    "id": 550,
    "title": "Fight Club",
    "overview": "An insomniac office worker forms an underground fight club.",
    "release_date": "1999-10-15",
    "runtime": 139,
    "poster_path": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
    "original_language": "en",
    "popularity": 61.416,
    "vote_average": 8.433,
    "vote_count": 26280,
    "genres": [{"id": 18, "name": "Drama"}, {"id": 53, "name": "Thriller"}],
}

FAKE_POPULAR_PAGE = {
    "page": 1,
    "results": [FAKE_MOVIE_ITEM],
    "total_pages": 500,
    "total_results": 10000,
}

FAKE_SEARCH_RESULTS = {
    "page": 1,
    "results": [FAKE_MOVIE_ITEM],
    "total_pages": 1,
    "total_results": 1,
}


# ---------------------------------------------------------------------------
# Minimal Settings for testing (avoids loading the real .env)
# ---------------------------------------------------------------------------

def _test_settings(api_key: str = "test-fake-key") -> Settings:
    """Return a Settings instance with a fake API key for unit tests."""
    return Settings(
        tmdb_api_key=api_key,
        tmdb_base_url="https://api.themoviedb.org/3",
        database_url="postgresql+psycopg://postgres:postgres@localhost:5432/cinemind",
    )


# ===========================================================================
# TMDBClient unit tests (no real HTTP calls)
# ===========================================================================


class TestTMDBClientGetGenres:
    """TMDBClient.get_genres() parses the TMDB genre list response."""

    def test_returns_list_of_genre_dicts(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"genres": FAKE_GENRES}
        mock_response.raise_for_status = MagicMock()

        mock_http = MagicMock(spec=httpx.Client)
        mock_http.get.return_value = mock_response

        client = TMDBClient(settings=_test_settings(), http_client=mock_http)
        genres = client.get_genres()

        assert isinstance(genres, list)
        assert len(genres) == 3
        assert genres[0] == {"id": 28, "name": "Action"}

    def test_returns_empty_list_when_key_missing(self):
        """If TMDB returns no genres key, return an empty list gracefully."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        mock_http = MagicMock(spec=httpx.Client)
        mock_http.get.return_value = mock_response

        client = TMDBClient(settings=_test_settings(), http_client=mock_http)
        assert client.get_genres() == []

    def test_api_key_not_in_request_url(self):
        """The API key must be sent as a query param, not visible in the path."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"genres": FAKE_GENRES}
        mock_response.raise_for_status = MagicMock()

        mock_http = MagicMock(spec=httpx.Client)
        mock_http.get.return_value = mock_response

        client = TMDBClient(settings=_test_settings(api_key="super-secret"), http_client=mock_http)
        client.get_genres()

        call_kwargs = mock_http.get.call_args
        # URL (first positional arg) must not contain the API key
        called_url: str = call_kwargs[0][0]
        assert "super-secret" not in called_url, (
            "API key must not appear in the URL path — it should be in params only"
        )


class TestTMDBClientSearchMovies:
    """TMDBClient.search_movies() parses TMDB search results."""

    def test_returns_search_results_dict(self):
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SEARCH_RESULTS
        mock_response.raise_for_status = MagicMock()

        mock_http = MagicMock(spec=httpx.Client)
        mock_http.get.return_value = mock_response

        client = TMDBClient(settings=_test_settings(), http_client=mock_http)
        result = client.search_movies("Fight Club")

        assert "results" in result
        assert result["results"][0]["id"] == 550


class TestTMDBClientGetMovieDetails:
    """TMDBClient.get_movie_details() returns the full detail dict."""

    def test_returns_detail_with_genres(self):
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_MOVIE_DETAIL
        mock_response.raise_for_status = MagicMock()

        mock_http = MagicMock(spec=httpx.Client)
        mock_http.get.return_value = mock_response

        client = TMDBClient(settings=_test_settings(), http_client=mock_http)
        detail = client.get_movie_details(550)

        assert detail["id"] == 550
        assert detail["runtime"] == 139
        assert len(detail["genres"]) == 2


class TestTMDBClientErrorHandling:
    """TMDBClient wraps HTTP / network errors in TMDBError."""

    def test_raises_tmdb_error_on_4xx(self):
        mock_http = MagicMock(spec=httpx.Client)
        # Simulate a 404 response
        mock_request = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        http_exc = httpx.HTTPStatusError("404", request=mock_request, response=mock_resp)
        mock_http.get.side_effect = http_exc

        client = TMDBClient(settings=_test_settings(), http_client=mock_http)
        with pytest.raises(TMDBError):
            client.get_movie_details(999999)

    def test_raises_tmdb_error_on_network_failure(self):
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.get.side_effect = httpx.ConnectError("Connection refused")

        client = TMDBClient(settings=_test_settings(), http_client=mock_http)
        with pytest.raises(TMDBError):
            client.get_genres()

    def test_error_message_does_not_contain_api_key(self):
        """TMDBError messages must never include the secret API key."""
        mock_http = MagicMock(spec=httpx.Client)
        mock_request = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        http_exc = httpx.HTTPStatusError("401", request=mock_request, response=mock_resp)
        mock_http.get.side_effect = http_exc

        client = TMDBClient(settings=_test_settings(api_key="my-secret-key"), http_client=mock_http)
        with pytest.raises(TMDBError) as exc_info:
            client.get_genres()

        assert "my-secret-key" not in str(exc_info.value), (
            "API key must not appear in TMDBError message — it could end up in logs"
        )


# ===========================================================================
# Movie endpoint integration tests (SQLite in-memory, no live PostgreSQL)
# ===========================================================================


@pytest.fixture()
def sqlite_engine():
    """
    Single in-memory SQLite engine with a StaticPool.

    SQLite in-memory databases are per-connection: each new connection gets
    an empty database.  StaticPool forces SQLAlchemy to reuse a single
    underlying connection across all sessions — the same connection that
    create_all() used — so all sessions see the same tables and data.
    
    This is the standard pattern for SQLite in-memory testing with SQLAlchemy.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def sqlite_session_factory(sqlite_engine):
    """Session factory bound to the shared SQLite engine."""
    factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=sqlite_engine,
    )
    return factory


@pytest.fixture()
def client_with_db(sqlite_session_factory):
    """
    Return a TestClient whose DB dependency is overridden with SQLite.

    The TMDBClient dependency is NOT overridden here — sync endpoint tests
    do that separately per test.
    """
    def override_get_db():
        db = sqlite_session_factory()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app), sqlite_session_factory
    app.dependency_overrides.clear()


class TestListMoviesEndpoint:
    """GET /api/v1/movies — list endpoint."""

    def test_returns_200_and_empty_list_when_no_movies(self, client_with_db):
        client, _ = client_with_db
        response = client.get("/api/v1/movies")
        assert response.status_code == 200
        body = response.json()
        assert body["movies"] == []
        assert body["total"] == 0
        assert "limit" in body
        assert "offset" in body

    def test_returns_seeded_movies(self, client_with_db):
        client, session_factory = client_with_db
        # Seed a movie directly into SQLite
        db = session_factory()
        genre = Genre(tmdb_genre_id=28, name="Action")
        movie = Movie(
            tmdb_id=550,
            title="Fight Club",
            popularity=61.4,
            vote_average=8.4,
            vote_count=26000,
        )
        db.add_all([genre, movie])
        db.flush()
        movie.genres.append(genre)
        db.commit()
        db.close()

        response = client.get("/api/v1/movies")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["movies"][0]["title"] == "Fight Club"
        assert body["movies"][0]["tmdb_id"] == 550
        assert len(body["movies"][0]["genres"]) == 1

    def test_pagination_limit_offset(self, client_with_db):
        client, session_factory = client_with_db
        db = session_factory()
        for i in range(5):
            db.add(Movie(tmdb_id=100 + i, title=f"Movie {i}", popularity=float(i)))
        db.commit()
        db.close()

        response = client.get("/api/v1/movies?limit=2&offset=0")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 5
        assert len(body["movies"]) == 2
        assert body["limit"] == 2
        assert body["offset"] == 0

    def test_invalid_limit_returns_422(self, client_with_db):
        client, _ = client_with_db
        response = client.get("/api/v1/movies?limit=0")
        assert response.status_code == 422

    def test_poster_url_is_computed(self, client_with_db):
        client, session_factory = client_with_db
        db = session_factory()
        db.add(Movie(
            tmdb_id=999,
            title="Poster Movie",
            poster_path="/abc.jpg",
        ))
        db.commit()
        db.close()

        response = client.get("/api/v1/movies")
        assert response.status_code == 200
        movie_data = response.json()["movies"][0]
        assert movie_data["poster_url"] == "https://image.tmdb.org/t/p/w500/abc.jpg"

    def test_no_poster_returns_null_url(self, client_with_db):
        client, session_factory = client_with_db
        db = session_factory()
        db.add(Movie(tmdb_id=888, title="No Poster Movie", poster_path=None))
        db.commit()
        db.close()

        response = client.get("/api/v1/movies")
        movie_data = response.json()["movies"][0]
        assert movie_data["poster_url"] is None


class TestSearchMoviesEndpoint:
    """GET /api/v1/movies/search — keyword search."""

    def test_search_returns_match(self, client_with_db):
        client, session_factory = client_with_db
        db = session_factory()
        db.add(Movie(tmdb_id=550, title="Fight Club", overview="An underground fight club."))
        db.add(Movie(tmdb_id=551, title="Forrest Gump", overview="Life is like a box of chocolates."))
        db.commit()
        db.close()

        response = client.get("/api/v1/movies/search?q=fight")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["movies"][0]["title"] == "Fight Club"

    def test_search_no_results(self, client_with_db):
        client, _ = client_with_db
        response = client.get("/api/v1/movies/search?q=zxzxzxnomatch")
        assert response.status_code == 200
        assert response.json()["total"] == 0

    def test_search_requires_query_param(self, client_with_db):
        client, _ = client_with_db
        response = client.get("/api/v1/movies/search")
        assert response.status_code == 422

    def test_search_matches_overview(self, client_with_db):
        client, session_factory = client_with_db
        db = session_factory()
        db.add(Movie(
            tmdb_id=552,
            title="Obscure Title",
            overview="This is about a unique word: glockenspiel",
        ))
        db.commit()
        db.close()

        response = client.get("/api/v1/movies/search?q=glockenspiel")
        assert response.json()["total"] == 1


class TestGetMovieDetailEndpoint:
    """GET /api/v1/movies/{id} — single movie detail."""

    def test_returns_movie_when_found(self, client_with_db):
        client, session_factory = client_with_db
        db = session_factory()
        movie = Movie(
            tmdb_id=550,
            title="Fight Club",
            release_date=date(1999, 10, 15),
            runtime=139,
        )
        db.add(movie)
        db.commit()
        movie_id = movie.id
        db.close()

        response = client.get(f"/api/v1/movies/{movie_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == movie_id
        assert body["title"] == "Fight Club"
        assert body["runtime"] == 139

    def test_returns_404_when_not_found(self, client_with_db):
        client, _ = client_with_db
        response = client.get("/api/v1/movies/99999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestSyncEndpoint:
    """POST /api/v1/movies/sync — TMDB sync trigger."""

    def test_sync_with_mocked_tmdb_client(self, client_with_db):
        """
        Verify the sync endpoint calls the service and returns SyncResult.
        The TMDB client is mocked at the dependency level — no real API call.
        """
        client, _ = client_with_db

        mock_client = MagicMock(spec=TMDBClient)
        mock_client.get_genres.return_value = FAKE_GENRES
        mock_client.get_popular_movies.return_value = FAKE_POPULAR_PAGE

        from app.api.routes.movies import get_tmdb_client
        app = client.app
        app.dependency_overrides[get_tmdb_client] = lambda: mock_client

        response = client.post("/api/v1/movies/sync")
        assert response.status_code == 200
        body = response.json()
        assert "genres_synced" in body
        assert "movies_synced" in body
        assert "errors" in body
        assert "message" in body
        assert body["errors"] == 0

        app.dependency_overrides.pop(get_tmdb_client, None)

    def test_sync_returns_503_when_api_key_missing(self, sqlite_session_factory):
        """If TMDB_API_KEY is empty, the sync endpoint returns 503."""
        def override_get_db():
            db = sqlite_session_factory()
            try:
                yield db
            finally:
                db.close()

        # Settings with no API key
        empty_key_settings = Settings(
            tmdb_api_key="",
            database_url="postgresql+psycopg://postgres:postgres@localhost:5432/cinemind",
        )

        from app.core.config import get_settings
        app = create_app()
        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_settings] = lambda: empty_key_settings

        test_client = TestClient(app)
        response = test_client.post("/api/v1/movies/sync")
        assert response.status_code == 503
        app.dependency_overrides.clear()


# ===========================================================================
# TMDB key presence check (safe — does not expose the key)
# ===========================================================================

class TestTMDBKeyConfiguration:
    """Verify the TMDB API key is properly loaded from environment (not blank)."""

    def test_tmdb_api_key_is_set_in_env(self):
        """
        Check that the settings object has a non-empty TMDB API key.

        This test uses the real Settings (which reads the .env file).
        It does NOT print, log, or assert the key value — only its presence.

        If this test fails, add TMDB_API_KEY to backend/.env.
        """
        from app.core.config import get_settings
        settings = get_settings()
        assert settings.tmdb_api_key, (
            "TMDB_API_KEY is not set in .env — add it to backend/.env. "
            "See backend/.env.example for the expected variable name."
        )

    def test_tmdb_api_key_not_default_placeholder(self):
        """
        The key must not be the example placeholder value.
        """
        from app.core.config import get_settings
        settings = get_settings()
        assert settings.tmdb_api_key != "replace-with-your-tmdb-api-key", (
            "TMDB_API_KEY still has the placeholder value — replace it with a real key."
        )
