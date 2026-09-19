"""
Tests for the ORM models (Milestone 3).

These tests verify model structure — column names, types, constraints, and
relationships — without requiring a live PostgreSQL instance.

We use SQLite (in-memory) as the test database for structural tests because:
  1. No PostgreSQL required in CI or local dev for unit tests.
  2. SQLite supports the subset of SQL we use (no pgvector, no DB-specific types).
  3. Integration tests against real PostgreSQL are run separately (manual / CI).

The test database URL is injected via the TESTING_DATABASE_URL pattern, but
for structural column inspection we go directly via Base.metadata — no
connection required at all.
"""

import inspect

import pytest
from sqlalchemy import Boolean, Date, DateTime, Float, Integer, String, inspect as sa_inspect
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.models import Genre, Movie, User, movie_genres


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(model, name):
    """Return the Column object for a model attribute by column name."""
    return model.__table__.c[name]


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------

class TestUserModel:
    """User table structure matches DATABASE_DESIGN.md."""

    def test_tablename(self):
        assert User.__tablename__ == "users"

    def test_expected_columns(self):
        cols = {c.name for c in User.__table__.columns}
        assert cols == {"id", "email", "hashed_password", "display_name", "is_active", "created_at"}

    def test_id_is_primary_key(self):
        assert _col(User, "id").primary_key

    def test_email_is_unique_and_indexed(self):
        col = _col(User, "email")
        assert col.unique
        # Check a named unique index/constraint exists
        constraint_names = {c.name for c in User.__table__.constraints}
        assert "uq_users_email" in constraint_names

    def test_email_max_length(self):
        assert _col(User, "email").type.length == 255

    def test_hashed_password_not_nullable(self):
        assert not _col(User, "hashed_password").nullable

    def test_display_name_max_length(self):
        assert _col(User, "display_name").type.length == 100

    def test_is_active_is_boolean(self):
        assert isinstance(_col(User, "is_active").type, Boolean)

    def test_created_at_is_timezone_aware(self):
        col = _col(User, "created_at")
        assert isinstance(col.type, DateTime)
        assert col.type.timezone is True

    def test_no_plaintext_password_column(self):
        """Ensure there is no 'password' column — only 'hashed_password'."""
        col_names = {c.name for c in User.__table__.columns}
        assert "password" not in col_names


# ---------------------------------------------------------------------------
# Genre model
# ---------------------------------------------------------------------------

class TestGenreModel:
    """Genre table structure matches DATABASE_DESIGN.md."""

    def test_tablename(self):
        assert Genre.__tablename__ == "genres"

    def test_expected_columns(self):
        cols = {c.name for c in Genre.__table__.columns}
        assert cols == {"id", "tmdb_genre_id", "name"}

    def test_id_is_primary_key(self):
        assert _col(Genre, "id").primary_key

    def test_tmdb_genre_id_is_unique(self):
        assert _col(Genre, "tmdb_genre_id").unique

    def test_name_max_length(self):
        assert _col(Genre, "name").type.length == 100


# ---------------------------------------------------------------------------
# Movie model
# ---------------------------------------------------------------------------

class TestMovieModel:
    """Movie table structure matches DATABASE_DESIGN.md."""

    def test_tablename(self):
        assert Movie.__tablename__ == "movies"

    def test_expected_columns(self):
        cols = {c.name for c in Movie.__table__.columns}
        expected = {
            "id", "tmdb_id", "imdb_id", "title", "overview", "release_date",
            "runtime", "poster_path", "original_language",
            "popularity", "vote_average", "vote_count", "cached_at",
            "review_summary", "review_summary_updated_at"
        }
        assert cols == expected

    def test_no_overview_embedding_column(self):
        """overview_embedding (pgvector) must NOT exist until V2 Milestone 15."""
        col_names = {c.name for c in Movie.__table__.columns}
        assert "overview_embedding" not in col_names

    def test_tmdb_id_is_unique(self):
        assert _col(Movie, "tmdb_id").unique

    def test_title_is_indexed(self):
        index_cols = {
            col.name
            for idx in Movie.__table__.indexes
            for col in idx.columns
        }
        assert "title" in index_cols

    def test_nullable_optional_fields(self):
        for col_name in ("overview", "release_date", "runtime", "poster_path",
                         "original_language", "popularity", "vote_average", "vote_count"):
            assert _col(Movie, col_name).nullable, f"{col_name} should be nullable"

    def test_cached_at_timezone_aware(self):
        col = _col(Movie, "cached_at")
        assert isinstance(col.type, DateTime)
        assert col.type.timezone is True

    def test_imdb_id_type(self):
        assert isinstance(_col(Movie, "imdb_id").type, String)
        assert _col(Movie, "imdb_id").type.length == 50


# ---------------------------------------------------------------------------
# movie_genres association table
# ---------------------------------------------------------------------------

class TestMovieGenresTable:
    """movie_genres join table structure matches DATABASE_DESIGN.md."""

    def test_table_exists_in_metadata(self):
        assert "movie_genres" in Base.metadata.tables

    def test_has_composite_primary_key(self):
        pk_cols = {col.name for col in movie_genres.primary_key}
        assert pk_cols == {"movie_id", "genre_id"}

    def test_fk_to_movies_with_cascade(self):
        fk = next(
            fk for fk in movie_genres.foreign_keys
            if fk.column.table.name == "movies"
        )
        assert fk.ondelete == "CASCADE"

    def test_fk_to_genres_with_cascade(self):
        fk = next(
            fk for fk in movie_genres.foreign_keys
            if fk.column.table.name == "genres"
        )
        assert fk.ondelete == "CASCADE"


# ---------------------------------------------------------------------------
# All 4 expected tables are in metadata (no more, no fewer for Milestone 3)
# ---------------------------------------------------------------------------

class TestMetadataTables:
    def test_exactly_expected_tables_registered(self):
        tables = set(Base.metadata.tables.keys())
        assert tables == {"users", "movies", "genres", "movie_genres", "ratings", "reviews", "watchlist", "recommendation_results"}


# ---------------------------------------------------------------------------
# SQLite structural smoke test: tables can actually be created
# ---------------------------------------------------------------------------

class TestTablesCreateOnSQLite:
    """
    Creates all tables on an in-memory SQLite DB and verifies they exist.
    This catches column type or FK definition errors without needing PostgreSQL.
    """

    def test_create_all_tables(self):
        engine = create_engine("sqlite:///:memory:")
        # SQLite doesn't enforce FKs by default; enable for correctness
        with engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
        Base.metadata.create_all(engine)
        insp = sa_inspect(engine)
        created = set(insp.get_table_names())
        assert {"users", "movies", "genres", "movie_genres"}.issubset(created)
        Base.metadata.drop_all(engine)

    def test_insert_and_query_on_sqlite(self):
        """Round-trip test: insert a user, movie, genre, and link them."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)

        with Session() as db:
            genre = Genre(tmdb_genre_id=28, name="Action")
            user = User(
                email="test@example.com",
                hashed_password="$2b$12$hashed",
                display_name="Tester",
            )
            movie = Movie(
                tmdb_id=550,
                title="Fight Club",
                overview="First rule of Fight Club...",
                popularity=8.4,
                vote_average=8.8,
                vote_count=24000,
            )
            db.add_all([genre, user, movie])
            db.flush()

            # Link movie to genre via many-to-many
            movie.genres.append(genre)
            db.commit()

            # Verify reads
            saved_user = db.query(User).filter_by(email="test@example.com").one()
            assert saved_user.display_name == "Tester"

            saved_movie = db.query(Movie).filter_by(tmdb_id=550).one()
            assert saved_movie.title == "Fight Club"
            assert len(saved_movie.genres) == 1
            assert saved_movie.genres[0].name == "Action"

        Base.metadata.drop_all(engine)
