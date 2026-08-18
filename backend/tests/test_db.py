"""Tests for the database foundation (Milestone 2).

These tests verify the database layer is correctly configured without
requiring a live PostgreSQL instance — they unit-test structure and
behaviour, not actual connectivity.

Tests are grouped into three classes:
  - TestDatabaseSettings  : environment config exposes a valid DB URL.
  - TestSQLAlchemySetup   : engine, session factory, and Base are wired correctly.
  - TestDbConnectivityCheck: check_db_connection() always returns bool, never raises.
"""

import inspect

from sqlalchemy import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings
from app.db.database import (
    Base,
    SessionLocal,
    check_db_connection,
    engine,
    get_db,
)


class TestDatabaseSettings:
    """Settings correctly expose a usable database URL."""

    def test_settings_has_database_url(self) -> None:
        settings = get_settings()
        assert settings.database_url, "DATABASE_URL must not be empty"

    def test_database_url_targets_postgresql(self) -> None:
        settings = get_settings()
        assert settings.database_url.startswith(
            "postgresql"
        ), "DATABASE_URL must use a postgresql:// scheme"

    def test_database_url_uses_psycopg_dialect(self) -> None:
        """URL must use the psycopg3 dialect (postgresql+psycopg://)."""
        settings = get_settings()
        assert "psycopg" in settings.database_url, (
            "DATABASE_URL should use the psycopg3 driver "
            "(postgresql+psycopg://...) to match requirements.txt"
        )


class TestSQLAlchemySetup:
    """SQLAlchemy engine, session factory, and Base are correctly initialised."""

    def test_engine_is_sqlalchemy_engine(self) -> None:
        assert isinstance(engine, Engine)

    def test_engine_targets_postgresql(self) -> None:
        assert engine.dialect.name == "postgresql", (
            f"Expected 'postgresql' dialect, got '{engine.dialect.name}'"
        )

    def test_session_local_is_sessionmaker(self) -> None:
        assert isinstance(
            SessionLocal, sessionmaker
        ), "SessionLocal must be a SQLAlchemy sessionmaker instance"

    def test_session_local_not_autocommit(self) -> None:
        """Sessions must not autocommit — transactions should be explicit."""
        assert SessionLocal.kw.get("autocommit") is False

    def test_session_local_not_autoflush(self) -> None:
        """Sessions must not autoflush — the caller controls flush boundaries."""
        assert SessionLocal.kw.get("autoflush") is False

    def test_base_is_declarative_base(self) -> None:
        """Base must be a subclass of DeclarativeBase."""
        assert issubclass(Base, DeclarativeBase), (
            "Base should inherit from sqlalchemy.orm.DeclarativeBase"
        )

    def test_base_has_metadata(self) -> None:
        """Base must expose metadata so Alembic can introspect the schema."""
        assert hasattr(Base, "metadata")

    def test_get_db_is_generator_function(self) -> None:
        """get_db must be a generator function (used as a FastAPI dependency)."""
        assert inspect.isgeneratorfunction(get_db), (
            "get_db should be a generator function so FastAPI can inject "
            "and clean up the session via 'yield'"
        )


class TestDbConnectivityCheck:
    """check_db_connection() is safe to call regardless of DB availability."""

    def test_returns_bool(self) -> None:
        result = check_db_connection()
        assert isinstance(result, bool), (
            f"check_db_connection() must return bool, got {type(result)}"
        )

    def test_never_raises(self) -> None:
        """Even when PostgreSQL is unreachable, no exception must propagate."""
        try:
            check_db_connection()
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"check_db_connection() raised an unexpected exception: {exc}"
            ) from exc
