"""
Database engine, session factory, and declarative base.

Milestone 2 scope: establishes the SQLAlchemy connection layer only.
No ORM table models are defined here — those arrive in Milestone 3
(see app/models/ and docs/DATABASE_DESIGN.md for the planned schema).

Architecture decision (documented per DEVELOPMENT_RULES.md rule 5):
  Synchronous SQLAlchemy is used for all MVP milestones. FastAPI
  automatically runs sync route functions in a thread pool, so there is
  no blocking concern at project scale. Async SQLAlchemy would require
  AsyncSession, async_sessionmaker, and 'async def' throughout — meaningful
  added complexity for a BTech project. This choice can be revisited before
  V2 if profiling shows a genuine throughput bottleneck.
"""

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


def _build_engine():
    """
    Create the SQLAlchemy engine from application settings.

    Called once at module import time. SQLAlchemy's create_engine() is
    lazy — no actual connection is established until the first database
    operation. Keeping this at module level lets FastAPI's dependency
    injection reuse the same connection pool across all requests.
    """
    settings = get_settings()
    return create_engine(
        settings.database_url,
        # Re-verify connections before checkout from the pool.
        # Catches stale / server-side-closed connections transparently.
        pool_pre_ping=True,
        # Echo SQL to stdout only in debug mode; never in production
        # (avoids sensitive data appearing in logs).
        echo=settings.debug,
    )


# Module-level engine — created once, shared across all requests via the pool.
engine = _build_engine()

# Session factory bound to the engine.
# autocommit=False: transactions are explicit (caller calls db.commit()).
# autoflush=False: changes are not auto-flushed before queries, giving
#                  route/service code full control over the flush boundary.
SessionLocal: sessionmaker[Session] = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


class Base(DeclarativeBase):
    """
    Declarative base class shared by all ORM models.

    Import and subclass this in each model file under app/models/:

        from app.db.database import Base

        class User(Base):
            __tablename__ = "users"
            ...

    ORM models are introduced in Milestone 3.
    """


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency: yield a database session for the duration of one
    request, then close it unconditionally.

    Usage in a route handler::

        from sqlalchemy.orm import Session
        from fastapi import Depends
        from app.db.database import get_db

        def my_route(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """
    Probe the database with a trivial query.

    Returns True if PostgreSQL is reachable, False otherwise.
    Never raises an exception — callers (e.g. the health endpoint) should
    treat False as a degraded-but-non-fatal state, not an application error.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False
