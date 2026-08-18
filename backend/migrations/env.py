"""
Alembic environment configuration for CineMind AI.

Key customisations vs. the default generated env.py:
1. DATABASE_URL is read from app.core.config.get_settings() (which loads from
   the .env file), so no secrets ever live in alembic.ini or this file.
2. target_metadata is set to Base.metadata after importing all model modules,
   enabling autogenerate to detect schema changes.
3. All models are imported via `app.models` to ensure their tables are
   registered in Base.metadata before Alembic inspects it.
"""

import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# ---------------------------------------------------------------------------
# Make sure the backend/ directory is on sys.path so `from app.xxx` works
# when Alembic is invoked from the backend/ directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ---------------------------------------------------------------------------
# Import application settings and Base *before* models so the engine and
# metadata object exist when models are imported.
# ---------------------------------------------------------------------------
from app.core.config import get_settings  # noqa: E402
from app.db.database import Base  # noqa: E402

# Import all models so they register their tables in Base.metadata.
# Without these imports, autogenerate would see an empty schema.
import app.models  # noqa: E402, F401

# ---------------------------------------------------------------------------
# Alembic config object (access to alembic.ini values)
# ---------------------------------------------------------------------------
config = context.config

# Override sqlalchemy.url with the value from our Settings (reads .env).
# This is why alembic.ini has a blank sqlalchemy.url — the real URL
# never lives in a committed file.
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

# Set up Python logging from the alembic.ini [loggers] section.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The metadata object that autogenerate compares against the live DB schema.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode (generates SQL without a live connection).

    Useful for generating migration scripts to review before applying, or for
    environments where direct DB access isn't available.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Render constraint names so Alembic can detect/drop named constraints.
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode (applies migrations against a live DB).

    Uses NullPool so each migration run gets a fresh connection — important
    for correctness when running DDL statements.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Compare server defaults so autogenerate detects server_default changes.
            compare_server_defaults=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
