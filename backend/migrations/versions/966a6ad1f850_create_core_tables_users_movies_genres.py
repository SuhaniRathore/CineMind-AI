"""create_core_tables_users_movies_genres

Milestone 3 initial migration.

Creates the 4 core tables specified in ROADMAP.md Milestone 3:
  - users
  - genres
  - movies
  - movie_genres  (join table)

Design decisions documented in DATABASE_DESIGN.md and confirmed before
implementation:
  - CASCADE DELETE on all FK relationships.
  - is_active flag on users for soft-disabling without data loss.
  - overview_embedding (pgvector) intentionally absent — V2 Milestone 15.
  - All timestamp columns use timezone-aware UTC.

Revision ID: 966a6ad1f850
Revises: (initial migration, no parent)
Create Date: 2026-08-16
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "966a6ad1f850"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # genres — no FKs; created first so movies can reference it
    # ------------------------------------------------------------------
    op.create_table(
        "genres",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tmdb_genre_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_genres_id", "genres", ["id"], unique=False)
    op.create_index("ix_genres_tmdb_genre_id", "genres", ["tmdb_genre_id"], unique=True)

    # ------------------------------------------------------------------
    # users — no FKs to other tables in this migration
    # ------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=100), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_id", "users", ["id"], unique=False)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # ------------------------------------------------------------------
    # movies — no FK to genres (handled via movie_genres join table)
    # ------------------------------------------------------------------
    op.create_table(
        "movies",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tmdb_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("overview", sa.String(), nullable=True),
        sa.Column("release_date", sa.Date(), nullable=True),
        sa.Column("runtime", sa.Integer(), nullable=True),
        sa.Column("poster_path", sa.String(length=500), nullable=True),
        sa.Column("original_language", sa.String(length=10), nullable=True),
        sa.Column("popularity", sa.Float(), nullable=True),
        sa.Column("vote_average", sa.Float(), nullable=True),
        sa.Column("vote_count", sa.Integer(), nullable=True),
        sa.Column(
            "cached_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="Timestamp of last sync from TMDB",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tmdb_id", name="uq_movies_tmdb_id"),
    )
    op.create_index("ix_movies_id", "movies", ["id"], unique=False)
    op.create_index("ix_movies_tmdb_id", "movies", ["tmdb_id"], unique=True)
    op.create_index("ix_movies_title", "movies", ["title"], unique=False)

    # ------------------------------------------------------------------
    # movie_genres — join table; composite PK; CASCADE on both FKs
    # ------------------------------------------------------------------
    op.create_table(
        "movie_genres",
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("genre_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["movie_id"],
            ["movies.id"],
            name="fk_movie_genres_movie_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["genre_id"],
            ["genres.id"],
            name="fk_movie_genres_genre_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("movie_id", "genre_id", name="pk_movie_genres"),
    )


def downgrade() -> None:
    # Drop in reverse dependency order
    op.drop_table("movie_genres")
    op.drop_index("ix_movies_title", table_name="movies")
    op.drop_index("ix_movies_tmdb_id", table_name="movies")
    op.drop_index("ix_movies_id", table_name="movies")
    op.drop_table("movies")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_id", table_name="users")
    op.drop_table("users")
    op.drop_index("ix_genres_tmdb_genre_id", table_name="genres")
    op.drop_index("ix_genres_id", table_name="genres")
    op.drop_table("genres")
