"""ml_foundation_schema

Milestone 12 — Data Expansion & ML Foundation.

Adds three new columns required for the ML/data pipeline:

1. movies.imdb_id (nullable String, unique)
   - Links our movie cache to IMDb / MovieLens via the links.csv bridge file.
   - Nullable because most existing rows were synced before this migration.
   - Populated by the TMDB enricher or MovieLens loader (Milestone 12).

2. recommendation_results.algorithm (String(50), NOT NULL, default 'content_based')
   - Records which engine produced each row:
       'content_based'  — TF-IDF genre/metadata similarity (existing)
       'collaborative'  — SVD matrix factorisation (new M12)
       'hybrid'         — weighted blend (new M12)
   - Defaults to 'content_based' so all existing rows are correctly labelled.

3. recommendation_results.reason_text (nullable String(500))
   - Human-readable explanation for a recommendation
     (e.g. "Because you liked Action movies highly rated").
   - NULL for programmatically generated rows until the explainability layer
     is wired in (planned for Milestone 16 per ROADMAP.md).

All changes are backwards-compatible:
  - New columns are nullable or have server defaults — no existing rows break.
  - downgrade() removes the columns cleanly.

Revision ID: a1b2c3d4e5f6
Revises: b846381dfe71
Create Date: 2026-09-04
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "b846381dfe71"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # movies: add imdb_id
    # ------------------------------------------------------------------
    op.add_column(
        "movies",
        sa.Column("imdb_id", sa.String(length=50), nullable=True),
    )
    # Unique index — IMDb IDs are globally unique across movies.
    # Using a partial/conditional unique index is not portable, so we
    # use a standard unique index; NULL values are allowed (multiple NULLs
    # are permitted by PostgreSQL's unique index semantics).
    op.create_index(
        "ix_movies_imdb_id",
        "movies",
        ["imdb_id"],
        unique=True,
        postgresql_where=sa.text("imdb_id IS NOT NULL"),
    )

    # ------------------------------------------------------------------
    # recommendation_results: add algorithm + reason_text
    # ------------------------------------------------------------------
    op.add_column(
        "recommendation_results",
        sa.Column(
            "algorithm",
            sa.String(length=50),
            nullable=False,
            server_default="content_based",
        ),
    )
    op.add_column(
        "recommendation_results",
        sa.Column("reason_text", sa.String(length=500), nullable=True),
    )


def downgrade() -> None:
    # Remove in reverse order of addition
    op.drop_column("recommendation_results", "reason_text")
    op.drop_column("recommendation_results", "algorithm")
    op.drop_index("ix_movies_imdb_id", table_name="movies")
    op.drop_column("movies", "imdb_id")
