"""add review NLP analysis fields

Revision ID: 3d397c68f9f0
Revises: a1b2c3d4e5f6
Create Date: 2026-09-09 11:42:08.383210

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3d397c68f9f0"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add NLP analysis fields to reviews."""
    op.add_column(
        "reviews",
        sa.Column("sentiment_label", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "reviews",
        sa.Column("sentiment_score", sa.Float(), nullable=True),
    )
    op.add_column(
        "reviews",
        sa.Column("aspect_sentiment", sa.JSON(), nullable=True),
    )
    op.add_column(
        "reviews",
        sa.Column(
            "nlp_processed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Remove NLP analysis fields from reviews."""
    op.drop_column("reviews", "nlp_processed_at")
    op.drop_column("reviews", "aspect_sentiment")
    op.drop_column("reviews", "sentiment_score")
    op.drop_column("reviews", "sentiment_label")