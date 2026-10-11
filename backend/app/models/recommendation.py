"""
SQLAlchemy ORM model for the 'recommendation_results' table.

Milestone 8 scope: MVP Recommendations (Content-Based)
Milestone 12 additions:
  - algorithm: records which engine produced the row so content-based,
    collaborative, and hybrid results can be queried/compared independently.
  - reason_text: stores a human-readable explanation for a recommendation.
    Populated by the hybrid/explainability layer (Milestone 16). NULL for
    all programmatically generated rows until that milestone is reached.

Design decisions:
- Caches recommendations generated on-the-fly for users.
- CASCADE DELETE on user_id and movie_id ensures cleanup.
- algorithm defaults to 'content_based' so all pre-M12 rows are labelled
  correctly without requiring a data backfill.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base

# Valid algorithm identifiers — used here for documentation; enforced in
# the service layer rather than a DB CHECK to keep SQLite compatibility.
ALGORITHM_CONTENT_BASED = "content_based"
ALGORITHM_COLLABORATIVE = "collaborative"
ALGORITHM_HYBRID = "hybrid"
ALGORITHM_POPULARITY_BASED = "popularity_based"


class RecommendationResult(Base):
    """Cached recommendation results for a user."""

    __tablename__ = "recommendation_results"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    movie_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("movies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    # algorithm: which engine produced this row (Milestone 12).
    # One of: 'content_based', 'collaborative', 'hybrid', 'popularity_based'.
    algorithm: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        server_default=ALGORITHM_CONTENT_BASED,
    )
    # reason_text: human-readable explanation (Milestone 16 explainability).
    # NULL until the explainability layer is wired in.
    reason_text: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="recommendation_results")  # noqa: F821
    movie: Mapped["Movie"] = relationship("Movie", back_populates="recommendation_results")  # noqa: F821

    def __repr__(self) -> str:
        return (
            f"<RecommendationResult id={self.id} user_id={self.user_id} "
            f"movie_id={self.movie_id} score={self.relevance_score} "
            f"algorithm={self.algorithm!r}>"
        )
