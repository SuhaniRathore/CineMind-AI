"""
SQLAlchemy ORM model for the 'recommendation_results' table.

Milestone 8 scope: MVP Recommendations (Content-Based)
Design decisions:
- Caches recommendations generated on-the-fly for users.
- CASCADE DELETE on user_id and movie_id ensures cleanup.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


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

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="recommendation_results")  # noqa: F821
    movie: Mapped["Movie"] = relationship("Movie", back_populates="recommendation_results")  # noqa: F821

    def __repr__(self) -> str:
        return f"<RecommendationResult id={self.id} user_id={self.user_id} movie_id={self.movie_id} score={self.relevance_score}>"
