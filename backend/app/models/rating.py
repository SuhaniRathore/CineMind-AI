"""
SQLAlchemy ORM model for the 'ratings' table.

Milestone 6 scope: Create ratings table to allow users to rate movies.
Design decisions:
- Rating scale is strictly an integer from 1 to 10.
- A user can only rate a specific movie once. Enforced by uq_ratings_user_movie constraint.
- CASCADE DELETE on user_id and movie_id ensures ratings are cleaned up if the parent is removed.
"""

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Rating(Base):
    """User rating for a movie."""

    __tablename__ = "ratings"
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_ratings_user_movie"),
        CheckConstraint("rating >= 1 AND rating <= 10", name="chk_ratings_scale"),
    )

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
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships back to User and Movie
    user: Mapped["User"] = relationship("User", back_populates="ratings")  # noqa: F821
    movie: Mapped["Movie"] = relationship("Movie", back_populates="ratings")  # noqa: F821

    def __repr__(self) -> str:
        return f"<Rating id={self.id} user_id={self.user_id} movie_id={self.movie_id} rating={self.rating}>"
