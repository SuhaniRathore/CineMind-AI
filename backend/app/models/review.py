"""
SQLAlchemy ORM model for the 'reviews' table.

Milestone 7 scope: Create reviews table to allow users to write text reviews for movies.
Design decisions:
- A user can only write one review per movie (unique constraint on user_id, movie_id).
- Title is up to 150 chars, body is up to 5000 chars.
- CASCADE DELETE on user_id and movie_id ensures reviews are cleaned up if the parent is removed.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Review(Base):
    """User review for a movie."""

    __tablename__ = "reviews"
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_reviews_user_movie"),
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
    title: Mapped[str] = mapped_column(String(150), nullable=False)
    body: Mapped[str] = mapped_column(String(5000), nullable=False)
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
    user: Mapped["User"] = relationship("User", back_populates="reviews")  # noqa: F821
    movie: Mapped["Movie"] = relationship("Movie", back_populates="reviews")  # noqa: F821

    def __repr__(self) -> str:
        return f"<Review id={self.id} user_id={self.user_id} movie_id={self.movie_id} title={self.title!r}>"
