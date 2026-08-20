"""
SQLAlchemy ORM model for the 'watchlist' table.

Milestone 7 scope: Create watchlist table for users to save movies.
Design decisions:
- A user can only add a specific movie to their watchlist once (unique constraint on user_id, movie_id).
- One master watchlist per user.
- CASCADE DELETE on user_id and movie_id ensures watchlist items are cleaned up if the parent is removed.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class WatchlistItem(Base):
    """User watchlist item for a movie."""

    __tablename__ = "watchlist"
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_watchlist_user_movie"),
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships back to User and Movie
    user: Mapped["User"] = relationship("User", back_populates="watchlist")  # noqa: F821
    movie: Mapped["Movie"] = relationship("Movie", back_populates="watchlist")  # noqa: F821

    def __repr__(self) -> str:
        return f"<WatchlistItem id={self.id} user_id={self.user_id} movie_id={self.movie_id}>"
