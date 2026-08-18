"""
SQLAlchemy ORM model for the 'genres' table.

Genres are seeded from TMDB's fixed genre list (Milestone 4).
tmdb_genre_id is TMDB's stable integer genre ID (e.g. 28 = Action).

The many-to-many relationship to Movie is via the movie_genres join table
defined in movie.py. The back-reference is added there.
"""

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class Genre(Base):
    """Movie genre — sourced from TMDB's fixed genre catalogue."""

    __tablename__ = "genres"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    tmdb_genre_id: Mapped[int] = mapped_column(
        Integer, unique=True, index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Relationship to Movie via movie_genres join table.
    # movies relationship is declared on Movie and uses secondary="movie_genres",
    # SQLAlchemy resolves both sides automatically.

    def __repr__(self) -> str:
        return f"<Genre id={self.id} name={self.name!r} tmdb_genre_id={self.tmdb_genre_id}>"
