"""
SQLAlchemy ORM model for the 'movies' table and the 'movie_genres' join table.

Design decisions (confirmed before implementation):
- movies is a CACHE of TMDB data — tmdb_id is the natural external key.
- imdb_id is added at Milestone 12 to bridge our movie cache with the
  MovieLens dataset via the links.csv mapping file (movieId → imdbId → tmdbId).
  It is nullable because most rows were synced before this column existed.
- CASCADE DELETE on movie deletion: when a movie row is deleted, all child
  rows in ratings, reviews, watchlist, watch_history, movie_genres, and
  recommendation_results are automatically removed. This is enforced at the
  DB level via ondelete="CASCADE" on each child FK.
- overview_embedding (pgvector) is deferred to V2 Milestone 15.
- popularity, vote_average, vote_count are nullable — not every movie has
  them on first cache, and they are refreshed periodically from TMDB.
- title is indexed to support fast basic title searches (Milestone 4).
- cached_at records the last TMDB sync timestamp.

Relationships to child tables (ratings, reviews, watchlist, watch_history,
recommendation_results) are added at the milestones that create those tables.
"""

from datetime import date, datetime

from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, String, Table, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base

# ---------------------------------------------------------------------------
# Association table: movie_genres (many-to-many between movies and genres)
# Composite PK on (movie_id, genre_id) per DATABASE_DESIGN.md.
# CASCADE on both FKs: removing a movie or genre removes its join rows.
# ---------------------------------------------------------------------------
movie_genres = Table(
    "movie_genres",
    Base.metadata,
    Column(
        "movie_id",
        Integer,
        ForeignKey("movies.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "genre_id",
        Integer,
        ForeignKey("genres.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class Movie(Base):
    """
    Movie record, cached from TMDB.

    tmdb_id is the stable external identifier — used to fetch/refresh
    metadata from the TMDB API (Milestone 4). All TMDB-sourced fields are
    stored here to avoid live outbound calls on every request.
    """

    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    tmdb_id: Mapped[int] = mapped_column(
        Integer, unique=True, index=True, nullable=False
    )
    # imdb_id: added at Milestone 12 to enable MovieLens cross-referencing.
    # Nullable because existing rows were cached before this column existed.
    # The partial unique index (ix_movies_imdb_id) in the Alembic migration
    # enforces uniqueness only for non-NULL values.
    imdb_id: Mapped[str | None] = mapped_column(
        String(50), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    overview: Mapped[str | None] = mapped_column(String, nullable=True)
    release_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    runtime: Mapped[int | None] = mapped_column(Integer, nullable=True)  # minutes
    poster_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    original_language: Mapped[str | None] = mapped_column(String(10), nullable=True)

    # TMDB aggregate stats — nullable; refreshed periodically (Milestone 4+)
    popularity: Mapped[float | None] = mapped_column(Float, nullable=True)
    vote_average: Mapped[float | None] = mapped_column(Float, nullable=True)
    vote_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # overview_embedding (pgvector) intentionally absent — added at V2 Milestone 15.

    # ------------------------------------------------------------------
    # Derived NLP analysis fields (Milestone 13)
    # ------------------------------------------------------------------
    review_summary: Mapped[str | None] = mapped_column(String, nullable=True)
    review_summary_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cached_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp of last sync from TMDB",
    )

    # Many-to-many: movies ↔ genres via movie_genres join table
    genres: Mapped[list["Genre"]] = relationship(  # noqa: F821
        "Genre",
        secondary=movie_genres,
        backref="movies",
        lazy="select",
    )

    # Relationships to child tables are added at the milestone that creates them:
    #   - ratings              → Milestone 6
    #   - reviews, watchlist   → Milestone 7
    #   - recommendation_results → Milestone 8
    # DB-level CASCADE DELETE is enforced via ondelete="CASCADE" on each child FK.

    def __repr__(self) -> str:
        return f"<Movie id={self.id} tmdb_id={self.tmdb_id} title={self.title!r}>"

    # Relationships
    ratings: Mapped[list["Rating"]] = relationship(  # noqa: F821
        "Rating",
        back_populates="movie",
        cascade="all, delete-orphan",
    )
    reviews: Mapped[list["Review"]] = relationship(  # noqa: F821
        "Review",
        back_populates="movie",
        cascade="all, delete-orphan",
    )
    watchlist: Mapped[list["WatchlistItem"]] = relationship(  # noqa: F821
        "WatchlistItem",
        back_populates="movie",
        cascade="all, delete-orphan",
    )
    recommendation_results: Mapped[list["RecommendationResult"]] = relationship(  # noqa: F821
        "RecommendationResult",
        back_populates="movie",
        cascade="all, delete-orphan",
    )
