"""
ORM models package.

All model modules must be imported here so that:
1. SQLAlchemy's mapper registry sees every model before Alembic inspects
   Base.metadata — without this, autogenerate would miss tables.
2. The app can do `from app.models import User, Movie, Genre` cleanly.

Import order: Base first (via app.db.database), then models with no FKs
(Genre), then models with FKs (Movie which references Genre via movie_genres).
"""

from app.models.genre import Genre  # noqa: F401 — must be before Movie
from app.models.movie import Movie, movie_genres  # noqa: F401
from app.models.user import User  # noqa: F401
from app.models.rating import Rating  # noqa: F401

from app.models.review import Review  # noqa: F401
from app.models.watchlist import WatchlistItem  # noqa: F401

__all__ = ["Genre", "Movie", "movie_genres", "User", "Rating", "Review", "WatchlistItem"]
