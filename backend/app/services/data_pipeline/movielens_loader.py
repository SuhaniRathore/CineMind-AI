"""
MovieLens dataset loader — Milestone 12.

Responsibilities:
  - Parse MovieLens CSV files (movies.csv, ratings.csv, links.csv, tags.csv).
  - Map MovieLens movie IDs → our internal movie rows via tmdb_id (using links.csv).
  - Upsert new movies discovered in MovieLens that are not yet in our DB
    (using tmdb_id as the stable key, just like the TMDB sync flow).
  - Update movies.imdb_id for cross-referencing.
  - Return a training-ready ratings DataFrame (userId, internalMovieId, rating)
    with ratings normalised from MovieLens 0.5–5.0 scale to 1–10 scale
    (multiply by 2) so they are comparable to in-app ratings for model training.

IMPORTANT: MovieLens rating rows are NEVER inserted into the app `ratings` table.
  - The app table has a CHK(rating >= 1 AND rating <= 10) integer constraint.
  - MovieLens ratings are floats (0.5 step increments).
  - The mapping user_id (MovieLens) != user_id (our app) would corrupt data.
  - The training DataFrame is used exclusively by collaborative.py.

Dataset structure expected in data_dir/movielens/<variant>/:
  movies.csv   — movieId, title, genres (pipe-separated)
  ratings.csv  — userId, movieId, rating (0.5–5.0), timestamp
  links.csv    — movieId, imdbId, tmdbId
  tags.csv     — userId, movieId, tag, timestamp  (optional, not loaded yet)

Usage::

    from app.services.data_pipeline.movielens_loader import MovieLensLoader
    from app.core.config import get_settings

    settings = get_settings()
    loader = MovieLensLoader(settings)
    result = loader.load(db)
    train_df = result.training_ratings_df  # columns: ml_user_id, movie_id, rating
"""

from __future__ import annotations

import logging
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from app.models.genre import Genre
from app.models.movie import Movie

logger = logging.getLogger(__name__)

# MovieLens download URL template
_MOVIELENS_BASE_URL = "https://files.grouplens.org/datasets/movielens"


@dataclass
class MovieLensLoadResult:
    """Summary of a MovieLens load operation."""

    movies_linked: int = 0          # movies in our DB that got imdb_id linked
    movies_missing: int = 0         # MovieLens movies not found in our DB by tmdb_id
    genres_synced: int = 0          # genre rows upserted from MovieLens genre strings
    training_ratings_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: list[str] = field(default_factory=list)


class MovieLensLoader:
    """
    Load and map MovieLens dataset files to the CineMind database.

    Args:
        settings: Application settings (provides data_dir, movielens_variant).
    """

    def __init__(self, settings) -> None:
        self._data_dir = Path(settings.data_dir)
        self._variant = settings.movielens_variant  # e.g. 'ml-latest-small'
        self._dataset_dir = self._data_dir / "movielens" / self._variant

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dataset_exists(self) -> bool:
        """Return True if all required CSV files are present on disk."""
        required = ["movies.csv", "ratings.csv", "links.csv"]
        return all((self._dataset_dir / f).exists() for f in required)

    def download(self) -> Path:
        """
        Download and unzip the MovieLens dataset if not already present.

        Returns:
            Path to the extracted dataset directory.

        Raises:
            RuntimeError: If the download or extraction fails.
        """
        if self.dataset_exists():
            logger.info("MovieLens dataset already present at %s", self._dataset_dir)
            return self._dataset_dir

        zip_name = f"{self._variant}.zip"
        url = f"{_MOVIELENS_BASE_URL}/{zip_name}"
        dest_zip = self._data_dir / "movielens" / zip_name

        # Ensure parent dirs exist
        dest_zip.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading MovieLens dataset from %s …", url)
        try:
            import urllib.request
            urllib.request.urlretrieve(url, dest_zip)
        except Exception as exc:
            raise RuntimeError(f"Failed to download MovieLens from {url}: {exc}") from exc

        logger.info("Extracting %s …", dest_zip)
        try:
            with zipfile.ZipFile(dest_zip, "r") as zf:
                zf.extractall(dest_zip.parent)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"Failed to extract {dest_zip}: {exc}") from exc

        # Clean up zip after extraction
        try:
            os.remove(dest_zip)
        except OSError:
            pass

        logger.info("MovieLens dataset ready at %s", self._dataset_dir)
        return self._dataset_dir

    def load(self, db: "Session") -> MovieLensLoadResult:
        """
        Parse MovieLens CSVs and:
          1. Link existing movies by populating movies.imdb_id.
          2. Upsert genre rows discovered in MovieLens but not yet in our DB.
          3. Build and return a training-ready ratings DataFrame.

        Args:
            db: Active SQLAlchemy session (caller is responsible for commit).

        Returns:
            MovieLensLoadResult with summary stats and training DataFrame.
        """
        if not self.dataset_exists():
            raise FileNotFoundError(
                f"MovieLens dataset not found at {self._dataset_dir}. "
                "Run MovieLensLoader.download() first, or place the CSV files manually."
            )

        result = MovieLensLoadResult()

        # --- 1. Load raw CSVs ---
        logger.info("Loading MovieLens CSVs from %s …", self._dataset_dir)
        ml_movies_df = self._load_movies_csv()
        links_df = self._load_links_csv()
        ratings_df = self._load_ratings_csv()

        # --- 2. Merge movies with links to get tmdb_id ---
        # links.csv: movieId (int), imdbId (str e.g. '0114709'), tmdbId (int/float)
        merged = ml_movies_df.merge(links_df, on="movieId", how="left")
        merged["tmdbId"] = pd.to_numeric(merged["tmdbId"], errors="coerce").fillna(0).astype(int)

        # Ensure imdb_id is a string prefixed with 'tt' like TMDB expects
        merged["imdb_id_str"] = "tt" + merged["imdbId"].astype(str)

        # --- 3. Upsert genres from MovieLens genre strings ---
        result.genres_synced = self._sync_genres(db, ml_movies_df)

        # --- 4. Link movies: update imdb_id for known tmdb_ids ---
        ml_id_to_imdb = dict(zip(merged["tmdbId"], merged["imdb_id_str"]))
        ml_id_to_ml_movie_id = dict(zip(merged["tmdbId"], merged["movieId"]))

        # Build a mapping: tmdb_id → internal movie.id  (for training data)
        tmdb_to_internal: dict[int, int] = {}

        all_movies = db.query(Movie).filter(Movie.tmdb_id.in_(list(ml_id_to_imdb.keys()))).all()
        for movie in all_movies:
            imdb_str = ml_id_to_imdb.get(movie.tmdb_id, "")
            if imdb_str and movie.imdb_id != imdb_str:
                movie.imdb_id = imdb_str
                result.movies_linked += 1
            tmdb_to_internal[movie.tmdb_id] = movie.id

        db.flush()
        logger.info(
            "MovieLens link: %d movies linked with imdb_id, %d tmdb_ids not in our DB",
            result.movies_linked,
            len(merged) - len(all_movies),
        )
        result.movies_missing = len(merged) - len(all_movies)

        # --- 5. Build training ratings DataFrame ---
        # Map ml movieId → our internal movie_id (via tmdb_id bridge)
        tmdb_series = merged.set_index("movieId")["tmdbId"]
        ml_movie_to_tmdb = tmdb_series.to_dict()

        # Expand to: ml_movie_id → internal_movie_id
        ml_to_internal = {
            ml_id: tmdb_to_internal[tmdb_id]
            for ml_id, tmdb_id in ml_movie_to_tmdb.items()
            if tmdb_id in tmdb_to_internal
        }

        train_df = ratings_df[ratings_df["movieId"].isin(ml_to_internal)].copy()
        train_df["movie_id"] = train_df["movieId"].map(ml_to_internal)

        # Normalise MovieLens 0.5–5.0 float rating → 1–10 scale (×2)
        # This is for MODEL TRAINING ONLY — not written to the DB ratings table.
        train_df["rating_normalised"] = (train_df["rating"] * 2).clip(1, 10)

        result.training_ratings_df = train_df[["userId", "movie_id", "rating_normalised"]].rename(
            columns={"userId": "ml_user_id", "rating_normalised": "rating"}
        ).reset_index(drop=True)

        # Prefix MovieLens user IDs to prevent collision with app user IDs
        result.training_ratings_df["ml_user_id"] = "ml_" + result.training_ratings_df["ml_user_id"].astype(str)

        logger.info(
            "Training DataFrame ready: %d rows, %d unique ML users, %d unique movies",
            len(result.training_ratings_df),
            result.training_ratings_df["ml_user_id"].nunique(),
            result.training_ratings_df["movie_id"].nunique(),
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_movies_csv(self) -> pd.DataFrame:
        """Load movies.csv: movieId, title, genres (pipe-separated string)."""
        path = self._dataset_dir / "movies.csv"
        df = pd.read_csv(path, dtype={"movieId": int, "title": str, "genres": str})
        logger.debug("Loaded %d rows from movies.csv", len(df))
        return df

    def _load_links_csv(self) -> pd.DataFrame:
        """Load links.csv: movieId, imdbId, tmdbId."""
        path = self._dataset_dir / "links.csv"
        df = pd.read_csv(path, dtype=str)  # read as str to handle leading zeros in imdbId
        df["movieId"] = df["movieId"].astype(int)
        logger.debug("Loaded %d rows from links.csv", len(df))
        return df

    def _load_ratings_csv(self) -> pd.DataFrame:
        """Load ratings.csv: userId, movieId, rating (float), timestamp."""
        path = self._dataset_dir / "ratings.csv"
        df = pd.read_csv(path, dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int})
        logger.debug("Loaded %d rows from ratings.csv", len(df))
        return df

    def _sync_genres(self, db: "Session", ml_movies_df: pd.DataFrame) -> int:
        """
        Upsert genre rows from MovieLens genre strings.

        MovieLens genres are pipe-separated strings (e.g. "Action|Comedy").
        We reuse the same Genre table (with tmdb_genre_id) — MovieLens genres
        that don't have a TMDB genre ID are assigned negative IDs to avoid
        collision with real TMDB genre IDs (which start at 12).

        Returns:
            Number of genre rows created.
        """
        # Known TMDB genre IDs for the genres that appear in both
        # MovieLens and TMDB. Non-overlapping ML genres get a synthetic ID.
        # This mapping is approximate — good enough for TF-IDF feature building.
        _ML_GENRE_TO_TMDB: dict[str, int] = {
            "Action": 28,
            "Adventure": 12,
            "Animation": 16,
            "Children": -1,   # No exact TMDB match
            "Comedy": 35,
            "Crime": 80,
            "Documentary": 99,
            "Drama": 18,
            "Fantasy": 14,
            "Film-Noir": -2,
            "Horror": 27,
            "IMAX": -3,
            "Musical": 10402,
            "Mystery": 9648,
            "Romance": 10749,
            "Sci-Fi": 878,
            "Thriller": 53,
            "War": 10752,
            "Western": 37,
            "(no genres listed)": -99,
        }

        # Collect all unique genre names from MovieLens
        all_genres: set[str] = set()
        for genre_str in ml_movies_df["genres"].dropna():
            all_genres.update(genre_str.split("|"))

        created = 0
        for genre_name in sorted(all_genres):
            if genre_name == "(no genres listed)":
                continue
            tmdb_genre_id = _ML_GENRE_TO_TMDB.get(genre_name, hash(genre_name) % -1000)
            existing = db.query(Genre).filter_by(tmdb_genre_id=tmdb_genre_id).first()
            if existing is None:
                db.add(Genre(tmdb_genre_id=tmdb_genre_id, name=genre_name))
                created += 1

        db.flush()
        logger.info("MovieLens genre sync: %d new genre(s) created", created)
        return created
