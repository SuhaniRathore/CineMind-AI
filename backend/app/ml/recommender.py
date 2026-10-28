"""
Content-based recommendation engine (Milestone 8, fixed at Milestone 12).

Uses scikit-learn TfidfVectorizer and cosine_similarity to find movies
similar to a user's highly-rated movies based on genres and metadata.

Cold-start fallback:
  - Users with no ratings receive popular-movie recommendations.
  - Users without highly-rated movies receive popular unrated movies.
  - If content-based recommendations cannot be generated, the system falls
    back to available popular movies.

Performance:
  - Loads preferred movies and unrated candidates separately.
  - Uses selectinload to batch-load movie genres and avoid N+1 queries.
  - Uses SQLAlchemy yield_per to process candidates in batches.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.movie import Movie
from app.models.recommendation import (
    ALGORITHM_CONTENT_BASED,
    ALGORITHM_POPULARITY_BASED,
)
from app.models.rating import Rating


def _popular_fallback(
    db: Session,
    limit: int,
    exclude_ids: set[int] | None = None,
) -> list[tuple[int, float]]:
    """
    Return popular movies when user-rating signals are unavailable.

    Only movies with popularity metadata are eligible. Already-rated movies
    can be excluded for users who have provided ratings.
    """
    query = select(Movie).where(Movie.popularity.is_not(None))

    if exclude_ids:
        query = query.where(Movie.id.notin_(exclude_ids))

    movies = db.scalars(
        query.order_by(
            Movie.popularity.desc(),
            Movie.vote_count.desc().nullslast(),
            Movie.vote_average.desc().nullslast(),
        ).limit(limit)
    ).all()

    if not movies:
        return []

    max_popularity = max(
        float(movie.popularity or 0.0)
        for movie in movies
    )
    denominator = max(max_popularity, 1.0)

    return [
        (
            movie.id,
            max(0.0, float(movie.popularity or 0.0)) / denominator,
        )
        for movie in movies
    ]


def generate_recommendations(
    db: Session,
    user_id: int,
    limit: int = 10,
) -> list[tuple[int, float]]:
    """Return recommendation IDs and scores, preserving the original API."""
    return [
        (movie_id, score)
        for movie_id, score, _ in generate_recommendations_with_source(
            db,
            user_id,
            limit,
        )
    ]


def generate_recommendations_with_source(
    db: Session,
    user_id: int,
    limit: int = 10,
) -> list[tuple[int, float, str]]:
    """
    Generate recommendations and retain whether popularity fallback was used.

    Returns (movie_id, relevance_score, algorithm) tuples, sorted descending.

    Users without useful rating signals receive popular-movie recommendations
    instead of an empty result, provided popularity metadata is available.
    """

    def tagged(
        recommendations: list[tuple[int, float]],
        algorithm: str,
    ) -> list[tuple[int, float, str]]:
        return [
            (movie_id, score, algorithm)
            for movie_id, score in recommendations
        ]

    # 1. Fetch the user's ratings.
    user_ratings = db.scalars(
        select(Rating).where(Rating.user_id == user_id)
    ).all()

    # Cold start: no ratings available.
    if not user_ratings:
        return tagged(
            _popular_fallback(db, limit),
            ALGORITHM_POPULARITY_BASED,
        )

    rated_movie_ids = {
        rating.movie_id
        for rating in user_ratings
    }

    # Consider highly rated movies (rating >= 7) as positive signals.
    highly_rated = [
        rating
        for rating in user_ratings
        if rating.rating >= 7
    ]

    # Cold start: ratings exist, but none is highly rated.
    if not highly_rated:
        return tagged(
            _popular_fallback(
                db,
                limit,
                exclude_ids=rated_movie_ids,
            ),
            ALGORITHM_POPULARITY_BASED,
        )

    preferred_movie_ids = {
        rating.movie_id
        for rating in highly_rated
    }

    # 2. Load preferred movies and their genres in batches.
    preferred_movies = db.scalars(
        select(Movie)
        .options(selectinload(Movie.genres))
        .where(Movie.id.in_(preferred_movie_ids))
    ).all()

    # Defensive fallback if rated movie records are unavailable.
    if not preferred_movies:
        return tagged(
            _popular_fallback(
                db,
                limit,
                exclude_ids=rated_movie_ids,
            ),
            ALGORITHM_POPULARITY_BASED,
        )

    # Candidate movies: those the user has not rated.
    # selectinload batches genre loading instead of querying genres per movie.
    candidate_query = (
        select(Movie)
        .options(selectinload(Movie.genres))
        .where(Movie.id.notin_(rated_movie_ids))
        .execution_options(yield_per=1000)
    )

    n_preferred = len(preferred_movies)
    metadata_corpus: list[str] = []
    candidate_movie_ids_list: list[int] = []

    # 3. Process preferred movies.
    for movie in preferred_movies:
        genres_str = " ".join(
            genre.name.lower().replace(" ", "")
            for genre in movie.genres
        )
        metadata = (
            f"{genres_str} {movie.original_language or ''}"
        ).strip()

        metadata_corpus.append(metadata or "unknown")

    # Process candidate movies.
    for movie in db.scalars(candidate_query):
        genres_str = " ".join(
            genre.name.lower().replace(" ", "")
            for genre in movie.genres
        )
        metadata = (
            f"{genres_str} {movie.original_language or ''}"
        ).strip()

        metadata_corpus.append(metadata or "unknown")
        candidate_movie_ids_list.append(movie.id)

    # No unrated candidates available.
    if not candidate_movie_ids_list:
        return tagged(
            _popular_fallback(
                db,
                limit,
                exclude_ids=rated_movie_ids,
            ),
            ALGORITHM_POPULARITY_BASED,
        )

    # 4. Compute the TF-IDF matrix.
    vectorizer = TfidfVectorizer()

    try:
        tfidf_matrix = vectorizer.fit_transform(metadata_corpus)
    except ValueError:
        return tagged(
            _popular_fallback(
                db,
                limit,
                exclude_ids=rated_movie_ids,
            ),
            ALGORITHM_POPULARITY_BASED,
        )

    # 5. Compute similarities between preferred movies and candidates.
    preferred_vectors = tfidf_matrix[:n_preferred]
    candidate_vectors = tfidf_matrix[n_preferred:]

    similarities = cosine_similarity(
        preferred_vectors,
        candidate_vectors,
    )

    # Maximum similarity to any preferred movie.
    max_similarities = np.max(
        similarities,
        axis=0,
    )

    # 6. Filter and sort recommendations.
    recommendations: list[tuple[int, float]] = []

    for index, score in enumerate(max_similarities):
        if score <= 0.0:
            continue

        recommendations.append(
            (
                candidate_movie_ids_list[index],
                float(score),
            )
        )

    recommendations.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    # Use popular movies when content metadata produces no matches.
    if not recommendations:
        return tagged(
            _popular_fallback(
                db,
                limit,
                exclude_ids=rated_movie_ids,
            ),
            ALGORITHM_POPULARITY_BASED,
        )

    return tagged(
        recommendations[:limit],
        ALGORITHM_CONTENT_BASED,
    )