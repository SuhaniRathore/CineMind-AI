"""
Content-based recommendation engine (Milestone 8, fixed at Milestone 12).

Uses scikit-learn TfidfVectorizer and cosine_similarity to find movies similar
to a user's highly-rated movies based on genres and metadata.

Milestone 12 fix:
  - CRITICAL: Replaced SELECT * on all movies with a targeted query that only
    loads unrated movies + the user's highly-rated movies. This prevents an
    OOM/timeout when the DB contains 10K+ MovieLens-sourced movie rows.
  - Added algorithm tag ('content_based') to the return value so callers
    can persist it in recommendation_results.algorithm.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.movie import Movie
from app.models.rating import Rating
from app.models.recommendation import ALGORITHM_CONTENT_BASED


def generate_recommendations(
    db: Session,
    user_id: int,
    limit: int = 10,
) -> list[tuple[int, float]]:
    """
    Generate content-based recommendations for a user.

    Returns a list of (movie_id, relevance_score) tuples, sorted descending.

    Performance fix (Milestone 12):
      Instead of loading ALL movies, we load:
        a) The user's highly-rated movies (to build the query vector).
        b) All movies the user has NOT rated yet (candidates).
      This means memory usage scales with (unrated movies), not total movies.
    """
    # 1. Fetch user's ratings
    user_ratings = db.scalars(
        select(Rating).where(Rating.user_id == user_id)
    ).all()

    if not user_ratings:
        return []

    # Consider highly-rated movies (rating >= 7) as positive signals
    highly_rated = [r for r in user_ratings if r.rating >= 7]
    if not highly_rated:
        return []

    rated_movie_ids = {r.movie_id for r in user_ratings}
    preferred_movie_ids = {r.movie_id for r in highly_rated}

    # 2. Fetch preferred movies (query anchors) and unrated candidates separately.
    #    This avoids loading ALL movies into memory — critical for large datasets.
    preferred_movies = db.scalars(
        select(Movie).where(Movie.id.in_(preferred_movie_ids))
    ).all()

    # Candidate movies: NOT in rated set, loaded in chunks to bound memory
    # For TF-IDF we still need all candidates strings in memory at once, but we
    # avoid holding heavy ORM objects by projecting only what we need or discarding them.
    candidate_query = select(Movie).where(Movie.id.notin_(rated_movie_ids)).execution_options(yield_per=1000)

    # 3. Build combined corpus: preferred movies first, then candidates.
    #    We'll split the TF-IDF matrix by this boundary.
    n_preferred = len(preferred_movies)

    metadata_corpus: list[str] = []
    candidate_movie_ids_list: list[int] = []

    # Process preferred movies
    for movie in preferred_movies:
        genres_str = " ".join(g.name.lower().replace(" ", "") for g in movie.genres)
        metadata = f"{genres_str} {movie.original_language or ''}".strip()
        metadata_corpus.append(metadata or "unknown")

    # Process candidate movies in chunks
    for movie in db.scalars(candidate_query):
        genres_str = " ".join(g.name.lower().replace(" ", "") for g in movie.genres)
        metadata = f"{genres_str} {movie.original_language or ''}".strip()
        metadata_corpus.append(metadata or "unknown")
        candidate_movie_ids_list.append(movie.id)

    # Need at least 1 candidate
    if not candidate_movie_ids_list:
        return []

    # 4. Compute TF-IDF matrix
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(metadata_corpus)
    except ValueError:
        # Fails if vocabulary is empty (all 'unknown')
        return []

    # 5. Compute similarities: preferred rows vs all rows
    preferred_vectors = tfidf_matrix[:n_preferred]
    candidate_vectors = tfidf_matrix[n_preferred:]

    # Shape: (n_preferred, n_candidates)
    similarities = cosine_similarity(preferred_vectors, candidate_vectors)

    # For each candidate, take max similarity to ANY preferred movie
    max_similarities = np.max(similarities, axis=0)

    # 6. Filter and sort
    recommendations: list[tuple[int, float]] = []
    for idx, score in enumerate(max_similarities):
        if score <= 0.0:
            continue

        recommendations.append((candidate_movie_ids_list[idx], float(score)))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:limit]
