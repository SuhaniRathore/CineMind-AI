"""
Content-based recommendation engine (Milestone 8).

Uses scikit-learn TfidfVectorizer and cosine_similarity to find movies similar
to a user's highly-rated movies based on genres and metadata.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.movie import Movie
from app.models.rating import Rating


def generate_recommendations(db: Session, user_id: int, limit: int = 10) -> list[tuple[int, float]]:
    """
    Generate content-based recommendations for a user.
    Returns a list of (movie_id, relevance_score) tuples, sorted descending.
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

    # 2. Fetch all movies with their genres
    all_movies = db.scalars(select(Movie)).all()
    if len(all_movies) < 2:
        return []

    movie_id_to_idx = {}
    idx_to_movie_id = {}
    metadata_corpus = []

    for idx, movie in enumerate(all_movies):
        movie_id_to_idx[movie.id] = idx
        idx_to_movie_id[idx] = movie.id

        # Construct metadata string (genres + original_language)
        genres_str = " ".join(g.name.lower().replace(" ", "") for g in movie.genres)
        metadata = f"{genres_str} {movie.original_language or ''}".strip()
        
        # Fallback if no metadata
        if not metadata:
            metadata = "unknown"
            
        metadata_corpus.append(metadata)

    # 3. Compute TF-IDF matrix
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(metadata_corpus)
    except ValueError:
        # Fails if vocabulary is empty
        return []

    # 4. Compute similarities for preferred movies
    preferred_indices = [movie_id_to_idx[mid] for mid in preferred_movie_ids if mid in movie_id_to_idx]
    if not preferred_indices:
        return []

    # Get the vectors for preferred movies
    preferred_vectors = tfidf_matrix[preferred_indices]

    # Compute similarity of all movies to the preferred movies
    # Output shape: (num_preferred, num_all_movies)
    similarities = cosine_similarity(preferred_vectors, tfidf_matrix)

    # For each movie, take its max similarity to ANY of the preferred movies
    # This prevents recommending a movie strongly just because it's mildly similar to many things.
    max_similarities = np.max(similarities, axis=0)

    # 5. Filter and sort results
    recommendations = []
    for idx, score in enumerate(max_similarities):
        movie_id = idx_to_movie_id[idx]
        
        # Exclude movies the user has already rated (any rating)
        if movie_id in rated_movie_ids:
            continue
            
        # Ignore zero similarity
        if score <= 0.0:
            continue
            
        recommendations.append((movie_id, float(score)))

    # Sort descending by score
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N
    return recommendations[:limit]
