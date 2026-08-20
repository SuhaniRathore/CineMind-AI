from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.database import get_db
from app.ml.recommender import generate_recommendations
from app.models.movie import Movie
from app.models.recommendation import RecommendationResult
from app.models.user import User
from app.schemas.recommendation import RecommendationListResponse, RecommendationResponse
from app.schemas.movie import MovieOut

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("", response_model=RecommendationListResponse, summary="Get personalized recommendations")
def get_recommendations(
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RecommendationListResponse:
    """
    Get personalized content-based movie recommendations for the authenticated user.
    Recommendations are generated on-the-fly and cached in the database.
    """
    # 1. Clear old cached recommendations for this user
    db.query(RecommendationResult).filter(RecommendationResult.user_id == current_user.id).delete()
    db.commit()

    # 2. Generate new recommendations
    rec_tuples = generate_recommendations(db, current_user.id, limit=limit)

    # 3. Persist and prepare response
    results = []
    for movie_id, score in rec_tuples:
        result = RecommendationResult(
            user_id=current_user.id,
            movie_id=movie_id,
            relevance_score=score,
        )
        db.add(result)
        results.append(result)

    db.commit()

    # Need to load movies to return in response
    # Re-fetch with relationships or just return the IDs. The schema supports returning Movie.
    # We will fetch the actual DB objects to return full representations.
    db_results = db.scalars(
        select(RecommendationResult)
        .where(RecommendationResult.user_id == current_user.id)
        .order_by(RecommendationResult.relevance_score.desc())
    ).all()

    # Return
    return RecommendationListResponse(
        recommendations=[
            RecommendationResponse.model_validate(r) for r in db_results
        ],
        total=len(db_results),
    )
