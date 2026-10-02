"""
Recommendations API router — updated at Milestone 12.

Uses HybridRecommender which:
  - Falls back to content-based if the collaborative model is not trained.
  - Uses collaborative + content blend when model is available.
  - Writes algorithm tag + reason_text to recommendation_results rows.

The collaborative model is loaded from the ModelStore at request time.
If the model artifact does not exist (not yet trained), the router falls
back silently to pure content-based recommendations — zero downtime.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import Settings, get_settings
from app.db.database import get_db
from app.ml.hybrid import HybridRecommender
from app.ml.model_store import ModelStore
from app.models.recommendation import RecommendationResult
from app.models.user import User
from app.schemas.recommendation import RecommendationListResponse, RecommendationResponse

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("", response_model=RecommendationListResponse, summary="Get personalized recommendations")
def get_recommendations(
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> RecommendationListResponse:
    """
    Get personalized movie recommendations for the authenticated user.

    Uses a hybrid of content-based (TF-IDF genre similarity) and collaborative
    filtering (SVD matrix factorisation trained on MovieLens data).

    Falls back to content-based if:
      - The collaborative model has not been trained yet (run the pipeline first).
      - The user has fewer than settings.collab_min_ratings in-app ratings.

    Recommendations are regenerated on each call and cached in the database.
    """
    # Load collaborative model (returns None if artifact not found)
    store = ModelStore(settings)
    
    collab_model = None
    if settings.use_deep_collab:
        ncf_artifact = store.load_ncf("collaborative_ncf")
        if ncf_artifact:
            from app.ml.ncf_adapter import NCFCollaborativeAdapter
            model, preprocessor = ncf_artifact
            collab_model = NCFCollaborativeAdapter(model, preprocessor)
    else:
        collab_model = store.load("collaborative_svd")

    # Build hybrid recommender
    recommender = HybridRecommender(settings, collab_model)

    # 1. Clear old cached recommendations for this user
    db.query(RecommendationResult).filter(
        RecommendationResult.user_id == current_user.id
    ).delete()
    db.commit()

    # 2. Generate new recommendations
    # Returns: list of (movie_id, score, algorithm_tag, reason_text)
    rec_tuples = recommender.recommend(db, current_user.id, limit=limit)

    # 3. Persist and prepare response
    for movie_id, score, algorithm, reason_text in rec_tuples:
        result = RecommendationResult(
            user_id=current_user.id,
            movie_id=movie_id,
            relevance_score=score,
            algorithm=algorithm,
            reason_text=reason_text,
        )
        db.add(result)

    db.commit()

    # 4. Re-fetch with relationships to build response
    db_results = db.scalars(
        select(RecommendationResult)
        .where(RecommendationResult.user_id == current_user.id)
        .order_by(RecommendationResult.relevance_score.desc())
    ).all()

    return RecommendationListResponse(
        recommendations=[
            RecommendationResponse.model_validate(r) for r in db_results
        ],
        total=len(db_results),
    )
