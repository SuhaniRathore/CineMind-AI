from datetime import datetime
from pydantic import BaseModel, ConfigDict
from app.schemas.movie import MovieOut


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    movie_id: int
    relevance_score: float
    generated_at: datetime
    # Optionally include the movie details
    movie: MovieOut | None = None


class RecommendationListResponse(BaseModel):
    recommendations: list[RecommendationResponse]
    total: int
