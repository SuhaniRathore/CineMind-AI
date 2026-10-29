from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class ReviewBase(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(..., min_length=3, max_length=150)
    body: str = Field(..., min_length=10, max_length=5000)


class ReviewCreate(ReviewBase):
    movie_id: int


class ReviewUpdate(ReviewBase):
    pass


class ReviewResponse(ReviewBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    movie_id: int
    created_at: datetime
    updated_at: datetime

    # NLP analysis results
    sentiment_label: str | None = None
    sentiment_score: float | None = None
    aspect_sentiment: dict | None = None
    nlp_processed_at: datetime | None = None


class ReviewListResponse(BaseModel):
    reviews: list[ReviewResponse]
    total: int
    limit: int
    offset: int
