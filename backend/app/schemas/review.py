from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class ReviewBase(BaseModel):
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


class ReviewListResponse(BaseModel):
    reviews: list[ReviewResponse]
    total: int
    limit: int
    offset: int
