from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RatingBase(BaseModel):
    """Shared properties for ratings."""

    rating: int = Field(ge=1, le=10, description="Rating value from 1 to 10")


class RatingCreate(RatingBase):
    """Properties to receive on rating creation."""

    movie_id: int = Field(description="ID of the movie being rated")


class RatingUpdate(RatingBase):
    """Properties to receive on rating update."""

    pass


class RatingResponse(RatingBase):
    """Properties to return for a rating."""

    id: int
    user_id: int
    movie_id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RatingListResponse(BaseModel):
    """Paginated list of ratings."""

    ratings: list[RatingResponse]
    total: int
    limit: int
    offset: int
