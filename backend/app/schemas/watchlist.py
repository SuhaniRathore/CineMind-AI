from datetime import datetime
from pydantic import BaseModel, ConfigDict

from app.schemas.movie import MovieOut


class WatchlistCreate(BaseModel):
    movie_id: int


class WatchlistResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    movie_id: int
    created_at: datetime
    movie: MovieOut


class WatchlistListResponse(BaseModel):
    items: list[WatchlistResponse]
    total: int
    limit: int
    offset: int
