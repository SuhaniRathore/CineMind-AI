from datetime import datetime
from pydantic import BaseModel, ConfigDict


class WatchlistCreate(BaseModel):
    movie_id: int


class WatchlistResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    movie_id: int
    created_at: datetime


class WatchlistListResponse(BaseModel):
    items: list[WatchlistResponse]
    total: int
    limit: int
    offset: int
