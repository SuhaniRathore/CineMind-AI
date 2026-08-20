from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.database import get_db
from app.models.movie import Movie
from app.models.user import User
from app.models.watchlist import WatchlistItem
from app.schemas.watchlist import (
    WatchlistCreate,
    WatchlistListResponse,
    WatchlistResponse,
)

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


@router.post(
    "",
    response_model=WatchlistResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a movie to the watchlist",
)
def add_to_watchlist(
    watchlist_in: WatchlistCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> WatchlistResponse:
    """
    Add a movie to the current user's watchlist.
    A user can only add a specific movie once.
    """
    # Check if movie exists
    movie = db.get(Movie, watchlist_in.movie_id)
    if not movie:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Movie not found.",
        )

    # Create watchlist item
    item = WatchlistItem(
        user_id=current_user.id,
        movie_id=watchlist_in.movie_id,
    )
    db.add(item)
    try:
        db.commit()
        db.refresh(item)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This movie is already in your watchlist.",
        )
    return WatchlistResponse.model_validate(item)


@router.get(
    "",
    response_model=WatchlistListResponse,
    summary="List watchlist items for current user",
)
def list_watchlist(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> WatchlistListResponse:
    """
    Return a paginated list of watchlist items for the authenticated user.
    """
    query = select(WatchlistItem).where(WatchlistItem.user_id == current_user.id)

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0

    query = query.order_by(WatchlistItem.created_at.desc()).offset(offset).limit(limit)
    items = db.scalars(query).all()

    return WatchlistListResponse(
        items=[WatchlistResponse.model_validate(i) for i in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.delete(
    "/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Remove a movie from the watchlist",
)
def remove_from_watchlist(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete an existing watchlist item. Must belong to the current user.
    """
    item = db.get(WatchlistItem, item_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Watchlist item not found.",
        )

    if item.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    db.delete(item)
    db.commit()
