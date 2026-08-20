"""
Ratings API router — Milestone 6.

Endpoints:
  POST   /api/v1/ratings             — create a new rating
  GET    /api/v1/ratings             — list ratings
  PUT    /api/v1/ratings/{id}        — update an existing rating
  DELETE /api/v1/ratings/{id}        — delete a rating
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status, Response
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.api.deps import get_current_user
from app.db.database import get_db
from app.models.movie import Movie
from app.models.rating import Rating
from app.models.user import User
from app.schemas.rating import RatingCreate, RatingListResponse, RatingResponse, RatingUpdate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ratings", tags=["ratings"])


@router.post(
    "",
    response_model=RatingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new rating",
)
def create_rating(
    rating_in: RatingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RatingResponse:
    """
    Rate a movie (1 to 10 scale).
    A user can only rate a specific movie once.
    """
    # Check if movie exists
    movie = db.get(Movie, rating_in.movie_id)
    if not movie:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Movie not found.",
        )

    # Create rating
    rating = Rating(
        user_id=current_user.id,
        movie_id=rating_in.movie_id,
        rating=rating_in.rating,
    )
    db.add(rating)
    try:
        db.commit()
        db.refresh(rating)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="You have already rated this movie.",
        )
    return RatingResponse.model_validate(rating)


@router.get("", response_model=RatingListResponse, summary="List ratings")
def list_ratings(
    user_id: int | None = Query(default=None, description="Filter by user ID"),
    movie_id: int | None = Query(default=None, description="Filter by movie ID"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> RatingListResponse:
    """
    Return a paginated list of ratings.
    """
    query = select(Rating)
    if user_id is not None:
        query = query.where(Rating.user_id == user_id)
    if movie_id is not None:
        query = query.where(Rating.movie_id == movie_id)

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0

    query = query.order_by(Rating.created_at.desc()).offset(offset).limit(limit)
    ratings = db.scalars(query).all()

    return RatingListResponse(
        ratings=[RatingResponse.model_validate(r) for r in ratings],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.put("/{rating_id}", response_model=RatingResponse, summary="Update a rating")
def update_rating(
    rating_id: int,
    rating_in: RatingUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RatingResponse:
    """
    Update an existing rating. Must belong to the current user.
    """
    rating = db.get(Rating, rating_id)
    if not rating:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rating not found.",
        )

    if rating.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    rating.rating = rating_in.rating
    db.commit()
    db.refresh(rating)
    return RatingResponse.model_validate(rating)


@router.delete(
    "/{rating_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a rating",
)
def delete_rating(
    rating_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete an existing rating. Must belong to the current user.
    """
    rating = db.get(Rating, rating_id)
    if not rating:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rating not found.",
        )

    if rating.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    db.delete(rating)
    db.commit()
