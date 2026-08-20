from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.database import get_db
from app.models.movie import Movie
from app.models.review import Review
from app.models.user import User
from app.schemas.review import (
    ReviewCreate,
    ReviewListResponse,
    ReviewResponse,
    ReviewUpdate,
)

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.post(
    "",
    response_model=ReviewResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new review",
)
def create_review(
    review_in: ReviewCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ReviewResponse:
    """
    Write a review for a movie.
    A user can only write one review per movie.
    """
    # Check if movie exists
    movie = db.get(Movie, review_in.movie_id)
    if not movie:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Movie not found.",
        )

    # Create review
    review = Review(
        user_id=current_user.id,
        movie_id=review_in.movie_id,
        title=review_in.title,
        body=review_in.body,
    )
    db.add(review)
    try:
        db.commit()
        db.refresh(review)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="You have already reviewed this movie.",
        )
    return ReviewResponse.model_validate(review)


@router.get("", response_model=ReviewListResponse, summary="List reviews")
def list_reviews(
    user_id: int | None = Query(default=None, description="Filter by user ID"),
    movie_id: int | None = Query(default=None, description="Filter by movie ID"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> ReviewListResponse:
    """
    Return a paginated list of reviews.
    """
    query = select(Review)
    if user_id is not None:
        query = query.where(Review.user_id == user_id)
    if movie_id is not None:
        query = query.where(Review.movie_id == movie_id)

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0

    query = query.order_by(Review.created_at.desc()).offset(offset).limit(limit)
    reviews = db.scalars(query).all()

    return ReviewListResponse(
        reviews=[ReviewResponse.model_validate(r) for r in reviews],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.put("/{review_id}", response_model=ReviewResponse, summary="Update a review")
def update_review(
    review_id: int,
    review_in: ReviewUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ReviewResponse:
    """
    Update an existing review. Must belong to the current user.
    """
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found.",
        )

    if review.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    review.title = review_in.title
    review.body = review_in.body
    db.commit()
    db.refresh(review)
    return ReviewResponse.model_validate(review)


@router.delete(
    "/{review_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a review",
)
def delete_review(
    review_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete an existing review. Must belong to the current user.
    """
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found.",
        )

    if review.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    db.delete(review)
    db.commit()
