"""
Users API router — Milestone 5.

Endpoints:
  GET /api/v1/users/me — return the currently authenticated user's profile

This router serves as the canonical example of a protected route:
the Depends(get_current_user) dependency enforces JWT authentication
and handles all 401 cases before the route handler is ever called.

Future milestones will add more user-scoped endpoints here
(e.g., PATCH /users/me for profile updates, added in Milestone 10 hardening).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.user import UserOut

router = APIRouter(prefix="/users", tags=["users"])


@router.get(
    "/me",
    response_model=UserOut,
    summary="Get current user profile",
)
def get_me(current_user: User = Depends(get_current_user)) -> UserOut:
    """
    Return the authenticated user's public profile.

    Requires a valid JWT Bearer token in the Authorization header.
    Returns HTTP 401 if the token is missing, expired, or invalid.
    """
    return UserOut.model_validate(current_user)
