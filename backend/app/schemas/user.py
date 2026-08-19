"""
Pydantic schemas for the auth and user endpoints (Milestone 5).

Schemas follow the same conventions as app/schemas/movie.py:
  - All response schemas use ConfigDict(from_attributes=True) so they can be
    constructed from SQLAlchemy ORM instances.
  - Request schemas (UserCreate) are plain BaseModel — no ORM mapping needed.
  - Sensitive fields (hashed_password) are never included in response schemas.

Schemas defined here:
  UserCreate   — body for POST /auth/register
  UserOut      — response for register, login (user portion), GET /users/me
  Token        — response for POST /auth/login  (access_token + token_type)
  TokenData    — internal: the claims decoded from a JWT (not returned to client)
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    """Request body for POST /auth/register."""

    email: EmailStr
    password: str = Field(min_length=8, description="Plain-text password (min 8 chars)")
    display_name: str = Field(min_length=1, max_length=100)


class UserOut(BaseModel):
    """User record as returned by the API — never includes hashed_password."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    display_name: str
    is_active: bool
    created_at: datetime


class Token(BaseModel):
    """Response body for POST /auth/login."""

    access_token: str
    token_type: str = "bearer"
    user: UserOut


class TokenData(BaseModel):
    """Claims decoded from a JWT — used internally by get_current_user dep."""

    user_id: int
