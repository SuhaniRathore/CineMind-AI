"""
Auth API router — Milestone 5.

Endpoints:
  POST /api/v1/auth/register  — create a new user account
  POST /api/v1/auth/login     — validate credentials, return a JWT

Architecture rules (ARCHITECTURE.md / DEVELOPMENT_RULES.md rule 9):
  - Routes are thin: validate input via Pydantic, delegate to auth_service,
    return a typed response model.  No crypto or DB logic lives here.
  - The JWT secret / algorithm is never referenced directly in route code —
    settings are injected via Depends(get_settings).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.security import create_access_token
from app.db.database import get_db
from app.schemas.user import Token, UserCreate, UserOut
from app.services import auth_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
def register(
    body: UserCreate,
    db: Session = Depends(get_db),
) -> UserOut:
    """
    Create a new user account.

    - **email**: must be a valid email address; must be unique across all accounts.
    - **password**: plain-text (min 8 chars) — hashed server-side; never stored raw.
    - **display_name**: 1–100 character screen name.

    Returns the created user record (without password) with HTTP 201.
    Returns HTTP 409 if the email is already registered.
    """
    try:
        user = auth_service.register_user(
            db,
            email=body.email,
            password=body.password,
            display_name=body.display_name,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return UserOut.model_validate(user)


@router.post(
    "/login",
    response_model=Token,
    summary="Login and receive a JWT access token",
)
def login(
    body: UserCreate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Token:
    """
    Authenticate with email and password.

    On success returns a JSON object with:
    - **access_token**: a signed JWT Bearer token.
    - **token_type**: always ``"bearer"``.
    - **user**: the authenticated user's public profile.

    Returns HTTP 401 if the credentials are invalid.

    Note: the request body is UserCreate (email + password) — the
    display_name field is present in the schema but ignored on login.
    A dedicated LoginRequest schema is a future polish item (Milestone 10).
    """
    user = auth_service.authenticate_user(
        db,
        email=body.email,
        password=body.password,
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(user.id, settings)
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserOut.model_validate(user),
    )
