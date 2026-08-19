"""
FastAPI dependencies — Milestone 5.

This module wires the pure-crypto functions from app.core.security into
FastAPI's dependency-injection system.  Route handlers declare:

    current_user: User = Depends(get_current_user)

and FastAPI will:
  1. Extract the Bearer token from the Authorization header.
  2. Decode it via security.decode_access_token.
  3. Load the User row from the database.
  4. Raise HTTP 401 on any failure (missing header, expired token, unknown user).

Architecture note (ARCHITECTURE.md):
  - This file is the only place that imports from both app.core.security and
    app.db / app.models — it is the glue layer between crypto and ORM.
  - Business logic does NOT live here; this file is concerned only with
    authentication, not authorisation (role checks belong in service layer).
"""

from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.security import decode_access_token
from app.db.database import get_db
from app.models.user import User

# HTTPBearer extracts the token from the "Authorization: Bearer <token>" header.
# auto_error=True (default) returns 403 when the header is missing — we
# override this to 401 via our own exception handling below.
_bearer = HTTPBearer(auto_error=False)

_CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials.",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> User:
    """
    FastAPI dependency — resolve the currently authenticated user.

    Extracts the JWT Bearer token from the Authorization header, verifies it,
    and returns the corresponding User ORM instance.

    Raises HTTP 401 if:
      - The Authorization header is missing or not a Bearer token.
      - The token has expired.
      - The token signature is invalid (tampered / wrong key).
      - The user_id in the token no longer exists in the database.
      - The user account is inactive (is_active=False).
    """
    if credentials is None:
        raise _CREDENTIALS_EXCEPTION

    try:
        user_id = decode_access_token(credentials.credentials, settings)
    except jwt.PyJWTError:
        raise _CREDENTIALS_EXCEPTION

    user: User | None = db.get(User, user_id)
    if user is None or not user.is_active:
        raise _CREDENTIALS_EXCEPTION

    return user
