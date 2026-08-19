"""
Authentication service — Milestone 5.

Business logic for user registration and authentication.  Route handlers
call these functions and never touch security primitives directly.

Design decisions (DEVELOPMENT_RULES.md rule 5):
  - email uniqueness is enforced at the DB level (UniqueConstraint on the
    'users' table, added in Milestone 3).  The service catches the
    IntegrityError that results from a duplicate email and raises a clean
    ValueError so the route layer can translate it to HTTP 409.
  - Passwords are validated for minimum length before hashing to avoid
    wasting bcrypt CPU on obviously-bad inputs.
  - No email-verification step at this milestone — that is a V2 concern.

Functions:
  register_user(db, email, password, display_name) -> User
  authenticate_user(db, email, password) -> User | None
"""

from __future__ import annotations

import logging

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.security import hash_password, verify_password
from app.models.user import User

logger = logging.getLogger(__name__)

_MIN_PASSWORD_LENGTH = 8


def register_user(
    db: Session,
    *,
    email: str,
    password: str,
    display_name: str,
) -> User:
    """
    Create and persist a new user account.

    Args:
        db:            Active SQLAlchemy session.
        email:         Unique email address (validated by Pydantic upstream).
        password:      Plain-text password — hashed before storage; never stored.
        display_name:  Human-readable screen name.

    Returns:
        The newly created User ORM instance (committed, with id populated).

    Raises:
        ValueError: if the email is already registered.
        ValueError: if password is shorter than _MIN_PASSWORD_LENGTH.
    """
    if len(password) < _MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"Password must be at least {_MIN_PASSWORD_LENGTH} characters long."
        )

    user = User(
        email=email.lower().strip(),
        hashed_password=hash_password(password),
        display_name=display_name,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        logger.debug("register_user: duplicate email %r", email)
        raise ValueError(f"Email '{email}' is already registered.")

    db.refresh(user)
    logger.info("register_user: created user id=%d email=%r", user.id, user.email)
    return user


def authenticate_user(
    db: Session,
    *,
    email: str,
    password: str,
) -> User | None:
    """
    Verify credentials and return the User if valid, or None if not.

    Returning None (rather than raising) keeps the route layer in control of
    the HTTP status code — the caller converts None to HTTP 401.

    The negative path uses a dummy hash comparison so the response time is
    roughly equal whether the email exists or not (timing-oracle mitigation).
    """
    user: User | None = (
        db.query(User).filter(User.email == email.lower().strip()).first()
    )

    if user is None:
        # Constant-time dummy comparison to prevent email enumeration via timing.
        _dummy = hash_password("dummy-timing-pad")
        verify_password("dummy-timing-pad", _dummy)
        return None

    if not verify_password(password, user.hashed_password):
        return None

    if not user.is_active:
        return None

    return user
