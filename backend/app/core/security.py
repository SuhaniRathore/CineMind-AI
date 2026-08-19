"""
Pure-crypto security utilities — Milestone 5.

This module contains only cryptographic primitives and is intentionally
kept free of FastAPI, SQLAlchemy, or business-logic imports.

Design decisions (DEVELOPMENT_RULES.md rule 5):
  - bcrypt is used directly (no passlib wrapper) — passlib is unmaintained
    since 2022; raw bcrypt is simpler and sufficient for HS256 + bcrypt.
  - PyJWT is used for JWT creation/verification.
  - JWT payload carries only the minimum required claim: sub = str(user_id).
    No email or role is embedded to reduce the blast radius of a leaked token.
  - Token expiry is read from settings so it can be shortened in production
    without touching code.
  - Exceptions are NOT caught here — callers (deps.py) map them to HTTP 401.

Functions exposed:
  hash_password(plain: str) -> str
  verify_password(plain: str, hashed: str) -> bool
  create_access_token(user_id: int, settings: Settings) -> str
  decode_access_token(token: str, settings: Settings) -> int   # returns user_id
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.core.config import Settings


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def hash_password(plain: str) -> str:
    """
    Return the bcrypt hash of *plain* as a UTF-8 string.

    bcrypt.hashpw expects bytes; we encode and decode around the call so
    the rest of the codebase deals only with plain strings.
    """
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(plain.encode("utf-8"), salt)
    return hashed_bytes.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """
    Return True if *plain* matches the stored bcrypt *hashed* value.

    bcrypt.checkpw handles the constant-time comparison internally, so there
    is no timing-attack risk from this call.
    """
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------


def create_access_token(user_id: int, settings: Settings) -> str:
    """
    Create a signed JWT access token for *user_id*.

    Claims:
      sub  — str(user_id)   (standard JWT subject claim)
      exp  — UTC expiry timestamp (settings.access_token_expire_minutes from now)
      iat  — UTC issued-at timestamp

    Returns the encoded token as a string.
    """
    now = datetime.now(tz=timezone.utc)
    expire = now + timedelta(minutes=settings.access_token_expire_minutes)
    payload: dict = {
        "sub": str(user_id),
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: Settings) -> int:
    """
    Decode and verify *token*, returning the user_id embedded in the ``sub`` claim.

    Raises:
      jwt.ExpiredSignatureError  — token has expired
      jwt.InvalidTokenError      — token is malformed, tampered, or uses wrong key

    Both exception types are subclasses of jwt.PyJWTError — callers in
    deps.py catch PyJWTError and raise HTTP 401.
    """
    payload = jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )
    # sub is always stored as a string; convert back to int for typed callers
    sub: str = payload["sub"]
    return int(sub)
