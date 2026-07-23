"""
Core Security Utilities
========================
Responsibilities:
  - Password hashing (bcrypt via passlib)
  - JWT access token creation & verification
  - JWT refresh token creation & verification
  - TokenType enum for internal safety checks
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

# ── Password hashing ──────────────────────────────────────────────────────────
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time comparison between *plain* and *hashed*."""
    return _pwd_context.verify(plain, hashed)


# ── Token types ───────────────────────────────────────────────────────────────
class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"


# ── JWT helpers ───────────────────────────────────────────────────────────────
def _build_token(
    subject: str,
    token_type: TokenType,
    extra_claims: dict[str, Any] | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    now = datetime.now(tz=timezone.utc)
    if expires_delta is None:
        if token_type == TokenType.ACCESS:
            expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        else:
            expires_delta = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    payload: dict[str, Any] = {
        "sub": subject,
        "type": token_type.value,
        "iat": now,
        "exp": now + expires_delta,
        "jti": secrets.token_hex(16),
    }
    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_access_token(
    subject: str,
    roles: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    claims: dict[str, Any] = {}
    if roles:
        claims["roles"] = roles
    if extra:
        claims.update(extra)
    return _build_token(subject, TokenType.ACCESS, claims)


def create_refresh_token(subject: str) -> str:
    return _build_token(subject, TokenType.REFRESH)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and verify an access token. Raises JWTError on failure."""
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    if payload.get("type") != TokenType.ACCESS.value:
        raise JWTError("Token type mismatch — expected access token.")
    return payload


def decode_refresh_token(token: str) -> dict[str, Any]:
    """Decode and verify a refresh token. Raises JWTError on failure."""
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    if payload.get("type") != TokenType.REFRESH.value:
        raise JWTError("Token type mismatch — expected refresh token.")
    return payload
