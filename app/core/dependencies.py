"""
FastAPI Dependencies — Authentication & Authorization
======================================================
Re-usable injectable dependencies that protect endpoints.

WHY HTTPBearer instead of OAuth2PasswordBearer?
------------------------------------------------
HTTPBearer makes Swagger show a simple "paste your token" text box.
Users call POST /auth/login to get their access_token, then paste it
directly into the Swagger Authorize dialog — no redundant credential
entry required.

Usage patterns:
  - current_user              → any authenticated, active user
  - require_roles(["admin"])  → only users with the specified roles
  - get_optional_current_user → public endpoints that are role-aware
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.domain.models.user import User
from app.domain.repositories.user_repository import UserRepository
from app.infrastructure.database.session import get_db

logger = logging.getLogger(__name__)

# Swagger "Authorize" shows: Value: [_____________]  ← paste access_token here
_bearer_scheme = HTTPBearer(
    scheme_name="Bearer Token",
    description=(
        "Paste the **access_token** you received from `POST /auth/login`. "
        "Do **not** include the 'Bearer' prefix — Swagger adds it automatically."
    ),
    auto_error=True,
)
_bearer_scheme_optional = HTTPBearer(auto_error=False)


# ── Core resolver ─────────────────────────────────────────────────────────────

def _get_user_from_token(token: str, db: Session, *, require_active: bool = True) -> User:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise credentials_exc
    except JWTError as exc:
        logger.warning("JWT decode failed: %s", exc)
        raise credentials_exc

    user = UserRepository.get_by_id(db, user_id)
    if user is None:
        raise credentials_exc

    if require_active and not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact an administrator.",
        )
    return user


# ── Public dependencies ───────────────────────────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Require a valid Bearer access token.
    Extracts token from: Authorization: Bearer <token>
    """
    return _get_user_from_token(credentials.credentials, db)


def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme_optional),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Returns None (instead of raising 401) when no token is provided."""
    if credentials is None:
        return None
    try:
        return _get_user_from_token(credentials.credentials, db)
    except HTTPException:
        return None


# ── Role-based access control (RBAC) ─────────────────────────────────────────

def require_roles(allowed_roles: List[str]):
    """
    Dependency factory for role-based access control.

    Usage:
        @router.delete("/x", dependencies=[Depends(require_roles(["admin"]))])
        def handler(user: User = Depends(require_roles(["admin", "moderator"]))):
    """
    def _checker(user: User = Depends(get_current_user)) -> User:
        user_roles: list[str] = (
            json.loads(user.roles) if isinstance(user.roles, str) else user.roles
        )
        if user.is_superuser:
            return user
        if not any(r in user_roles for r in allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Access denied. Required roles: {allowed_roles}. "
                    f"Your roles: {user_roles}."
                ),
            )
        return user
    return _checker


# ── Convenience role shortcuts ────────────────────────────────────────────────

def require_admin(user: User = Depends(require_roles(["admin"]))) -> User:
    """Require the 'admin' role (or superuser flag)."""
    return user


def require_verified_user(user: User = Depends(get_current_user)) -> User:
    """Placeholder for email-verification check."""
    return user
