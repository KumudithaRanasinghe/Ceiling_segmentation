"""
Auth API Endpoints
===================
All authentication-related routes live here.

Endpoints:
  POST /auth/register         — create a new account
  POST /auth/login            — JSON login → returns access_token + refresh_token
  POST /auth/refresh          — rotate refresh token
  POST /auth/logout           — revoke refresh session
  GET  /auth/me               — return current user profile
  PUT  /auth/me/password      — change password
  GET  /auth/users            — list users (admin only)
  DELETE /auth/users/{id}     — deactivate a user (admin only)

Swagger authentication:
  1. Call POST /auth/login with your credentials.
  2. Copy the access_token from the response.
  3. Click the 🔒 Authorize button in Swagger UI.
  4. Paste the token in the "Value" field — done.
     (Do NOT include the "Bearer " prefix — Swagger adds it.)

Security:
  - All state-mutating endpoints require a valid access token.
  - Rate limiting is applied to this entire router by middleware.
  - Errors are returned with generic messages to avoid information leakage.
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, require_admin, require_roles
from app.domain.models.user import User
from app.domain.repositories.user_repository import UserRepository
from app.domain.schemas.auth import (
    ChangePasswordRequest,
    LoginRequest,
    MessageResponse,
    RefreshRequest,
    RegisterRequest,
    TokenPair,
    UserResponse,
)
from app.infrastructure.database.session import get_db
from app.services.auth_service import AuthService

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helper: convert ValueError to 400 ────────────────────────────────────────

def _bad_request(exc: ValueError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(exc),
    )


# ── POST /register ────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    responses={
        400: {"description": "Email or username already taken, or invalid input."},
    },
)
def register(
    payload: RegisterRequest,
    db: Session = Depends(get_db),
) -> UserResponse:
    """
    Create a new account.

    - Password must be ≥ 8 chars with upper, lower, digit, and special char.
    - Returns the created user profile (no tokens — require explicit login).
    """
    svc = AuthService(db)
    try:
        user, _ = svc.register(payload)
    except ValueError as exc:
        raise _bad_request(exc)
    return UserResponse.model_validate(user)


# ── POST /login (JSON) ────────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=TokenPair,
    summary="Login with JSON body (username/email + password)",
    responses={
        400: {"description": "Invalid credentials."},
    },
)
def login(
    payload: LoginRequest,
    db: Session = Depends(get_db),
) -> TokenPair:
    """
    Authenticate and receive an access + refresh token pair.

    Use the `access_token` in subsequent requests:
    ```
    Authorization: Bearer <access_token>
    ```
    """
    svc = AuthService(db)
    try:
        return svc.login(payload)
    except ValueError as exc:
        raise _bad_request(exc)


# ── POST /refresh ─────────────────────────────────────────────────────────────

@router.post(
    "/refresh",
    response_model=TokenPair,
    summary="Rotate refresh token and get a new token pair",
    responses={
        400: {"description": "Invalid, expired, or already-used refresh token."},
    },
)
def refresh_token(
    payload: RefreshRequest,
    db: Session = Depends(get_db),
) -> TokenPair:
    """
    Exchange a valid refresh token for a **new** access + refresh token pair.

    The old refresh token is immediately invalidated (single-use rotation).
    If you attempt to reuse the same refresh token, the session is revoked entirely.
    """
    svc = AuthService(db)
    try:
        return svc.refresh(payload)
    except ValueError as exc:
        raise _bad_request(exc)


# ── POST /logout ──────────────────────────────────────────────────────────────

@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout (revoke refresh token session)",
)
def logout(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageResponse:
    """
    Revoke the current refresh session.

    Note: access tokens remain valid until they naturally expire (JWT is stateless).
    For immediate access token revocation, add a token blocklist (e.g. in Redis).
    """
    AuthService(db).logout(current_user)
    return MessageResponse(message="Successfully logged out.")


# ── GET /me ───────────────────────────────────────────────────────────────────

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
def get_me(current_user: User = Depends(get_current_user)) -> UserResponse:
    """Return the profile of the currently authenticated user."""
    return UserResponse.model_validate(current_user)


# ── PUT /me/password ──────────────────────────────────────────────────────────

@router.put(
    "/me/password",
    response_model=MessageResponse,
    summary="Change your password",
    responses={
        400: {"description": "Current password is incorrect."},
    },
)
def change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageResponse:
    """
    Change account password.

    - Verifies the current password before accepting the new one.
    - Revokes all active refresh sessions (forces re-login on all devices).
    """
    svc = AuthService(db)
    try:
        svc.change_password(current_user, payload)
    except ValueError as exc:
        raise _bad_request(exc)
    return MessageResponse(message="Password updated successfully. Please log in again.")


# ── GET /users (admin) ────────────────────────────────────────────────────────

@router.get(
    "/users",
    response_model=List[UserResponse],
    summary="List all users [admin only]",
)
def list_users(
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> List[UserResponse]:
    """Return all registered users. Requires the `admin` role."""
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [UserResponse.model_validate(u) for u in users]


# ── DELETE /users/{user_id} (admin) ──────────────────────────────────────────

@router.delete(
    "/users/{user_id}",
    response_model=MessageResponse,
    summary="Deactivate a user account [admin only]",
    responses={
        404: {"description": "User not found."},
        400: {"description": "Cannot deactivate a superuser."},
    },
)
def deactivate_user(
    user_id: str,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> MessageResponse:
    """
    Soft-deactivate a user (sets is_active=False, clears refresh tokens).
    The user record is NOT deleted — preserves audit history.
    """
    user = UserRepository.get_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    if user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate a superuser via API.",
        )
    UserRepository.deactivate(db, user)
    return MessageResponse(message=f"User '{user.username}' has been deactivated.")
