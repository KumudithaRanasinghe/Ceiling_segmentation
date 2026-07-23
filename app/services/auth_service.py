"""
Authentication Service
========================
Orchestrates all auth flows. Combines security utilities,
repository access, and business rules.

Flows: register, login, refresh (rotation), logout, change_password
"""
from __future__ import annotations

import json
import logging

from jose import JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    verify_password,
)
from app.domain.models.user import User
from app.domain.repositories.user_repository import UserRepository
from app.domain.schemas.auth import (
    ChangePasswordRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenPair,
)

logger = logging.getLogger(__name__)

# Used only for hashing the refresh token before storing
_token_hash_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Stateless service — all state lives in the DB via the repository."""

    def __init__(self, db: Session) -> None:
        self._db = db

    # ── 1. Registration ───────────────────────────────────────────────────────

    def register(self, payload: RegisterRequest) -> tuple[User, TokenPair]:
        if UserRepository.email_exists(self._db, payload.email):
            raise ValueError(f"Email '{payload.email}' is already registered.")
        if UserRepository.username_exists(self._db, payload.username):
            raise ValueError(f"Username '{payload.username}' is already taken.")

        user = UserRepository.create(
            self._db,
            email=payload.email,
            username=payload.username,
            full_name=payload.full_name,
            plain_password=payload.password,
            roles=["user"],
        )
        logger.info("New user registered: %s (%s)", user.username, user.id)
        token_pair = self._issue_token_pair(user)
        return user, token_pair

    # ── 2. Login ──────────────────────────────────────────────────────────────

    def login(self, payload: LoginRequest) -> TokenPair:
        user = UserRepository.authenticate(
            self._db,
            identifier=payload.username_or_email,
            plain_password=payload.password,
        )
        if user is None:
            raise ValueError("Invalid username/email or password.")
        if not user.is_active:
            raise ValueError("Account is deactivated. Contact an administrator.")
        logger.info("User logged in: %s", user.username)
        return self._issue_token_pair(user)

    # ── 3. Token Refresh (single-use rotation) ────────────────────────────────

    def refresh(self, payload: RefreshRequest) -> TokenPair:
        invalid_err = "Invalid or expired refresh token."
        try:
            claims = decode_refresh_token(payload.refresh_token)
            user_id: str = claims["sub"]
        except (JWTError, KeyError):
            raise ValueError(invalid_err)

        user = UserRepository.get_by_id(self._db, user_id)
        if user is None or not user.is_active:
            raise ValueError(invalid_err)

        if not user.refresh_token_hash:
            raise ValueError(invalid_err)
        if not _token_hash_ctx.verify(payload.refresh_token, user.refresh_token_hash):
            # Possible reuse attack — revoke session entirely
            UserRepository.clear_refresh_token(self._db, user)
            logger.warning("Refresh token reuse detected for user %s — session cleared.", user.id)
            raise ValueError(invalid_err)

        logger.info("Refresh token rotated for user: %s", user.username)
        return self._issue_token_pair(user)

    # ── 4. Logout ─────────────────────────────────────────────────────────────

    def logout(self, user: User) -> None:
        UserRepository.clear_refresh_token(self._db, user)
        logger.info("User logged out: %s", user.username)

    # ── 5. Change Password ────────────────────────────────────────────────────

    def change_password(self, user: User, payload: ChangePasswordRequest) -> None:
        if not verify_password(payload.current_password, user.hashed_password):
            raise ValueError("Current password is incorrect.")
        UserRepository.update_password(self._db, user, payload.new_password)
        logger.info("Password changed for user: %s", user.username)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _issue_token_pair(self, user: User) -> TokenPair:
        """Create access + refresh tokens and persist the refresh hash."""
        roles: list[str] = json.loads(user.roles) if isinstance(user.roles, str) else user.roles
        access_token = create_access_token(
            subject=user.id,
            roles=roles,
            extra={"email": user.email, "username": user.username},
        )
        refresh_token = create_refresh_token(subject=user.id)
        refresh_hash = _token_hash_ctx.hash(refresh_token)
        UserRepository.save_refresh_token_hash(self._db, user, refresh_hash)

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )
