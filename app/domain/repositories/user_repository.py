"""
User Repository
================
Pure data-access layer for the User entity.
Provides secure database reads, writes, and admin IAM queries.
"""
from __future__ import annotations

import json
import uuid
from typing import List, Optional, Tuple

from sqlalchemy import desc, func, or_
from sqlalchemy.orm import Session

from app.core.security import hash_password, verify_password
from app.domain.models.user import User


class UserRepository:
    """Data-access methods for User entities."""

    # ── Reads ─────────────────────────────────────────────────────────────────

    @staticmethod
    def get_by_id(db: Session, user_id: str) -> Optional[User]:
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email.lower()).first()

    @staticmethod
    def get_by_username(db: Session, username: str) -> Optional[User]:
        return db.query(User).filter(User.username == username.lower()).first()

    @staticmethod
    def get_by_username_or_email(db: Session, identifier: str) -> Optional[User]:
        user = UserRepository.get_by_username(db, identifier)
        if user is None:
            user = UserRepository.get_by_email(db, identifier)
        return user

    @staticmethod
    def email_exists(db: Session, email: str) -> bool:
        return UserRepository.get_by_email(db, email) is not None

    @staticmethod
    def username_exists(db: Session, username: str) -> bool:
        return UserRepository.get_by_username(db, username) is not None

    @staticmethod
    def count_total(db: Session) -> int:
        return db.query(func.count(User.id)).scalar() or 0

    @staticmethod
    def list_users_admin(
        db: Session,
        page: int = 1,
        page_size: int = 20,
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
        role: Optional[str] = None,
    ) -> Tuple[List[User], int]:
        """Admin user directory query with search, filter, and pagination."""
        query = db.query(User)

        if search:
            search_pattern = f"%{search.lower()}%"
            query = query.filter(
                or_(
                    User.email.ilike(search_pattern),
                    User.username.ilike(search_pattern),
                    User.full_name.ilike(search_pattern),
                )
            )
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        if role:
            query = query.filter(User.roles.like(f'%"{role}"%'))

        total = query.count()
        offset = (page - 1) * page_size
        users = query.order_by(desc(User.created_at)).offset(offset).limit(page_size).all()
        return users, total

    # ── Writes ────────────────────────────────────────────────────────────────

    @staticmethod
    def create(
        db: Session,
        *,
        email: str,
        username: str,
        full_name: Optional[str],
        plain_password: str,
        roles: list[str] | None = None,
        is_superuser: bool = False,
    ) -> User:
        user = User(
            id=str(uuid.uuid4()),
            email=email.lower(),
            username=username.lower(),
            full_name=full_name,
            hashed_password=hash_password(plain_password),
            roles=json.dumps(roles or ["user"]),
            is_superuser=is_superuser,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def save_refresh_token_hash(db: Session, user: User, refresh_token_hash: str) -> None:
        user.refresh_token_hash = refresh_token_hash
        db.commit()

    @staticmethod
    def clear_refresh_token(db: Session, user: User) -> None:
        user.refresh_token_hash = None
        db.commit()

    @staticmethod
    def update_password(db: Session, user: User, new_plain_password: str) -> None:
        user.hashed_password = hash_password(new_plain_password)
        user.refresh_token_hash = None
        db.commit()

    @staticmethod
    def update_roles(db: Session, user: User, roles: list[str]) -> None:
        user.roles = json.dumps(roles)
        db.commit()

    @staticmethod
    def set_active_status(db: Session, user: User, is_active: bool) -> None:
        user.is_active = is_active
        if not is_active:
            user.refresh_token_hash = None
        db.commit()

    @staticmethod
    def deactivate(db: Session, user: User) -> None:
        UserRepository.set_active_status(db, user, is_active=False)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def authenticate(db: Session, identifier: str, plain_password: str) -> Optional[User]:
        """
        Verify credentials. Returns User on success, None on failure.
        Always performs constant-time comparison to prevent timing attacks.
        """
        user = UserRepository.get_by_username_or_email(db, identifier)
        if user is None:
            hash_password("dummy_constant_time_guard")
            return None
        if not verify_password(plain_password, user.hashed_password):
            return None
        return user
