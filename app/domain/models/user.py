"""
User Domain Model (SQLAlchemy ORM)
====================================
Represents an application user stored in the database.

Design decisions:
  - UUID primary key prevents enumeration attacks (vs sequential int IDs).
  - Roles stored as a JSON array string (no extra join table for simplicity).
  - is_active flag allows soft-disable without deleting data.
  - refresh_token_hash stores a bcrypt hash of the most recent refresh token
    (enables single-use rotation and server-side revocation).
  - created_at / updated_at use timezone-aware UTC timestamps.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    """Persisted user entity."""

    __tablename__ = "users"

    # ── Identity ──────────────────────────────────────────────────────────────
    id: str = Column(
        String(36),
        primary_key=True,
        default=_new_uuid,
        comment="UUID v4 — prevents sequential-ID enumeration.",
    )
    email: str = Column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
        comment="Unique email address (lower-cased before storage).",
    )
    username: str = Column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )
    full_name: str | None = Column(String(255), nullable=True)

    # ── Credentials ───────────────────────────────────────────────────────────
    hashed_password: str = Column(
        String(60),   # bcrypt always produces 60-char strings
        nullable=False,
    )
    # Stores the bcrypt hash of the latest refresh token.
    # NULL means no valid refresh session exists.
    refresh_token_hash: str | None = Column(Text, nullable=True)

    # ── Access control ────────────────────────────────────────────────────────
    # JSON-encoded list e.g. '["user", "admin"]'
    roles: str = Column(
        Text,
        nullable=False,
        default='["user"]',
        comment='JSON array of role strings, e.g. ["user", "admin"]',
    )

    # ── Status flags ──────────────────────────────────────────────────────────
    is_active: bool = Column(Boolean, nullable=False, default=True)
    is_superuser: bool = Column(Boolean, nullable=False, default=False)

    # ── Audit timestamps ──────────────────────────────────────────────────────
    created_at: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )
    updated_at: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User id={self.id!r} email={self.email!r} active={self.is_active}>"
