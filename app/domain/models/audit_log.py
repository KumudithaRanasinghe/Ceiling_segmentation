"""
AuditLog Domain Model (SQLAlchemy ORM)
======================================
Stores immutable audit trails for administrative operations and security events.
MySQL & SQLite compatible schema.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, Text

from app.domain.models.user import Base


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class AuditLog(Base):
    """Immutable audit trail for tracking administrative events."""

    __tablename__ = "audit_logs"

    id: str = Column(
        String(36),
        primary_key=True,
        default=_new_uuid,
        comment="UUID v4 of the audit record.",
    )
    actor_id: str = Column(
        String(36),
        nullable=False,
        index=True,
        comment="User ID who executed the action or 'system'.",
    )
    actor_email: str = Column(
        String(255),
        nullable=False,
        comment="Email of the actor for quick lookup.",
    )
    action: str = Column(
        String(64),
        nullable=False,
        index=True,
        comment="e.g. ROLE_UPDATED, USER_DEACTIVATED, JOB_DELETED, SETTINGS_UPDATED",
    )
    resource_type: str = Column(
        String(64),
        nullable=False,
        index=True,
        comment="e.g. user, job, system_setting",
    )
    resource_id: str | None = Column(
        String(64),
        nullable=True,
        comment="Target resource ID (e.g. target user_id or job_id).",
    )
    details_json: str | None = Column(
        Text,
        nullable=True,
        comment="Structured payload of the event details/parameters.",
    )
    ip_address: str | None = Column(
        String(45),
        nullable=True,
        comment="IPv4 / IPv6 address of the request.",
    )
    created_at: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        index=True,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<AuditLog id={self.id!r} actor={self.actor_email!r} "
            f"action={self.action!r}>"
        )
