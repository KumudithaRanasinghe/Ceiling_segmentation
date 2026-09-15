"""
SystemSetting Domain Model (SQLAlchemy ORM)
===========================================
Stores dynamic system-wide configuration such as pricing catalogs, waste factors,
thresholds, and operational flags.
MySQL & SQLite compatible schema.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, Text

from app.domain.models.user import Base


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class SystemSetting(Base):
    """Dynamic key-value system settings."""

    __tablename__ = "system_settings"

    key: str = Column(
        String(64),
        primary_key=True,
        comment="Unique setting key (e.g. material_pricing, waste_factor).",
    )
    value_json: str = Column(
        Text,
        nullable=False,
        comment="JSON serialized setting value.",
    )
    category: str = Column(
        String(64),
        nullable=False,
        default="general",
        index=True,
        comment="e.g. pricing, inference, system",
    )
    description: str | None = Column(
        String(255),
        nullable=True,
        comment="Human-readable description of what this setting controls.",
    )
    updated_by: str | None = Column(
        String(36),
        nullable=True,
        comment="User ID who last modified this setting.",
    )
    updated_at: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<SystemSetting key={self.key!r} category={self.category!r}>"
