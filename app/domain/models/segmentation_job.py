"""
SegmentationJob Domain Model (SQLAlchemy ORM)
==============================================
Stores the result of every completed segmentation inference job.

Why SQL (SQLite/PostgreSQL) over NoSQL for this data?
------------------------------------------------------
See the database design decision document for full reasoning. Summary:

  ✅ SQL chosen because:
  - Data has a well-defined, stable schema (job_id, user_id, timestamps,
    structured material estimates, etc.)
  - We already have SQLAlchemy + SQLite set up — zero extra infrastructure.
  - Segmentation results are naturally relational: one User → many Jobs.
  - Rich querying is needed: filter by user, date range, status, sort by
    created_at — these are SQL's strongest use-case.
  - The JSON blob (full_result) uses TEXT/JSONB to store flexible nested data
    without losing the benefits of relational keys and indexes.
  - ACID guarantees matter: a half-written job record must never be visible.

  ❌ NoSQL (MongoDB / DynamoDB) would NOT be better here because:
  - Data is not document-centric — it's record-centric with foreign keys.
  - We'd lose JOIN capability (user ↔ job) or have to denormalize.
  - Adding another DB technology (Mongo) increases infrastructure complexity
    for no tangible benefit at this scale.
  - SQLite's JSON1 extension already handles the flexible blob column.

Schema decisions:
  - UUID job_id (matches the job_id in SegmentationResponse).
  - user_id FK into the users table — lets us query "all jobs for user X".
  - status: pending | running | success | failed (supports future async jobs).
  - full_result_json: stores the complete SegmentationResponse as JSON text.
    This is a deliberate denormalization — avoids complex normalizing 8+ tables
    while still allowing the common query pattern (fetch by job_id → return JSON).
  - image_filename, image_size_bytes, content_type: lightweight metadata for
    debugging and audit without re-storing the raw image bytes.
  - inference_time_ms and model_version: captured for performance monitoring.
  - expires_at: enables a cron/background task to purge old records (TTL).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import relationship

from app.domain.models.user import Base


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class SegmentationJob(Base):
    """Persisted segmentation inference job."""

    __tablename__ = "segmentation_jobs"

    # ── Primary key ───────────────────────────────────────────────────────────
    id: str = Column(
        String(36),
        primary_key=True,
        default=_new_uuid,
        comment="UUID v4 — used as job_id in API responses.",
    )

    # ── Ownership ─────────────────────────────────────────────────────────────
    user_id: str = Column(
        String(36),
        nullable=False,
        index=True,
        comment="References users.id — NOT a FK constraint to keep migrations simple.",
    )

    # ── Job lifecycle ─────────────────────────────────────────────────────────
    status: str = Column(
        String(16),
        nullable=False,
        default="success",
        comment="pending | running | success | failed",
    )
    error_detail: str | None = Column(
        Text,
        nullable=True,
        comment="Error message when status=failed.",
    )

    # ── Input metadata ────────────────────────────────────────────────────────
    image_filename: str | None = Column(String(255), nullable=True)
    image_size_bytes: int | None = Column(Integer, nullable=True)
    content_type: str | None = Column(String(64), nullable=True)
    pixels_per_meter: float | None = Column(Float, nullable=True)

    # ── Result (JSON blob) ────────────────────────────────────────────────────
    # Stores the full SegmentationResponse as a JSON string.
    # SQLite uses TEXT; Postgres uses JSONB (swap type in production migration).
    full_result_json: str | None = Column(
        Text,
        nullable=True,
        comment="Full SegmentationResponse serialized as JSON.",
    )

    # ── Performance metadata ──────────────────────────────────────────────────
    inference_time_ms: float | None = Column(Float, nullable=True)
    model_version: str | None = Column(String(64), nullable=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        index=True,
    )
    completed_at: datetime | None = Column(
        DateTime(timezone=True),
        nullable=True,
    )
    expires_at: datetime | None = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="After this timestamp the record can be purged by a cleanup job.",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<SegmentationJob id={self.id!r} user={self.user_id!r} "
            f"status={self.status!r}>"
        )
