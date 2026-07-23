"""
SegmentationJob Repository
===========================
Pure data-access layer for SegmentationJob entities.
No business logic here — only DB reads and writes.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.domain.models.segmentation_job import SegmentationJob
from app.domain.schemas.segmentation import SegmentationResponse


class JobRepository:
    """CRUD operations for SegmentationJob."""

    # ── Reads ─────────────────────────────────────────────────────────────────

    @staticmethod
    def get_by_id(db: Session, job_id: str) -> Optional[SegmentationJob]:
        return db.query(SegmentationJob).filter(SegmentationJob.id == job_id).first()

    @staticmethod
    def get_by_id_and_user(
        db: Session, job_id: str, user_id: str
    ) -> Optional[SegmentationJob]:
        """Enforce ownership — users can only see their own jobs."""
        return (
            db.query(SegmentationJob)
            .filter(
                SegmentationJob.id == job_id,
                SegmentationJob.user_id == user_id,
            )
            .first()
        )

    @staticmethod
    def list_by_user(
        db: Session,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[SegmentationJob]:
        """Return the most recent jobs for a user (newest first)."""
        return (
            db.query(SegmentationJob)
            .filter(SegmentationJob.user_id == user_id)
            .order_by(SegmentationJob.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    # ── Writes ────────────────────────────────────────────────────────────────

    @staticmethod
    def create_from_response(
        db: Session,
        *,
        user_id: str,
        result: SegmentationResponse,
        image_filename: Optional[str] = None,
        image_size_bytes: Optional[int] = None,
        content_type: Optional[str] = None,
        pixels_per_meter: Optional[float] = None,
    ) -> SegmentationJob:
        """
        Persist a completed segmentation result.

        The job_id is taken from result.job_id so the client-facing ID
        and the DB primary key are always the same value.
        """
        now = datetime.now(tz=timezone.utc)
        ttl_days = getattr(settings, "JOB_RESULT_TTL_DAYS", 7)
        expires_at = now + timedelta(days=ttl_days)

        job = SegmentationJob(
            id=result.job_id,
            user_id=user_id,
            status=result.status,
            image_filename=image_filename,
            image_size_bytes=image_size_bytes,
            content_type=content_type,
            pixels_per_meter=pixels_per_meter,
            full_result_json=result.model_dump_json(),
            inference_time_ms=result.inference_time_ms,
            model_version=result.model_version,
            created_at=now,
            completed_at=now,
            expires_at=expires_at,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def result_from_job(job: SegmentationJob) -> Optional[SegmentationResponse]:
        """Deserialize a stored job's JSON blob back into a SegmentationResponse."""
        if job.full_result_json is None:
            return None
        return SegmentationResponse.model_validate_json(job.full_result_json)

    @staticmethod
    def purge_expired(db: Session) -> int:
        """
        Delete all jobs where expires_at < now().
        Returns the count of deleted rows.
        Call this from a nightly cron / APScheduler task.
        """
        now = datetime.now(tz=timezone.utc)
        count = (
            db.query(SegmentationJob)
            .filter(SegmentationJob.expires_at < now)
            .delete(synchronize_session=False)
        )
        db.commit()
        return count
