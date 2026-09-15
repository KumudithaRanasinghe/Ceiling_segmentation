"""
SegmentationJob Repository
===========================
Pure data-access layer for SegmentationJob entities.
Provides optimized SQL queries for both user-level and admin-level telemetry & management.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.domain.models.segmentation_job import SegmentationJob
from app.domain.schemas.responses import SegmentationResponse


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

    @staticmethod
    def list_admin(
        db: Session,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        model_version: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Tuple[List[SegmentationJob], int]:
        """Admin multi-criteria filter & pagination across all users."""
        query = db.query(SegmentationJob)

        if status:
            query = query.filter(SegmentationJob.status == status)
        if user_id:
            query = query.filter(SegmentationJob.user_id == user_id)
        if model_version:
            query = query.filter(SegmentationJob.model_version == model_version)
        if start_date:
            query = query.filter(SegmentationJob.created_at >= start_date)
        if end_date:
            query = query.filter(SegmentationJob.created_at <= end_date)

        total = query.count()
        offset = (page - 1) * page_size
        jobs = query.order_by(desc(SegmentationJob.created_at)).offset(offset).limit(page_size).all()
        return jobs, total

    @staticmethod
    def get_jobs_in_timerange(
        db: Session,
        start_date: datetime,
        end_date: datetime,
    ) -> List[SegmentationJob]:
        """Fetch jobs between start_date and end_date for analytics aggregations."""
        return (
            db.query(SegmentationJob)
            .filter(
                SegmentationJob.created_at >= start_date,
                SegmentationJob.created_at <= end_date,
            )
            .order_by(SegmentationJob.created_at.asc())
            .all()
        )

    @staticmethod
    def count_by_status(db: Session) -> Dict[str, int]:
        """Return counts grouped by status."""
        rows = (
            db.query(SegmentationJob.status, func.count(SegmentationJob.id))
            .group_by(SegmentationJob.status)
            .all()
        )
        return {row[0]: row[1] for row in rows}

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
        """Persist a completed segmentation result."""
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
    def delete_job(db: Session, job: SegmentationJob) -> None:
        """Hard delete a single job."""
        db.delete(job)
        db.commit()

    @staticmethod
    def result_from_job(job: SegmentationJob) -> Optional[SegmentationResponse]:
        """Deserialize a stored job's JSON blob back into a SegmentationResponse."""
        if job.full_result_json is None:
            return None
        return SegmentationResponse.model_validate_json(job.full_result_json)

    @staticmethod
    def purge_expired(db: Session) -> int:
        """Delete all jobs where expires_at < now()."""
        now = datetime.now(tz=timezone.utc)
        count = (
            db.query(SegmentationJob)
            .filter(SegmentationJob.expires_at < now)
            .delete(synchronize_session=False)
        )
        db.commit()
        return count

    @staticmethod
    def purge_older_than(
        db: Session,
        cutoff: datetime,
        include_failed: bool = True,
    ) -> int:
        """Admin triggered purge of older jobs."""
        query = db.query(SegmentationJob).filter(SegmentationJob.created_at < cutoff)
        if not include_failed:
            query = query.filter(SegmentationJob.status != "failed")
        count = query.delete(synchronize_session=False)
        db.commit()
        return count
