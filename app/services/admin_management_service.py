"""
Admin Management Service
=========================
Handles business logic for admin-exclusive operations:
User governance, global job management, system pricing catalogs, and diagnostics.
Automatically records immutable audit trails.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import hash_password
from app.domain.models.audit_log import AuditLog
from app.domain.models.segmentation_job import SegmentationJob
from app.domain.models.user import User
from app.domain.repositories.audit_repository import AuditRepository
from app.domain.repositories.job_repository import JobRepository
from app.domain.repositories.setting_repository import SettingRepository
from app.domain.repositories.user_repository import UserRepository
from app.domain.schemas.requests.admin_requests import (
    AdminResetPasswordRequest,
    AdminUpdateUserStatusRequest,
    MaterialPricingUpdateRequest,
    PurgeJobsRequest,
    UpdateUserRoleRequest,
)
from app.domain.schemas.responses.admin_responses import (
    AdminJobListItem,
    AdminJobListResponse,
    AdminUserListItem,
    AdminUserListResponse,
    AuditLogItem,
    AuditLogListResponse,
    MaterialPricingConfigResponse,
    SystemHealthResponse,
)


class AdminManagementService:
    """Business logic handler for admin operations."""

    def __init__(self, db: Session, admin_actor: Optional[User] = None, ip_address: Optional[str] = None):
        self.db = db
        self.admin_actor = admin_actor
        self.ip_address = ip_address

    def _audit(self, action: str, resource_type: str, resource_id: Optional[str] = None, details: Optional[dict] = None) -> None:
        """Helper to record audit events."""
        actor_id = self.admin_actor.id if self.admin_actor else "system"
        actor_email = self.admin_actor.email if self.admin_actor else "system@ceiling.ai"
        AuditRepository.log_event(
            self.db,
            actor_id=actor_id,
            actor_email=actor_email,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=self.ip_address,
        )

    # ── User Governance ───────────────────────────────────────────────────────

    def list_users(
        self,
        page: int = 1,
        page_size: int = 20,
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
        role: Optional[str] = None,
    ) -> AdminUserListResponse:
        """List users with job statistics."""
        users, total = UserRepository.list_users_admin(
            self.db, page=page, page_size=page_size, search=search, is_active=is_active, role=role
        )

        user_ids = [u.id for u in users]
        # Query job count & last job timestamp per user
        job_stats = (
            self.db.query(
                SegmentationJob.user_id,
                func.count(SegmentationJob.id),
                func.max(SegmentationJob.created_at),
            )
            .filter(SegmentationJob.user_id.in_(user_ids))
            .group_by(SegmentationJob.user_id)
            .all()
        )
        stats_map = {row[0]: (row[1], row[2]) for row in job_stats}

        items: List[AdminUserListItem] = []
        for u in users:
            roles_list = json.loads(u.roles) if isinstance(u.roles, str) else u.roles
            j_count, last_job = stats_map.get(u.id, (0, None))
            items.append(
                AdminUserListItem(
                    id=u.id,
                    email=u.email,
                    username=u.username,
                    full_name=u.full_name,
                    roles=roles_list,
                    is_active=u.is_active,
                    is_superuser=u.is_superuser,
                    job_count=j_count,
                    last_job_at=last_job,
                    created_at=u.created_at,
                )
            )

        return AdminUserListResponse(
            total=total,
            page=page,
            page_size=page_size,
            users=items,
        )

    def update_user_roles(self, user_id: str, payload: UpdateUserRoleRequest) -> User:
        """Assign or revoke user roles."""
        user = UserRepository.get_by_id(self.db, user_id)
        if user is None:
            raise ValueError(f"User with ID '{user_id}' not found.")

        old_roles = json.loads(user.roles) if isinstance(user.roles, str) else user.roles
        UserRepository.update_roles(self.db, user, payload.roles)

        self._audit(
            action="USER_ROLE_UPDATED",
            resource_type="user",
            resource_id=user.id,
            details={"old_roles": old_roles, "new_roles": payload.roles, "target_email": user.email},
        )
        return user

    def set_user_status(self, user_id: str, payload: AdminUpdateUserStatusRequest) -> User:
        """Activate or deactivate user."""
        user = UserRepository.get_by_id(self.db, user_id)
        if user is None:
            raise ValueError(f"User with ID '{user_id}' not found.")
        if user.is_superuser:
            raise ValueError("Superuser accounts cannot be modified via this API.")

        old_status = user.is_active
        UserRepository.set_active_status(self.db, user, payload.is_active)

        self._audit(
            action="USER_STATUS_UPDATED",
            resource_type="user",
            resource_id=user.id,
            details={"old_status": old_status, "new_status": payload.is_active, "target_email": user.email},
        )
        return user

    def reset_user_password(self, user_id: str, payload: AdminResetPasswordRequest) -> None:
        """Admin forced password reset."""
        user = UserRepository.get_by_id(self.db, user_id)
        if user is None:
            raise ValueError(f"User with ID '{user_id}' not found.")

        UserRepository.update_password(self.db, user, payload.new_password)
        self._audit(
            action="USER_PASSWORD_RESET",
            resource_type="user",
            resource_id=user.id,
            details={"target_email": user.email},
        )

    # ── Job Management ────────────────────────────────────────────────────────

    def list_all_jobs(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        model_version: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> AdminJobListResponse:
        """Admin global job browser across all users."""
        jobs, total = JobRepository.list_admin(
            self.db,
            page=page,
            page_size=page_size,
            status=status,
            user_id=user_id,
            model_version=model_version,
            start_date=start_date,
            end_date=end_date,
        )

        # Batch load user emails
        u_ids = list({j.user_id for j in jobs})
        users = self.db.query(User.id, User.email).filter(User.id.in_(u_ids)).all()
        email_map = {u[0]: u[1] for u in users}

        items: List[AdminJobListItem] = []
        for job in jobs:
            total_area = None
            if job.status == "success" and job.full_result_json:
                try:
                    payload = json.loads(job.full_result_json)
                    total_area = payload.get("total_area_m2")
                except Exception:
                    pass

            items.append(
                AdminJobListItem(
                    id=job.id,
                    user_id=job.user_id,
                    user_email=email_map.get(job.user_id),
                    status=job.status,
                    image_filename=job.image_filename,
                    image_size_bytes=job.image_size_bytes,
                    content_type=job.content_type,
                    pixels_per_meter=job.pixels_per_meter,
                    inference_time_ms=job.inference_time_ms,
                    model_version=job.model_version,
                    total_area_m2=total_area,
                    created_at=job.created_at,
                    expires_at=job.expires_at,
                )
            )

        return AdminJobListResponse(
            total=total,
            page=page,
            page_size=page_size,
            jobs=items,
        )

    def delete_job(self, job_id: str) -> None:
        """Delete specific job and its saved result masks."""
        job = JobRepository.get_by_id(self.db, job_id)
        if job is None:
            raise ValueError(f"Job '{job_id}' not found.")

        # Remove mask directory from disk if exists
        job_dir = Path("results") / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)

        JobRepository.delete_job(self.db, job)
        self._audit(
            action="JOB_DELETED",
            resource_type="job",
            resource_id=job_id,
            details={"owner_user_id": job.user_id},
        )

    def purge_jobs(self, payload: PurgeJobsRequest) -> int:
        """Batch purge jobs older than N days."""
        now = datetime.now(tz=timezone.utc)
        cutoff = now - timedelta(days=payload.older_than_days)
        deleted_count = JobRepository.purge_older_than(
            self.db, cutoff=cutoff, include_failed=payload.include_failed
        )

        self._audit(
            action="BATCH_JOBS_PURGED",
            resource_type="job",
            details={
                "older_than_days": payload.older_than_days,
                "deleted_count": deleted_count,
            },
        )
        return deleted_count

    # ── Material Pricing & Dynamic Config ──────────────────────────────────────

    def get_pricing_config(self) -> MaterialPricingConfigResponse:
        """Retrieve material unit pricing catalog."""
        setting = SettingRepository.get(self.db, "material_pricing")
        default_prices = {
            "drywall_sheet_4x8": 14.50,
            "screw_pack_100ct": 8.20,
            "joint_compound_gal": 18.00,
            "furring_channel_10ft": 6.75,
            "perimeter_track_10ft": 7.50,
        }

        if setting is None:
            return MaterialPricingConfigResponse(
                unit_prices=default_prices,
                waste_factor=settings.MATERIAL_OVERHEAD_FACTOR,
                currency="USD",
                updated_at=datetime.now(tz=timezone.utc),
            )

        data = json.loads(setting.value_json)
        return MaterialPricingConfigResponse(
            unit_prices=data.get("unit_prices", default_prices),
            waste_factor=data.get("waste_factor", settings.MATERIAL_OVERHEAD_FACTOR),
            currency=data.get("currency", "USD"),
            updated_at=setting.updated_at,
            updated_by=setting.updated_by,
        )

    def update_pricing_config(self, payload: MaterialPricingUpdateRequest) -> MaterialPricingConfigResponse:
        """Update global material pricing catalog."""
        data = {
            "unit_prices": payload.unit_prices,
            "waste_factor": payload.waste_factor,
            "currency": payload.currency,
        }
        actor_id = self.admin_actor.id if self.admin_actor else None
        setting = SettingRepository.set(
            self.db,
            key="material_pricing",
            value=data,
            category="pricing",
            description="Global ceiling material estimation prices and waste multiplier",
            updated_by=actor_id,
        )

        self._audit(
            action="PRICING_CONFIG_UPDATED",
            resource_type="system_setting",
            resource_id="material_pricing",
            details=data,
        )

        return MaterialPricingConfigResponse(
            unit_prices=payload.unit_prices,
            waste_factor=payload.waste_factor,
            currency=payload.currency,
            updated_at=setting.updated_at,
            updated_by=actor_id,
        )

    # ── Audit Logs ────────────────────────────────────────────────────────────

    def list_audit_logs(
        self,
        page: int = 1,
        page_size: int = 50,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
    ) -> AuditLogListResponse:
        """Fetch paginated audit log entries."""
        logs, total = AuditRepository.list_logs(
            self.db, page=page, page_size=page_size, action=action, resource_type=resource_type
        )
        return AuditLogListResponse(
            total=total,
            page=page,
            page_size=page_size,
            logs=[AuditLogItem.model_validate(log) for log in logs],
        )

    # ── System Diagnostics ────────────────────────────────────────────────────

    def get_system_health(self) -> SystemHealthResponse:
        """System health and diagnostic checks."""
        users_count = self.db.query(func.count(User.id)).scalar() or 0
        jobs_count = self.db.query(func.count(SegmentationJob.id)).scalar() or 0
        audit_count = self.db.query(func.count(AuditLog.id)).scalar() or 0

        # Storage
        results_path = Path("results")
        file_count = 0
        total_bytes = 0
        if results_path.exists():
            for root, _, files in os.walk(results_path):
                file_count += len(files)
                for f in files:
                    try:
                        total_bytes += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass

        driver = "MySQL (PyMySQL)" if "mysql" in settings.DATABASE_URL else "SQLite"

        return SystemHealthResponse(
            database_status="healthy",
            database_driver=driver,
            total_users_count=users_count,
            total_jobs_count=jobs_count,
            total_audit_logs_count=audit_count,
            results_directory_files_count=file_count,
            results_directory_size_mb=round(total_bytes / (1024 * 1024), 2),
            model_loaded=True,
            active_device=settings.DEVICE,
            timestamp=datetime.now(tz=timezone.utc),
        )
