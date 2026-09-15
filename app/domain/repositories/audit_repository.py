"""
AuditLog Repository
====================
Data-access layer for administrative audit records.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.domain.models.audit_log import AuditLog


class AuditRepository:
    """CRUD operations for AuditLog."""

    @staticmethod
    def log_event(
        db: Session,
        *,
        actor_id: str,
        actor_email: str,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> AuditLog:
        """Create and persist an audit trail record."""
        entry = AuditLog(
            actor_id=actor_id,
            actor_email=actor_email,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details_json=json.dumps(details) if details is not None else None,
            ip_address=ip_address,
            created_at=datetime.now(tz=timezone.utc),
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return entry

    @staticmethod
    def list_logs(
        db: Session,
        page: int = 1,
        page_size: int = 50,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> Tuple[List[AuditLog], int]:
        """Fetch paginated audit logs with optional filters."""
        query = db.query(AuditLog)
        if action:
            query = query.filter(AuditLog.action == action)
        if resource_type:
            query = query.filter(AuditLog.resource_type == resource_type)
        if actor_id:
            query = query.filter(AuditLog.actor_id == actor_id)

        total = query.count()
        offset = (page - 1) * page_size
        logs = query.order_by(desc(AuditLog.created_at)).offset(offset).limit(page_size).all()
        return logs, total
