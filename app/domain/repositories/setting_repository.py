"""
SystemSetting Repository
========================
Data-access layer for dynamic system configurations and material pricing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.domain.models.system_setting import SystemSetting


class SettingRepository:
    """CRUD operations for SystemSetting."""

    @staticmethod
    def get(db: Session, key: str) -> Optional[SystemSetting]:
        """Fetch setting by key."""
        return db.query(SystemSetting).filter(SystemSetting.key == key).first()

    @staticmethod
    def get_value(db: Session, key: str, default: Any = None) -> Any:
        """Fetch and deserialize setting value by key, returning default if not found."""
        setting = SettingRepository.get(db, key)
        if setting is None or setting.value_json is None:
            return default
        try:
            return json.loads(setting.value_json)
        except Exception:
            return default

    @staticmethod
    def set(
        db: Session,
        key: str,
        value: Any,
        category: str = "general",
        description: Optional[str] = None,
        updated_by: Optional[str] = None,
    ) -> SystemSetting:
        """Upsert a system setting."""
        now = datetime.now(tz=timezone.utc)
        serialized = json.dumps(value)
        setting = SettingRepository.get(db, key)

        if setting is None:
            setting = SystemSetting(
                key=key,
                value_json=serialized,
                category=category,
                description=description,
                updated_by=updated_by,
                updated_at=now,
            )
            db.add(setting)
        else:
            setting.value_json = serialized
            if category:
                setting.category = category
            if description:
                setting.description = description
            setting.updated_by = updated_by
            setting.updated_at = now

        db.commit()
        db.refresh(setting)
        return setting

    @staticmethod
    def get_by_category(db: Session, category: str) -> List[SystemSetting]:
        """Fetch all settings under a given category."""
        return db.query(SystemSetting).filter(SystemSetting.category == category).all()
