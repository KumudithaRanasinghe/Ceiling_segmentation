"""
Admin Request Schemas
======================
Pydantic schemas for administrative actions, RBAC updates, and system configuration.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class UpdateUserRoleRequest(BaseModel):
    """Payload to modify user roles."""
    roles: List[str] = Field(
        ...,
        min_length=1,
        description="List of roles, e.g. ['user', 'admin']",
        examples=[["user", "admin"]],
    )


class AdminUpdateUserStatusRequest(BaseModel):
    """Payload to activate or deactivate a user account."""
    is_active: bool = Field(
        ...,
        description="True to activate, False to soft-disable account.",
    )


class AdminResetPasswordRequest(BaseModel):
    """Payload for admin-forced password reset."""
    new_password: str = Field(
        ...,
        min_length=8,
        description="New temporary or permanent password for the user.",
    )


class SystemSettingUpdateRequest(BaseModel):
    """Payload to update or create a system configuration setting."""
    value: Any = Field(
        ...,
        description="New setting value (primitive, list, or dictionary).",
    )
    category: Optional[str] = Field(
        "general",
        description="Category classification for the setting.",
    )
    description: Optional[str] = Field(
        None,
        description="Optional human-readable description.",
    )


class MaterialPricingUpdateRequest(BaseModel):
    """Payload to update global material cost catalogs and waste factors."""
    unit_prices: Dict[str, float] = Field(
        ...,
        description="Mapping of material names to unit price.",
        examples=[{
            "drywall_sheet_4x8": 14.50,
            "screw_pack_100ct": 8.20,
            "joint_compound_gal": 18.00,
            "furring_channel_10ft": 6.75,
            "perimeter_track_10ft": 7.50,
        }],
    )
    waste_factor: float = Field(
        1.10,
        ge=1.0,
        le=2.0,
        description="Waste buffer multiplier (e.g. 1.10 = 10% buffer).",
    )
    currency: str = Field(
        "USD",
        min_length=3,
        max_length=3,
        description="ISO 4217 Currency code, e.g. USD, EUR, LKR.",
    )


class PurgeJobsRequest(BaseModel):
    """Payload to initiate a batch job cleanup."""
    older_than_days: int = Field(
        7,
        ge=1,
        description="Purge completed jobs created more than N days ago.",
    )
    include_failed: bool = Field(
        True,
        description="Whether to also purge failed jobs.",
    )
