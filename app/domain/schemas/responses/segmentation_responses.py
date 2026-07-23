"""
Segmentation Response DTOs
============================
Pydantic models for /segment endpoint responses.

Includes both the heavy full-result SegmentationResponse and
the lightweight SegmentationJobSummary for list views.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class CeilingMaterialType(str, Enum):
    GYPSUM_BOARD = "gypsum_board"
    MODULAR_GRID = "modular_grid"
    SUSPENDED    = "suspended"
    PVC_PANEL    = "pvc_panel"
    WOOD_PANEL   = "wood_panel"
    UNKNOWN      = "unknown"


class RoofType(str, Enum):
    FLAT    = "flat"
    GABLE   = "gable"
    HIP     = "hip"
    SHED    = "shed"
    UNKNOWN = "unknown"


# ── Sub-models ────────────────────────────────────────────────────────────────

class DetectedRegion(BaseModel):
    """A single segmented region with area and confidence metadata."""
    region_id: int     = Field(..., description="Unique region index within this image.")
    area_pixels: int   = Field(..., ge=0, description="Region area in pixels.")
    area_m2: float     = Field(..., ge=0.0, description="Estimated area in square metres.")
    confidence: float  = Field(..., ge=0.0, le=1.0, description="Model confidence (0–1).")
    bounding_box: List[int]   = Field(..., description="[x, y, width, height] in pixels.")
    centroid: List[float]     = Field(..., description="[x, y] centroid coordinate in pixels.")


class MaterialEstimate(BaseModel):
    """Estimated material quantity for one ceiling material type."""
    material_type: CeilingMaterialType
    quantity_m2: float            = Field(..., ge=0.0, description="Net area required (m²).")
    quantity_with_waste_m2: float = Field(..., ge=0.0, description="Area + 10% waste buffer (m²).")
    unit_count: Optional[int]     = Field(None, description="Number of standard panels/sheets.")
    unit_size_m2: Optional[float] = Field(None, description="Standard panel size used (m²).")
    confidence: float             = Field(..., ge=0.0, le=1.0)


# ── Full inference result ──────────────────────────────────────────────────────

class SegmentationResponse(BaseModel):
    """
    Full segmentation result — everything the Flutter app needs to
    render results and build a material shopping list.
    """
    job_id: str   = Field(..., description="Unique identifier — use with GET /segment/{job_id}.")
    status: str   = Field(default="success")

    # Geometry
    roof_type: RoofType
    total_area_m2: float          = Field(..., ge=0.0, description="Total detected ceiling area (m²).")
    regions: List[DetectedRegion] = Field(default_factory=list)

    # Material estimation
    materials: List[MaterialEstimate] = Field(default_factory=list)

    # Generated assets
    segmentation_mask_url: str = Field(..., description="URL of the coloured segmentation mask PNG.")
    overlay_image_url: str     = Field(..., description="URL of the original image with mask overlay.")

    # Performance metadata
    inference_time_ms: float
    model_version: str
    pixels_per_meter: float = Field(..., description="Scale factor used for area conversion.")
    warnings: List[str]     = Field(default_factory=list, description="Non-fatal pipeline warnings.")

    model_config = {
        "use_enum_values": True,
        "protected_namespaces": (),
    }


# ── Lightweight job list item ─────────────────────────────────────────────────

class SegmentationJobSummary(BaseModel):
    """
    Compact job record for list views (GET /segment/).
    Does NOT include the full result payload.
    """
    job_id: str
    status: str
    image_filename: Optional[str]    = None
    inference_time_ms: Optional[float] = None
    model_version: Optional[str]     = None
    created_at: datetime
    expires_at: Optional[datetime]   = None

    model_config = {
        "from_attributes": True,
        "protected_namespaces": (),
    }
