"""
Domain Schemas — the explicit contract between API and clients.
Every field is documented. Pydantic enforces types at runtime.
"""
from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class CeilingMaterialType(str, Enum):
    GYPSUM_BOARD   = "gypsum_board"
    MODULAR_GRID   = "modular_grid"
    SUSPENDED      = "suspended"
    PVC_PANEL      = "pvc_panel"
    WOOD_PANEL     = "wood_panel"
    UNKNOWN        = "unknown"


class RoofType(str, Enum):
    FLAT   = "flat"
    GABLE  = "gable"
    HIP    = "hip"
    SHED   = "shed"
    UNKNOWN = "unknown"


# ── Sub-models ────────────────────────────────────────────────────────────────

class DetectedRegion(BaseModel):
    """A single segmented roof region."""
    region_id: int = Field(..., description="Unique region index in this image")
    area_pixels: int = Field(..., ge=0)
    area_m2: float   = Field(..., ge=0.0, description="Estimated area in square metres")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: List[int] = Field(..., description="[x, y, width, height] in pixels")
    centroid: List[float]   = Field(..., description="[x, y] centroid in pixels")


class MaterialEstimate(BaseModel):
    """Estimated material quantity for one ceiling material type."""
    material_type: CeilingMaterialType
    quantity_m2: float  = Field(..., ge=0.0, description="Net area required")
    quantity_with_waste_m2: float = Field(..., ge=0.0, description="Area + 10% waste buffer")
    unit_count: Optional[int] = Field(None, description="Number of standard panels/sheets")
    unit_size_m2: Optional[float] = Field(None, description="Standard panel size used")
    confidence: float = Field(..., ge=0.0, le=1.0)


# ── Response schema ───────────────────────────────────────────────────────────

class SegmentationResponse(BaseModel):
    """
    Primary API response.  Everything the Flutter app needs to render results
    and produce a material shopping list.
    """
    job_id: str                = Field(..., description="Unique identifier for this inference job")
    status: str                = Field(default="success")

    # Geometry
    roof_type: RoofType
    total_area_m2: float       = Field(..., ge=0.0)
    regions: List[DetectedRegion]

    # Material estimation
    materials: List[MaterialEstimate]

    # Assets
    segmentation_mask_url: str = Field(..., description="URL of coloured segmentation mask PNG")
    overlay_image_url: str     = Field(..., description="URL of original image with mask overlay")

    # Meta
    inference_time_ms: float
    model_version: str
    warnings: List[str]        = Field(default_factory=list)
    pixels_per_meter: float    = Field(..., description="Scale factor used for area conversion")

    class Config:
        use_enum_values = True


# ── Error schema ──────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: Optional[dict] = None
