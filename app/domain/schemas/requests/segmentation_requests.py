"""
Segmentation Request DTOs
===========================
Pydantic models for incoming payloads on /segment endpoints.

Note: The main segmentation input is a multipart file upload, handled by
FastAPI's UploadFile + Form directly in the endpoint. This file holds any
additional body-level request models if needed (e.g., batch requests,
configuration overrides).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SegmentationRequest(BaseModel):
    """
    Optional configuration overrides for a segmentation run.

    Clients send this alongside the image in multipart/form-data.
    All fields are optional — the server falls back to config defaults.
    """
    pixels_per_meter: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Scale calibration: how many pixels correspond to 1 metre. "
            "Place a ruler or known-length object in the image and measure "
            "it in pixels. Omit to use the server default (100 px/m)."
        ),
    )

    model_config = {"extra": "forbid"}
