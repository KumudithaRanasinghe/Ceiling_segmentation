"""
Core configuration — all values come from environment variables.
Use a .env file locally. Never hardcode secrets.
"""
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Project ────────────────────────────────────────────────────────
    PROJECT_NAME: str = "Ceiling AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False

    # ── CORS ───────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["*"]          # restrict in production

    # ── Model paths ────────────────────────────────────────────────────
    # Point both at your single model file if it does segmentation only.
    # Rename spaces to underscores: "segmentation optimized.pth" → "segmentation_optimized.pth"
    ENCODER: str = "efficientnet-b4" # "efficientnet-b4" or "resnet50"
    DENOISER_MODEL_PATH: Optional[str] = None
    SEGMENTER_MODEL_PATH: str = "models/best_model_optimized.pth"
    DEVICE: str = "cpu"                          # "cuda" or "cpu"

    # ── Inference ──────────────────────────────────────────────────────
    INPUT_SIZE: int = 512                        # must match your training resolution
    NUM_CLASSES: int = 4                         # background + number of material classes
    CONFIDENCE_THRESHOLD: float = 0.5
    MIN_REGION_AREA_PX: int = 500               # ignore tiny noise regions

    # ── Image constraints ─────────────────────────────────────────────
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_MIME_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]

    # ── Scale calibration ─────────────────────────────────────────────
    DEFAULT_PIXELS_PER_METER: float = 100.0

    # ── Material overhead factor (waste + overlap) ─────────────────────
    MATERIAL_OVERHEAD_FACTOR: float = 1.10      # 10% waste buffer

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()