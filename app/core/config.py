"""
Core configuration — all values come from environment variables.
Use a .env file locally. Never hardcode secrets.
"""
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings
import secrets


class Settings(BaseSettings):
    # ── Project ────────────────────────────────────────────────────────
    PROJECT_NAME: str = "Ceiling AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False

    # ── CORS ───────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["*"]          # restrict in production

    # ── Authentication & Security ──────────────────────────────────────
    # Generate a strong secret with: python -c "import secrets; print(secrets.token_hex(32))"
    # NEVER use the default in production.
    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION_USE_A_LONG_RANDOM_SECRET_KEY"
    ALGORITHM: str = "HS256"                     # HMAC-SHA256
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15        # short-lived access tokens
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7           # long-lived refresh tokens

    # ── Database ───────────────────────────────────────────────────────
    # SQLite by default (zero infra). Swap to Postgres in production:
    # DATABASE_URL=postgresql+psycopg2://user:pass@host/dbname
    DATABASE_URL: str = "sqlite:///./ceiling_ai.db"

    # ── Rate Limiting ──────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 10     # /auth/* per IP, 60s window
    SEGMENT_RATE_LIMIT_PER_MINUTE: int = 5       # /segment/* per USER, 60s window

    # ── Job Storage ────────────────────────────────────────────────────
    JOB_RESULT_TTL_DAYS: int = 7                 # auto-purge jobs older than N days

    # ── Model paths ────────────────────────────────────────────────────
    # Point both at your single model file if it does segmentation only.
    # Rename spaces to underscores: "segmentation optimized.pth" → "segmentation_optimized.pth"
    ENCODER: str = "efficientnet-b4" # "efficientnet-b4" or "resnet50"
    DENOISER_MODEL_PATH: Optional[str] = None
    SEGMENTER_MODEL_PATH: str = "models/best_model_optimized.pth"
    # V2 model — binary inside-floor vs background segmentation
    V2_SEGMENTER_MODEL_PATH: Optional[str] = None
    V2_NUM_CLASSES: int = 4                      # must match V2 training checkpoint
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
    # Default scale calibration for architectural floor plans in normalized 512x512 space:
    # A standard residential house spans ~12m–16m across ~400–450 pixels of the 512px canvas.
    # Therefore, 1 metre ≈ 34 pixels in 512x512 model mask space (34^2 ≈ 1,156 px²/m²).
    # This ensures uncalibrated house uploads yield realistic house areas (70–150 m²)
    # and realistic room areas (12–45 m²) rather than sub-room tiny fractions.
    DEFAULT_PIXELS_PER_METER: float = 34.0

    # ── Material overhead factor (waste + overlap) ─────────────────────
    MATERIAL_OVERHEAD_FACTOR: float = 1.10      # 10% waste buffer

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()