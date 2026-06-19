"""
Ceiling AI — FastAPI Application Factory.

Responsibilities:
  - Define the application lifespan (model load / unload)
  - Register middleware in the correct order (outermost first)
  - Mount static files for result images
  - Include all API routers
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.infrastructure.ml.model_registry import ModelRegistry

RESULTS_DIR = Path("results")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup → load models into memory once.
    Shutdown → release GPU / CPU memory cleanly.
    Using the lifespan context (not @app.on_event) is the modern FastAPI pattern.
    """
    configure_logging()
    RESULTS_DIR.mkdir(exist_ok=True)      # ensure output directory exists

    registry = ModelRegistry()
    await registry.load_all()
    app.state.model_registry = registry

    yield                                  # ← application runs here

    await registry.unload_all()


def create_application() -> FastAPI:
    application = FastAPI(
        title       = settings.PROJECT_NAME,
        description = (
            "AI-powered ceiling material estimation from roof area images. "
            "Upload a photo → get segmentation mask + material quantities."
        ),
        version     = settings.VERSION,
        openapi_url = f"{settings.API_V1_STR}/openapi.json",
        docs_url    = f"{settings.API_V1_STR}/docs",
        redoc_url   = f"{settings.API_V1_STR}/redoc",
        lifespan    = lifespan,
    )

    # ── Middleware (applied outermost-first) ──────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins     = settings.ALLOWED_ORIGINS,
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ── API routes ────────────────────────────────────────────────────────────
    application.include_router(api_router, prefix=settings.API_V1_STR)

    # ── Static files: serve result images at /results/{job_id}/mask.png ──────
    RESULTS_DIR.mkdir(exist_ok=True)
    application.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

    return application


app = create_application()
