"""
API Router — registers all v1 endpoints.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import admin, auth, health, segmentation

api_router = APIRouter()
api_router.include_router(auth.router,         prefix="/auth",     tags=["Authentication"])
api_router.include_router(segmentation.router, prefix="/segment",  tags=["Segmentation"])
api_router.include_router(health.router,       prefix="/health",   tags=["Health"])
api_router.include_router(admin.router,        prefix="/admin",    tags=["Admin"])
