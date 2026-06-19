"""
API Router — registers all v1 endpoints.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import segmentation, health

api_router = APIRouter()
api_router.include_router(segmentation.router, prefix="/segment",  tags=["Segmentation"])
api_router.include_router(health.router,       prefix="/health",   tags=["Health"])
