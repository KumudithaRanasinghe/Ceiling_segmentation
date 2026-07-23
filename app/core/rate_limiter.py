"""
Rate Limiter Middleware
========================
Sliding-window rate limiter for /auth/* routes.
In-memory implementation — for horizontal scaling, replace
the _store dict with Redis atomic increment + expiry.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter applied to /auth/* routes."""

    def __init__(self, app, *, route_prefix: str = "/api/v1/auth") -> None:
        super().__init__(app)
        self._prefix = route_prefix
        self._buckets: dict[str, Deque[float]] = defaultdict(deque)
        self._window_seconds: int = 60
        self._max_requests: int = getattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 10)

    def _get_client_ip(self, request: Request) -> str:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not request.url.path.startswith(self._prefix):
            return await call_next(request)

        ip = self._get_client_ip(request)
        now = time.monotonic()
        bucket = self._buckets[ip]

        while bucket and now - bucket[0] > self._window_seconds:
            bucket.popleft()

        if len(bucket) >= self._max_requests:
            retry_after = int(self._window_seconds - (now - bucket[0])) + 1
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": (
                        f"Too many requests. Max {self._max_requests} per "
                        f"{self._window_seconds}s per IP. Retry after {retry_after}s."
                    )
                },
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)
        return await call_next(request)
