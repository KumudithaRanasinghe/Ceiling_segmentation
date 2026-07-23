"""
Rate Limiter Middleware
========================
Multi-route sliding-window rate limiter with independent limits per route group.

Route groups and limits:
  /api/v1/auth/*     → 10 req/min per IP  (brute-force protection)
  /api/v1/segment/*  → 5 req/min per USER (ML inference is expensive)

Why separate limits?
  - Auth endpoints need IP-level limiting (user isn't authenticated yet).
  - Segment endpoints use per-USER limiting (fairer — a proxy/NAT behind one IP
    shouldn't penalise other users; also prevents token-sharing abuse).

Production note: Replace the in-memory dicts with Redis for multi-instance
deployments. Use a Lua script for atomic increment + expiry (INCR + EXPIRE).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.config import settings


@dataclass(frozen=True)
class RateLimitRule:
    """Defines rate-limit behaviour for a URL prefix."""
    prefix: str
    max_requests: int        # requests allowed per window
    window_seconds: int      # sliding window size in seconds
    key_fn: str              # "ip" | "user" — what to bucket by
    label: str               # human-readable name for error messages


# ── Rules ──────────────────────────────────────────────────────────────────────
# Evaluated in order; first match wins.
_RULES: list[RateLimitRule] = [
    RateLimitRule(
        prefix="/api/v1/segment",
        max_requests=getattr(settings, "SEGMENT_RATE_LIMIT_PER_MINUTE", 5),
        window_seconds=60,
        key_fn="user",    # per authenticated user (extracted from JWT sub claim)
        label="Segmentation inference",
    ),
    RateLimitRule(
        prefix="/api/v1/auth",
        max_requests=getattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 10),
        window_seconds=60,
        key_fn="ip",      # per IP (user not authenticated yet)
        label="Auth",
    ),
]


def _get_client_ip(request: Request) -> str:
    """Extract real IP, accounting for reverse proxies."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _get_user_id(request: Request) -> Optional[str]:
    """
    Extract user_id (JWT 'sub' claim) from the Authorization header without
    running the full dependency stack. Returns None if token is absent/invalid.

    We do a lightweight decode here (signature NOT re-verified — we only need
    the subject for bucketing, not for access control). Full verification is
    done by get_current_user() in the route handler.
    """
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return None
    token = auth_header[7:].strip()
    try:
        # Import here to avoid circular imports at module load time
        from jose import jwt as jose_jwt
        payload = jose_jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        return payload.get("sub")
    except Exception:
        return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter with independent limits for:
      • /api/v1/auth/*    — 10 req/60s per IP
      • /api/v1/segment/* — 5 req/60s per authenticated user
    """

    def __init__(self, app) -> None:
        super().__init__(app)
        # One bucket dict per rule, keyed by (rule.prefix, client_key)
        self._buckets: dict[str, Deque[float]] = defaultdict(deque)

    def _bucket_key(self, rule: RateLimitRule, request: Request) -> str:
        if rule.key_fn == "user":
            user_id = _get_user_id(request) or _get_client_ip(request)
            return f"user:{rule.prefix}:{user_id}"
        return f"ip:{rule.prefix}:{_get_client_ip(request)}"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        # Find matching rule (first prefix match wins)
        matched: Optional[RateLimitRule] = None
        for rule in _RULES:
            if path.startswith(rule.prefix):
                matched = rule
                break

        if matched is None:
            return await call_next(request)

        now = time.monotonic()
        key = self._bucket_key(matched, request)
        bucket = self._buckets[key]

        # Evict timestamps outside the sliding window
        while bucket and now - bucket[0] > matched.window_seconds:
            bucket.popleft()

        if len(bucket) >= matched.max_requests:
            retry_after = int(matched.window_seconds - (now - bucket[0])) + 1
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": (
                        f"{matched.label} rate limit exceeded. "
                        f"Max {matched.max_requests} requests per "
                        f"{matched.window_seconds}s. "
                        f"Retry after {retry_after}s."
                    ),
                    "retry_after_seconds": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(matched.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                },
            )

        bucket.append(now)

        # Attach rate-limit headers to successful responses too
        response = await call_next(request)
        remaining = max(0, matched.max_requests - len(bucket))
        response.headers["X-RateLimit-Limit"] = str(matched.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
