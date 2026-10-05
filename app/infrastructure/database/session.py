"""
Database Session & Engine Factory
====================================
Configures SQLAlchemy engine and session for user storage.

For SQLite (default, zero infra). Swap DATABASE_URL in .env to
postgresql+psycopg2://user:pass@host/dbname for production Postgres.

Design decisions:
  - Session is yielded as a FastAPI dependency (request-scoped).
  - autocommit=False: all writes must call explicit commit().
  - autoflush=False: prevents implicit DB round-trips mid-request.
  - expire_on_commit=False: lets us read attributes after commit without
    re-fetching (important when returning ORM objects after commit).
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.domain.models.user import Base
import app.domain.models.segmentation_job  # noqa: F401
import app.domain.models.audit_log          # noqa: F401
import app.domain.models.system_setting    # noqa: F401

_engine_kwargs = {
    "echo": settings.DEBUG,
}

if settings.DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
elif "mysql" in settings.DATABASE_URL:
    # MySQL production-grade connection pooling
    _engine_kwargs.update({
        "pool_pre_ping": True,
        "pool_recycle": 3600,
        "pool_size": 10,
        "max_overflow": 20,
    })
elif "postgresql" in settings.DATABASE_URL:
    # Postgres (Neon / Supabase) — pool_pre_ping keeps connections alive
    # across Neon's serverless idle disconnects
    _engine_kwargs.update({
        "pool_pre_ping": True,
        "pool_recycle": 1800,   # Neon idles out at ~5 min; recycle at 30 min
        "pool_size": 5,
        "max_overflow": 10,
    })

_engine = create_engine(
    settings.DATABASE_URL,
    **_engine_kwargs,
)

_SessionLocal = sessionmaker(
    bind=_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def init_db() -> None:
    """
    Create all tables if they don't exist yet (idempotent).
    Called once at application startup from main.py lifespan.
    """
    Base.metadata.create_all(bind=_engine)


def get_db():  # type: ignore[return]
    """
    FastAPI dependency: yields a request-scoped DB session,
    always closes it on exit (even if an exception is raised).

    Usage:
        @router.get("/")
        def handler(db: Session = Depends(get_db)):
            ...
    """
    db: Session = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
