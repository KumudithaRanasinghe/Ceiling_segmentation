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
import app.domain.models.segmentation_job  # noqa: F401 — registers SegmentationJob table with Base

_engine = create_engine(
    settings.DATABASE_URL,
    # SQLite requires this to allow the same connection across threads
    # (FastAPI runs handlers in a thread pool).
    connect_args=(
        {"check_same_thread": False}
        if settings.DATABASE_URL.startswith("sqlite")
        else {}
    ),
    echo=settings.DEBUG,   # log SQL statements in DEBUG mode only
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
