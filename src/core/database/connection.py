"""Async engine and session management."""

from __future__ import annotations

from functools import lru_cache
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from src.core.config.settings import get_settings
from src.core.logging.structured import get_logger

logger = get_logger(__name__)


def ensure_sqlite_parent(url: str) -> None:
    """Create the directory holding a SQLite file if it is missing.

    SQLite will not create it, and fails with an opaque "unable to open database
    file". Alembic builds its own engine, so this has to be callable from there
    too rather than living inside ``get_engine``.
    """
    if not url.startswith("sqlite"):
        return
    from pathlib import Path

    path = url.split("///", 1)[-1].split("?", 1)[0]
    if path and path != ":memory:":
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """The process-wide async engine."""
    settings = get_settings().database
    ensure_sqlite_parent(settings.url)

    kwargs = {"echo": settings.echo, "future": True}
    # SQLite has no server-side pool to configure, and passing pool args errors.
    if not settings.url.startswith("sqlite"):
        kwargs.update(
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout,
            pool_recycle=settings.pool_recycle,
            pool_pre_ping=True,
        )
    logger.info("database: %s", settings.url.split("@")[-1])
    return create_async_engine(settings.url, **kwargs)


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        get_engine(), class_=AsyncSession, expire_on_commit=False, autoflush=False
    )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a session, rolled back on error."""
    async with get_session_factory()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def create_all() -> None:
    """Create tables from the models.

    For local development and tests only - deployments run Alembic migrations,
    which is the only path that can alter an existing schema safely.
    """
    from src.core.database.base import Base
    from src.models.database import validation_rule  # noqa: F401  registers the table

    async with get_engine().begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    logger.info("database tables ensured")


async def dispose_engine() -> None:
    """Close pooled connections on shutdown."""
    await get_engine().dispose()
