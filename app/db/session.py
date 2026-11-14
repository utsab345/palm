from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.db.base import Base

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _ensure_sqlite_directory(database_url: str) -> None:
    if not database_url.startswith("sqlite"):
        return
    path = database_url.split("///", 1)[-1]
    if not path or path == ":memory:":
        return
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        url = get_settings().database_url
        _ensure_sqlite_directory(url)
        _engine = create_async_engine(url, pool_pre_ping=True)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with get_session_factory()() as session:
        yield session


async def init_db() -> None:
    from app.db import models  # noqa: F401

    async with get_engine().begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
