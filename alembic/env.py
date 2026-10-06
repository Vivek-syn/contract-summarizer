"""Alembic environment.

The URL comes from DB_URL via settings, so migrations and the app can never
point at different databases.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy import pool

from src.core.config.settings import get_settings
from src.core.database.base import Base
from src.core.database.connection import ensure_sqlite_parent
from src.models.database import validation_rule  # noqa: F401  registers the table

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

_url = get_settings().database.url
ensure_sqlite_parent(_url)
config.set_main_option("sqlalchemy.url", _url)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # batch mode lets ALTER work on SQLite, which has no real ALTER TABLE
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_do_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
