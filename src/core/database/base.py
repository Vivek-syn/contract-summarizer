"""Declarative base and column types that work on both Postgres and SQLite.

Postgres is the deployment target; SQLite keeps local development and the test
suite free of a database server. The two type shims below are the only places
that difference leaks.
"""

from __future__ import annotations

import uuid

from sqlalchemy import JSON, String, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""


class GUID(TypeDecorator):
    """UUID column: native on Postgres, 36-char string elsewhere."""

    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        return value if dialect.name == "postgresql" else str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


# JSONB on Postgres (indexable, typed) and plain JSON elsewhere.
JSONType = JSONB().with_variant(JSON(), "sqlite")
