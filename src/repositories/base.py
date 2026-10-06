"""Base async repository.

All database access goes through a repository, so the pipeline and API layers
never hold a session or build a query.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging.structured import get_logger

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Common CRUD, extended with domain queries by each subclass."""

    model_class: Type[Any]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.logger = get_logger(f"repository.{self.__class__.__name__}")

    async def get_by_id(self, entity_id: UUID) -> Optional[T]:
        return await self.session.get(self.model_class, entity_id)

    async def get_all(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[T]:
        statement = select(self.model_class)
        for field, value in (filters or {}).items():
            statement = statement.where(getattr(self.model_class, field) == value)
        statement = statement.offset(skip).limit(limit)
        return list((await self.session.execute(statement)).scalars().all())

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        statement = select(func.count()).select_from(self.model_class)
        for field, value in (filters or {}).items():
            statement = statement.where(getattr(self.model_class, field) == value)
        return int((await self.session.execute(statement)).scalar_one())

    async def create(self, entity: T) -> T:
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def update(self, entity: T, values: Dict[str, Any]) -> T:
        for field, value in values.items():
            if value is not None or field in getattr(entity, "__nullable_fields__", ()):
                setattr(entity, field, value)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def delete_by_id(self, entity_id: UUID) -> bool:
        result = await self.session.execute(
            delete(self.model_class).where(self.model_class.id == entity_id)
        )
        await self.session.commit()
        return bool(result.rowcount)
