"""Data access for validation rules."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import or_, select

from src.core.logging.structured import get_logger
from src.models.database.validation_rule import ValidationRule
from src.repositories.base import BaseRepository

logger = get_logger(__name__)


class ValidationRuleRepository(BaseRepository[ValidationRule]):
    """Queries the validation stage and the rules API need."""

    model_class = ValidationRule

    async def get_by_code(self, code: str, doctype: Optional[str] = None) -> Optional[ValidationRule]:
        statement = select(ValidationRule).where(
            ValidationRule.code == code, ValidationRule.doctype == doctype
        )
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def get_for_doctype(
        self, doctype: Optional[str], *, enabled_only: bool = True
    ) -> List[ValidationRule]:
        """Rules that apply to a document type.

        Returns rules scoped to ``doctype`` plus the global ones (``doctype IS
        NULL``), which is what replaces choosing a playbook file. A type with no
        rules of its own still gets the organisation-wide ones.
        """
        statement = select(ValidationRule).where(
            or_(ValidationRule.doctype == doctype, ValidationRule.doctype.is_(None))
        )
        if enabled_only:
            statement = statement.where(ValidationRule.enabled.is_(True))
        statement = statement.order_by(ValidationRule.sort_order, ValidationRule.code)
        return list((await self.session.execute(statement)).scalars().all())

    async def get_by_group(
        self, group_name: str, doctype: Optional[str] = None
    ) -> List[ValidationRule]:
        statement = select(ValidationRule).where(ValidationRule.group_name == group_name)
        if doctype is not None:
            statement = statement.where(
                or_(ValidationRule.doctype == doctype, ValidationRule.doctype.is_(None))
            )
        return list((await self.session.execute(statement)).scalars().all())

    async def list_doctypes(self) -> List[Optional[str]]:
        """Every doctype that has at least one rule."""
        statement = select(ValidationRule.doctype).distinct()
        return list((await self.session.execute(statement)).scalars().all())

    async def upsert_many(self, rules: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        """Insert or update rules, keyed by ``(doctype, code)``.

        Used by seeding and by bulk edits from the API. Returns counts so the
        caller can report what actually changed rather than claiming success.
        """
        created = updated = 0
        for payload in rules:
            existing = await self.get_by_code(payload["code"], payload.get("doctype"))
            if existing is None:
                self.session.add(ValidationRule(**payload))
                created += 1
            else:
                for field, value in payload.items():
                    if field not in ("code", "doctype"):
                        setattr(existing, field, value)
                updated += 1
        await self.session.commit()
        logger.info("upserted rules: %d created, %d updated", created, updated)
        return {"created": created, "updated": updated}
