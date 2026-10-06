"""Validation rule CRUD.

Rules are data, not deployment artefacts: the people who own the policy edit
them here, without a release.
"""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.v1.models.requests import (
    BulkRulesRequest,
    CreateRuleRequest,
    UpdateRuleRequest,
)
from src.api.v1.models.responses import (
    BulkRulesResponse,
    RuleListResponse,
    RuleResponse,
)
from src.core.database.connection import get_db
from src.core.errors.exceptions import NotFoundError, ValidationError
from src.core.logging.structured import get_logger
from src.models.database.validation_rule import ValidationRule
from src.repositories.validation_rule_repository import ValidationRuleRepository

router = APIRouter(prefix="/rules", tags=["validation rules"])
logger = get_logger(__name__)


def _repo(session: AsyncSession = Depends(get_db)) -> ValidationRuleRepository:
    return ValidationRuleRepository(session)


@router.get("", response_model=RuleListResponse)
async def list_rules(
    doctype: Optional[str] = Query(
        default=None,
        description="Rules for this document type, plus the global ones. Omit for everything.",
    ),
    enabled_only: bool = Query(default=False),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
    repo: ValidationRuleRepository = Depends(_repo),
) -> RuleListResponse:
    """List validation rules."""
    if doctype is not None:
        rules = await repo.get_for_doctype(doctype, enabled_only=enabled_only)
    else:
        filters = {"enabled": True} if enabled_only else None
        rules = await repo.get_all(skip=skip, limit=limit, filters=filters)
    return RuleListResponse(
        rules=[RuleResponse.model_validate(r.to_dict()) for r in rules], count=len(rules)
    )


@router.post("", response_model=RuleResponse, status_code=status.HTTP_201_CREATED)
async def create_rule(
    request: CreateRuleRequest,
    repo: ValidationRuleRepository = Depends(_repo),
) -> RuleResponse:
    """Create a rule. ``(doctype, code)`` must be unique."""
    if await repo.get_by_code(request.code, request.doctype):
        raise ValidationError(
            f"rule {request.code!r} already exists for doctype {request.doctype!r}"
        )
    try:
        rule = await repo.create(ValidationRule(**request.model_dump()))
    except IntegrityError as exc:
        raise ValidationError(f"could not create rule: {exc.orig}") from exc
    logger.info("created rule %s (%s)", rule.code, rule.doctype or "global")
    return RuleResponse.model_validate(rule.to_dict())


@router.get("/{rule_id}", response_model=RuleResponse)
async def get_rule(
    rule_id: UUID,
    repo: ValidationRuleRepository = Depends(_repo),
) -> RuleResponse:
    rule = await repo.get_by_id(rule_id)
    if rule is None:
        raise NotFoundError(f"no rule with id {rule_id}")
    return RuleResponse.model_validate(rule.to_dict())


@router.patch("/{rule_id}", response_model=RuleResponse)
async def update_rule(
    rule_id: UUID,
    request: UpdateRuleRequest,
    repo: ValidationRuleRepository = Depends(_repo),
) -> RuleResponse:
    """Update a rule. Only the fields sent are changed."""
    rule = await repo.get_by_id(rule_id)
    if rule is None:
        raise NotFoundError(f"no rule with id {rule_id}")
    updated = await repo.update(rule, request.model_dump(exclude_unset=True))
    return RuleResponse.model_validate(updated.to_dict())


@router.delete("/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rule(
    rule_id: UUID,
    repo: ValidationRuleRepository = Depends(_repo),
) -> None:
    if not await repo.delete_by_id(rule_id):
        raise NotFoundError(f"no rule with id {rule_id}")


@router.post("/bulk", response_model=BulkRulesResponse)
async def upsert_rules(
    request: BulkRulesRequest,
    repo: ValidationRuleRepository = Depends(_repo),
) -> BulkRulesResponse:
    """Insert or update many rules at once, keyed by ``(doctype, code)``.

    This is how a playbook is loaded or replaced wholesale.
    """
    counts = await repo.upsert_many([r.model_dump() for r in request.rules])
    return BulkRulesResponse(**counts, total=len(request.rules))
