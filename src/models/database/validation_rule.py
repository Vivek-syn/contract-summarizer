"""Database model for playbook validation rules.

Rules live in the database rather than YAML files so they can be edited through
the API by the people who own the policy, versioned with the rest of the data,
enabled and disabled per document type, and audited - none of which a file on
the application's disk supports.

Scoping is by ``doctype``: a rule with a doctype applies only to that kind of
document, and a rule with ``doctype = NULL`` applies to every document. That is
what replaces "which playbook file do I load".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import Boolean, CheckConstraint, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database.base import GUID, Base, JSONType


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ValidationRule(Base):
    """One playbook rule.

    Example:
        >>> ValidationRule(
        ...     code="LIAB-01",
        ...     name="Liability cap",
        ...     doctype="msa",
        ...     group_name="liability",
        ...     severity="high",
        ...     rule="Aggregate liability must be capped at no more than 12 months of fees.",
        ...     fallback="Uncapped carve-outs acceptable only for the IP indemnity.",
        ...     enabled=True,
        ... )
    """

    __tablename__ = "validation_rules"

    # --- identity --------------------------------------------------------
    id: Mapped[Any] = mapped_column(GUID, primary_key=True, default=uuid4)
    code: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Human-facing id quoted in findings, e.g. LIAB-01. Unique per doctype.",
    )

    # --- scoping ---------------------------------------------------------
    doctype: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="Document type this applies to; NULL means every document type.",
    )
    group_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)

    # --- content ---------------------------------------------------------
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    rule: Mapped[str] = mapped_column(
        Text, nullable=False, comment="The requirement, in natural language."
    )
    fallback: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="An acceptable alternative position, if any."
    )
    preferred_language: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="Clause text the organisation would rather see."
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # --- configuration ---------------------------------------------------
    severity: Mapped[str] = mapped_column(
        String(16), nullable=False, default="medium",
        comment="low | medium | high. Drives whether a deviation blocks.",
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    keywords: Mapped[Optional[List[str]]] = mapped_column(
        JSONType,
        nullable=True,
        comment="Extra retrieval terms, for when a rule's wording misses the clause language.",
    )
    extra: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONType, nullable=True, comment="Caller-defined metadata, passed through untouched."
    )

    # --- timestamps ------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=utc_now, onupdate=utc_now, nullable=False
    )

    __table_args__ = (
        Index("uq_validation_rules_doctype_code", "doctype", "code", unique=True),
        Index("idx_validation_rules_doctype_enabled", "doctype", "enabled"),
        CheckConstraint(
            "severity IN ('low', 'medium', 'high')", name="chk_validation_rules_severity"
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ValidationRule(code={self.code!r}, doctype={self.doctype!r}, "
            f"severity={self.severity}, enabled={self.enabled})>"
        )

    def to_prompt_dict(self) -> Dict[str, str]:
        """The shape the validation stage passes to the model.

        Deliberately the same keys the YAML playbooks used, so the stage did not
        have to change when rules moved into the database.
        """
        payload = {
            "id": self.code,
            "title": self.name,
            "rule": self.rule,
            "severity": self.severity,
        }
        if self.fallback:
            payload["fallback"] = self.fallback
        if self.preferred_language:
            payload["preferred_language"] = self.preferred_language
        if self.keywords:
            payload["keywords"] = " ".join(self.keywords)
        return payload

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "code": self.code,
            "doctype": self.doctype,
            "group_name": self.group_name,
            "name": self.name,
            "rule": self.rule,
            "fallback": self.fallback,
            "preferred_language": self.preferred_language,
            "notes": self.notes,
            "severity": self.severity,
            "enabled": self.enabled,
            "sort_order": self.sort_order,
            "keywords": self.keywords,
            "extra": self.extra,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
