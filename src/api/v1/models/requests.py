"""API request models."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """Per-run overrides. Every field is optional; omitted ones use settings."""

    document_type: Optional[str] = Field(
        default=None,
        description="Skip detection and use this type. The user's override always wins.",
    )
    rule_group: Optional[str] = Field(
        default=None,
        description=(
            "Validate only this group of rules. Omit to use every enabled rule for "
            "the document type, plus the global ones."
        ),
    )
    enable_grounding: Optional[bool] = None
    enable_validation: Optional[bool] = None
    model: Optional[str] = Field(default=None, description="LiteLLM-qualified model id.")
    effort: Optional[str] = Field(default=None, description="low | medium | high | xhigh | max")


class CreateRunRequest(BaseModel):
    doc_id: str = Field(description="Id returned by the upload endpoint.")
    config: RunConfig = Field(default_factory=RunConfig)


class RetryRequest(BaseModel):
    """Scoped retry: invalidate one stage and everything downstream.

    This is what the human gate does. Naming ``typedetect`` re-runs three stages
    and leaves OCR alone; naming ``ocr`` re-runs everything.
    """

    stage: str = Field(
        description="Stage key to invalidate, e.g. 'typedetect', 'analysis', 'validation'."
    )
    reason: Optional[str] = Field(
        default=None,
        description=(
            "Why the reviewer rejected it. Recorded even though nothing reads it yet - "
            "these are labelled failure cases and cannot be reconstructed later."
        ),
    )
    config: Optional[RunConfig] = Field(
        default=None, description="Config changes to apply before re-running."
    )


class RuleBase(BaseModel):
    """Fields shared by rule create and bulk upsert."""

    code: str = Field(description="Human-facing id quoted in findings, e.g. LIAB-01.")
    name: str = Field(description="Short title.")
    rule: str = Field(description="The requirement, in natural language.")
    doctype: Optional[str] = Field(
        default=None,
        description="Document type this applies to. Null means every document type.",
    )
    group_name: Optional[str] = None
    fallback: Optional[str] = Field(
        default=None, description="An acceptable alternative position, if any."
    )
    preferred_language: Optional[str] = None
    notes: Optional[str] = None
    severity: Literal["low", "medium", "high"] = "medium"
    enabled: bool = True
    sort_order: int = 0
    keywords: Optional[List[str]] = Field(
        default=None, description="Extra retrieval terms for matching clauses to this rule."
    )
    extra: Optional[Dict[str, Any]] = None


class CreateRuleRequest(RuleBase):
    """A new validation rule."""


class UpdateRuleRequest(BaseModel):
    """A partial update. Only the fields sent are changed."""

    name: Optional[str] = None
    rule: Optional[str] = None
    doctype: Optional[str] = None
    group_name: Optional[str] = None
    fallback: Optional[str] = None
    preferred_language: Optional[str] = None
    notes: Optional[str] = None
    severity: Optional[Literal["low", "medium", "high"]] = None
    enabled: Optional[bool] = None
    sort_order: Optional[int] = None
    keywords: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None


class BulkRulesRequest(BaseModel):
    """Insert or update many rules, keyed by (doctype, code)."""

    rules: List[RuleBase] = Field(min_length=1)
