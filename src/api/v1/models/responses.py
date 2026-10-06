"""API response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.schemas.documents import PageQuality
from src.schemas.state import RunState, RunStatus, StageStatus


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    size_bytes: int
    flagged_pages: List[int] = Field(default_factory=list)
    page_quality: List[PageQuality] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    storage_uri: str


class StageResponse(BaseModel):
    key: str
    name: str
    status: StageStatus
    depends_on: List[str] = Field(default_factory=list)
    attempts: int = 0
    duration_seconds: Optional[float] = None
    cost_usd: float = 0.0
    tokens: int = 0
    error: Optional[str] = None


class RunResponse(BaseModel):
    run_id: str
    doc_id: str
    filename: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None
    stages: List[StageResponse] = Field(default_factory=list)
    progress: Dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0
    total_tokens: int = 0

    @classmethod
    def from_state(cls, state: RunState) -> "RunResponse":
        usage = state.usage
        return cls(
            run_id=state.run_id,
            doc_id=state.doc_id,
            filename=state.filename,
            status=state.status,
            created_at=state.created_at,
            updated_at=state.updated_at,
            error=state.error,
            stages=[
                StageResponse(
                    key=record.key,
                    name=record.name,
                    status=record.status,
                    depends_on=record.depends_on,
                    attempts=record.attempts,
                    duration_seconds=record.duration_seconds,
                    cost_usd=round(record.usage.cost_usd, 4),
                    tokens=record.usage.total_tokens,
                    error=record.error,
                )
                for record in state.stages.values()
            ],
            progress=state.progress(),
            cost_usd=round(usage.cost_usd, 4),
            total_tokens=usage.total_tokens,
        )


class RunListResponse(BaseModel):
    runs: List[RunResponse]
    count: int


class RetryResponse(BaseModel):
    run_id: str
    invalidated: List[str] = Field(description="Stages reset by this retry.")
    status: RunStatus


class ProviderHealth(BaseModel):
    name: str
    configured: bool
    detail: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    providers: List[ProviderHealth] = Field(default_factory=list)


class RuleResponse(BaseModel):
    """One validation rule as stored."""

    id: str
    code: str
    doctype: Optional[str] = None
    group_name: Optional[str] = None
    name: str
    rule: str
    fallback: Optional[str] = None
    preferred_language: Optional[str] = None
    notes: Optional[str] = None
    severity: str
    enabled: bool
    sort_order: int = 0
    keywords: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class RuleListResponse(BaseModel):
    rules: List[RuleResponse]
    count: int


class BulkRulesResponse(BaseModel):
    created: int
    updated: int
    total: int
