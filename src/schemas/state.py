"""Run state: a DAG of stage records.

Not a progress percentage - with parallel and optional branches a percentage is
a lie. Modelling it as a graph gives two things for free: per-stage cost
attribution for the quota service, and scoped retry, where the human gate
answers "which node do I invalidate" rather than "approve or reject".
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StageStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
    skipped = "skipped"
    stale = "stale"  # invalidated by a retry upstream


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls
        self.cost_usd += other.cost_usd

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class StageRecord(BaseModel):
    """One node in the run DAG.

    ``key`` is the addressable name a retry targets. Fan-out stages are keyed
    per unit - ``ocr:7``, ``validation:NOTICE-01`` - so one page or one rule can
    be re-run without touching its siblings.
    """

    key: str
    name: str
    depends_on: List[str] = Field(default_factory=list)
    status: StageStatus = StageStatus.pending
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    attempts: int = 0
    usage: Usage = Field(default_factory=Usage)
    error: Optional[str] = None
    output_ref: Optional[str] = Field(
        default=None, description="Storage path of this stage's JSON output."
    )
    meta: Dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def mark_running(self) -> None:
        self.status = StageStatus.running
        self.started_at = utc_now()
        self.attempts += 1
        self.error = None

    def mark_done(self, *, output_ref: Optional[str] = None, usage: Optional[Usage] = None) -> None:
        self.status = StageStatus.done
        self.finished_at = utc_now()
        self.output_ref = output_ref or self.output_ref
        if usage:
            self.usage.add(usage)

    def mark_failed(self, error: str) -> None:
        self.status = StageStatus.failed
        self.finished_at = utc_now()
        self.error = error

    def invalidate(self) -> None:
        """Reset for a re-run, keeping the attempt count as history."""
        self.status = StageStatus.stale
        self.started_at = None
        self.finished_at = None
        self.output_ref = None
        self.error = None


class RunState(BaseModel):
    run_id: str
    doc_id: str
    filename: str = ""
    status: RunStatus = RunStatus.queued
    config: Dict[str, Any] = Field(default_factory=dict)
    stages: Dict[str, StageRecord] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    error: Optional[str] = None
    result_ref: Optional[str] = None

    # ---- graph helpers -------------------------------------------------

    def add_stage(self, key: str, name: str, depends_on: Optional[List[str]] = None) -> StageRecord:
        record = StageRecord(key=key, name=name, depends_on=depends_on or [])
        self.stages[key] = record
        return record

    def get(self, key: str) -> Optional[StageRecord]:
        return self.stages.get(key)

    def ready(self) -> List[StageRecord]:
        """Stages whose dependencies are all done and which have not run."""
        out: List[StageRecord] = []
        for record in self.stages.values():
            if record.status not in (StageStatus.pending, StageStatus.stale):
                continue
            if all(
                self.stages[dep].status in (StageStatus.done, StageStatus.skipped)
                for dep in record.depends_on
                if dep in self.stages
            ):
                out.append(record)
        return out

    def dependents_of(self, key: str) -> List[str]:
        """Every stage downstream of ``key``, transitively."""
        found: List[str] = []
        frontier = [key]
        while frontier:
            current = frontier.pop()
            for candidate in self.stages.values():
                if current in candidate.depends_on and candidate.key not in found:
                    found.append(candidate.key)
                    frontier.append(candidate.key)
        return found

    def invalidate_from(self, key: str) -> List[str]:
        """Invalidate ``key`` and everything downstream. Returns what was reset.

        This is the whole of scoped retry: OCR never re-runs unless its own node
        is named, so fixing a wrong document type costs cents rather than a full
        re-OCR of the document.
        """
        if key not in self.stages:
            return []
        affected = [key] + self.dependents_of(key)
        for stage_key in affected:
            self.stages[stage_key].invalidate()
        self.status = RunStatus.queued
        self.error = None
        self.touch()
        return affected

    # ---- rollups -------------------------------------------------------

    @property
    def usage(self) -> Usage:
        total = Usage()
        for record in self.stages.values():
            total.add(record.usage)
        return total

    @property
    def is_terminal(self) -> bool:
        return self.status in (RunStatus.completed, RunStatus.failed, RunStatus.cancelled)

    def progress(self) -> Dict[str, int]:
        counts: Dict[str, int] = {status.value: 0 for status in StageStatus}
        for record in self.stages.values():
            counts[record.status.value] += 1
        return counts

    def touch(self) -> None:
        self.updated_at = utc_now()
