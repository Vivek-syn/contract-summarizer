"""Pydantic contracts for every stage of the pipeline.

These double as the JSON Schemas handed to Claude as strict tool definitions,
so every field description is prompt surface - keep them precise.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Risk(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Span(BaseModel):
    """A verbatim pointer back into the source document."""

    quote: str = Field(description="Exact verbatim text copied from the document. Never paraphrase.")
    page: int = Field(description="1-indexed page the quote appears on.")
    char_start: Optional[int] = Field(default=None, description="Filled in by the grounding verifier, not the model.")
    char_end: Optional[int] = Field(default=None, description="Filled in by the grounding verifier, not the model.")
    verified: bool = Field(default=False, description="Set by the grounding verifier once the quote is located.")
    match: Literal["exact", "fuzzy", "missing"] = "missing"


class Clause(BaseModel):
    id: str = Field(description="Stable slug, e.g. 'limitation-of-liability-1'.")
    category: str = Field(description="Clause type, e.g. Indemnity, Termination, Confidentiality, Payment Terms.")
    heading: str = Field(description="Short human label for the clause.")
    summary: str = Field(description="One or two sentences in plain English.")
    risk: Risk = Field(description="Risk to the party the playbook is written for.")
    rationale: str = Field(description="Why this risk level, referencing the quoted text.")
    spans: List[Span] = Field(description="At least one verbatim span supporting the summary.")


class ClauseSet(BaseModel):
    clauses: List[Clause] = Field(description="Every important clause found. Do not omit unfavourable ones.")


class SummarySection(BaseModel):
    heading: str
    bullets: List[str] = Field(description="Each bullet is a standalone fact.")
    clause_ids: List[str] = Field(default_factory=list, description="Clause ids this section is derived from.")


class StructuredSummary(BaseModel):
    title: str
    parties: List[str]
    agreement_type: str
    effective_date: Optional[str] = None
    term: Optional[str] = None
    governing_law: Optional[str] = None
    executive_summary: str = Field(description="4-6 sentences a busy reviewer can read alone.")
    sections: List[SummarySection]
    open_questions: List[str] = Field(default_factory=list)


class FindingStatus(str, Enum):
    compliant = "compliant"
    deviation = "deviation"
    missing = "missing"
    unclear = "unclear"


class PlaybookFinding(BaseModel):
    rule_id: str
    rule_title: str
    status: FindingStatus
    severity: Risk
    clause_ids: List[str] = Field(default_factory=list)
    explanation: str
    suggested_redline: Optional[str] = None


class ValidationReport(BaseModel):
    findings: List[PlaybookFinding]
    blocking_count: int = 0
    overall: Literal["pass", "pass_with_comments", "fail"] = "pass_with_comments"


class JudgeIssue(BaseModel):
    stage: Literal["clauses", "summary", "validation"]
    severity: Risk
    detail: str
    fix_instruction: str = Field(description="Imperative instruction the next attempt should follow.")


class JudgeVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0 = unusable, 1 = ready for human review.")
    grounded: bool = Field(description="True only if every material claim traces to verified quoted text.")
    coverage: float = Field(ge=0.0, le=1.0, description="Fraction of important clauses that were captured.")
    issues: List[JudgeIssue] = Field(default_factory=list)
    verdict: Literal["accept", "revise"] = "revise"


class PipelineResult(BaseModel):
    source: str
    ingest_mode: Literal["extracted_text", "ocr", "whole_doc"]
    clauses: ClauseSet
    summary: StructuredSummary
    validation: ValidationReport
    judge: JudgeVerdict
    attempts: int
    grounding_rate: float
    needs_human_review: bool
    usd_cost: float = 0.0
