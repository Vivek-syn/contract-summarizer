"""Stage contracts.

Every pipeline stage is a pure function of JSON in to JSON out; these are those
JSON shapes. They double as the schemas handed to the model as tool definitions,
so field descriptions are prompt surface - keep them precise.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# preprocess
# --------------------------------------------------------------------------

class PageQuality(BaseModel):
    """Per-page quality signals. Flags, never gates - we always go ahead."""

    page: int
    width: int = 0
    height: int = 0
    has_text_layer: bool = False
    estimated_dpi: Optional[int] = None
    blank: bool = False
    warnings: List[str] = Field(
        default_factory=list,
        description="e.g. low_resolution, no_text_layer, blank_page, oversized",
    )

    @property
    def ok(self) -> bool:
        return not self.warnings


class PreprocessResult(BaseModel):
    doc_id: str = Field(description="sha256 of the file bytes; the OCR cache key")
    filename: str
    content_type: str = "application/pdf"
    size_bytes: int = 0
    pages: int = 0
    page_quality: List[PageQuality] = Field(default_factory=list)
    proceed: bool = True
    notes: List[str] = Field(default_factory=list)

    @property
    def flagged_pages(self) -> List[int]:
        return [p.page for p in self.page_quality if p.warnings]


# --------------------------------------------------------------------------
# ocr
# --------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Normalised to [0, 1] against page width/height, origin top-left."""

    left: float
    top: float
    right: float
    bottom: float

    def union(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox(
            left=min(self.left, other.left),
            top=min(self.top, other.top),
            right=max(self.right, other.right),
            bottom=max(self.bottom, other.bottom),
        )


class OCRWord(BaseModel):
    """One word with its offset into the page text and its box on the page."""

    text: str
    start: int = Field(description="Char offset into OCRPage.text")
    end: int
    confidence: float = 1.0
    box: BoundingBox


class OCRPage(BaseModel):
    page: int
    text: str = ""
    words: List[OCRWord] = Field(default_factory=list)
    mean_confidence: float = 1.0
    low_confidence: bool = False
    width: float = 0.0
    height: float = 0.0
    tables_markdown: List[str] = Field(
        default_factory=list, description="Tables rendered as markdown, layout models only"
    )
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class OCRResult(BaseModel):
    doc_id: str
    provider: Literal["azure", "aws", "model_fallback", "text_layer", "none"]
    pages: List[OCRPage] = Field(default_factory=list)
    cached: bool = False

    @property
    def failed_pages(self) -> List[int]:
        return [p.page for p in self.pages if not p.ok]

    @property
    def low_confidence_pages(self) -> List[int]:
        return [p.page for p in self.pages if p.low_confidence]

    def full_text(self) -> str:
        """Page-marked text. Offsets from :meth:`page_offsets` index into this."""
        return "".join(f"\n\n[page {p.page}]\n{p.text}" for p in self.pages if p.ok)

    def page_offsets(self) -> Dict[int, int]:
        """Start offset of each page inside :meth:`full_text`."""
        offsets: Dict[int, int] = {}
        cursor = 0
        for page in self.pages:
            if not page.ok:
                continue
            block = f"\n\n[page {page.page}]\n{page.text}"
            offsets[page.page] = cursor + len(block) - len(page.text)
            cursor += len(block)
        return offsets


# --------------------------------------------------------------------------
# type detection
# --------------------------------------------------------------------------

class DocumentType(BaseModel):
    """A routing decision. Wrong here means everything downstream is wrong."""

    type: str = Field(description="Slug from the configured document types, or 'unknown'.")
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: List[str] = Field(default_factory=list)
    evidence: str = Field(default="", description="What in the text decided it.")
    evidence_page: int = 1
    overridden_by_user: bool = False


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

class Risk(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Clause(BaseModel):
    id: str = Field(description="Stable slug, e.g. 'limitation-of-liability-1'.")
    category: str = Field(description="Clause type, e.g. Indemnity, Termination, Notice Period.")
    heading: str = Field(description="Short human label.")
    summary: str = Field(description="One or two sentences in plain English.")
    risk: Risk
    rationale: str = Field(description="Why this risk level, referencing the quoted text.")
    quote: str = Field(description="Verbatim text copied from the document. Never paraphrase.")
    page: int = Field(description="1-indexed page the quote appears on.")
    # Filled by the grounding stage, not the model.
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class AnalysisResult(BaseModel):
    clauses: List[Clause] = Field(
        description="Every important clause found. Do not omit unfavourable ones."
    )


# --------------------------------------------------------------------------
# grounding
# --------------------------------------------------------------------------

class GroundedRegion(BaseModel):
    page: int
    boxes: List[BoundingBox] = Field(
        default_factory=list, description="One box per line, since words wrap."
    )


class GroundedClause(BaseModel):
    clause_id: str
    match: Literal["exact", "fuzzy", "missing"] = "missing"
    score: float = 0.0
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    regions: List[GroundedRegion] = Field(default_factory=list)
    min_confidence: float = Field(
        default=1.0, description="Lowest OCR word confidence under this clause."
    )

    @property
    def verified(self) -> bool:
        return self.match != "missing"


class GroundingResult(BaseModel):
    clauses: List[GroundedClause] = Field(default_factory=list)

    @property
    def rate(self) -> float:
        if not self.clauses:
            return 0.0
        return sum(c.verified for c in self.clauses) / len(self.clauses)

    @property
    def ungrounded(self) -> List[str]:
        return [c.clause_id for c in self.clauses if not c.verified]


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------

class SummarySection(BaseModel):
    heading: str
    bullets: List[str] = Field(description="Each bullet is a standalone fact.")
    clause_ids: List[str] = Field(default_factory=list)


class SegmentationResult(BaseModel):
    title: str
    parties: List[str] = Field(default_factory=list)
    document_type: str = ""
    effective_date: Optional[str] = None
    term: Optional[str] = None
    governing_law: Optional[str] = None
    executive_summary: str = Field(description="4-6 sentences a busy reviewer can read alone.")
    sections: List[SummarySection] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

class FindingStatus(str, Enum):
    compliant = "compliant"
    deviation = "deviation"
    missing = "missing"
    unclear = "unclear"


class Finding(BaseModel):
    rule_id: str
    rule_title: str
    status: FindingStatus
    severity: Risk
    clause_ids: List[str] = Field(default_factory=list)
    explanation: str
    suggested_redline: Optional[str] = None


class ValidationResult(BaseModel):
    findings: List[Finding] = Field(default_factory=list)
    blocking_count: int = 0
    overall: Literal["pass", "pass_with_comments", "fail"] = "pass_with_comments"


# --------------------------------------------------------------------------
# consolidate
# --------------------------------------------------------------------------

class ConsolidatedResult(BaseModel):
    """What the API returns and the markdown renderer reads."""

    run_id: str
    doc_id: str
    filename: str
    preprocess: PreprocessResult
    document_type: DocumentType
    ocr_provider: str
    ocr_cached: bool = False
    failed_pages: List[int] = Field(default_factory=list)
    low_confidence_pages: List[int] = Field(default_factory=list)
    analysis: AnalysisResult
    grounding: Optional[GroundingResult] = None
    segmentation: Optional[SegmentationResult] = None
    validation: Optional[ValidationResult] = None
    grounding_rate: float = 0.0
    needs_human_review: bool = True
    usage: Dict[str, float] = Field(default_factory=dict)
