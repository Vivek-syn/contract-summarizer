"""Segmentation: a type-aware structured summary.

Reads the grounded clause set, not the document. A clause whose quote could not
be located is marked unreliable in the prompt, so the summary does not quietly
inherit something the grounding stage already flagged.

The section taxonomy comes from the document type's config rather than a prose
list in a prompt, and returned headings are validated against it. That is what
makes "a contract gets clauses, a charge sheet gets evidence" a config change
rather than a code change.
"""

from __future__ import annotations

from typing import List, Optional

from src.core.logging.structured import get_logger
from src.schemas.documents import (
    AnalysisResult,
    DocumentType,
    GroundingResult,
    SegmentationResult,
)
from src.services.llm.adapter import LLMAdapter

logger = get_logger(__name__)

SYSTEM = """You write structured summaries of documents for a reviewer who has not
read the original.

Rules:
- Ground every statement in the clauses you are given, and cite their ids.
- Use ONLY the section headings you are given, in that order. Omit a section
  entirely if the document has nothing for it - do not invent new headings.
- State amounts, dates, durations and party names exactly as they appear.
- A clause marked UNVERIFIED could not be located in the document text. Do not
  build a claim on one; if you must mention it, put the uncertainty in
  `open_questions`.
- Anything you could not determine goes in `open_questions` rather than being
  guessed at."""

_DEFAULT_SECTIONS = [
    "Parties & Term",
    "Commercials",
    "Obligations",
    "Liability & Risk",
    "Termination",
    "Other",
]


async def run(
    analysis: AnalysisResult,
    doc_type: DocumentType,
    adapter: LLMAdapter,
    *,
    grounding: Optional[GroundingResult] = None,
    type_config: Optional[dict] = None,
    model: Optional[str] = None,
    feedback: Optional[List[str]] = None,
) -> SegmentationResult:
    """Produce the structured summary.

    Args:
        analysis: Extracted clauses.
        doc_type: Detected type, which selects the section taxonomy.
        adapter: LLM adapter.
        grounding: Grounding output, used to mark unreliable clauses.
        type_config: Per-type config supplying ``sections``.
        model: Model override.
        feedback: Corrections from a rejected previous attempt.

    Returns:
        A :class:`SegmentationResult` whose headings are constrained to the
        configured taxonomy.
    """
    sections: List[str] = list(
        (type_config or {}).get("sections") or _DEFAULT_SECTIONS
    )
    unverified = set(grounding.ungrounded) if grounding else set()

    def _render(clause) -> str:
        status = ' status="UNVERIFIED"' if clause.id in unverified else ""
        return (
            f'<clause id="{clause.id}" page="{clause.page}" '
            f'risk="{clause.risk.value}"{status}>\n'
            f"{clause.heading} ({clause.category})\n{clause.summary}\n"
            f"Rationale: {clause.rationale}\nQuoted: {clause.quote}\n</clause>"
        )

    rendered = "\n\n".join(_render(c) for c in analysis.clauses) or "(no clauses were extracted)"

    user = (
        f"<document_type>{type_config.get('name') if type_config else doc_type.type}"
        f"</document_type>\n\n"
        f"<required_sections>\n"
        + "\n".join(f"- {s}" for s in sections)
        + f"\n</required_sections>\n\n<clauses>\n{rendered}\n</clauses>\n\n"
    )
    if unverified:
        user += (
            f"<warning>\n{len(unverified)} clause(s) are marked UNVERIFIED: their quoted "
            f"text could not be located in the document. Treat them as unreliable.\n</warning>\n\n"
        )
    if feedback:
        user += (
            "<review_feedback>\nA previous attempt was rejected. Fix these:\n"
            + "\n".join(f"- {f}" for f in feedback)
            + "\n</review_feedback>\n\n"
        )
    user += "Write the structured summary."

    result = await adapter.structured(
        SegmentationResult,
        system=SYSTEM,
        user=user,
        tool_name="record_summary",
        tool_description="Record the structured summary of this document.",
        model=model,
    )

    result.document_type = doc_type.type

    # Enforce the taxonomy rather than trusting the prompt to have landed.
    allowed = {s.lower(): s for s in sections}
    kept = []
    for section in result.sections:
        canonical = allowed.get(section.heading.strip().lower())
        if canonical:
            section.heading = canonical
            kept.append(section)
        else:
            logger.warning(
                "segmentation returned off-taxonomy heading %r; folding into 'Other'",
                section.heading,
            )
            section.heading = sections[-1]
            kept.append(section)
    result.sections = kept

    logger.info("segmentation: %d sections", len(result.sections))
    return result
