"""Consolidation: assemble the result JSON and render the review markdown.

No model call. The JSON is the complete record that the frontend and scoped
retry work from; the markdown is a curated view of it for a human reviewer.

The review checklist comes from the document type's config, so an offer letter
is not signed off against a checklist written for an MSA.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.schemas.documents import (
    AnalysisResult,
    ConsolidatedResult,
    DocumentType,
    GroundingResult,
    OCRResult,
    PreprocessResult,
    Risk,
    SegmentationResult,
    ValidationResult,
)

_SEVERITY_ORDER = {Risk.high: 0, Risk.medium: 1, Risk.low: 2}
_MATCH_MARK = {"exact": "OK", "fuzzy": "~", "missing": "!!"}

_DEFAULT_CHECKLIST = [
    "Parties, dates and amounts checked against the document",
    "Every high-risk clause read in full",
    "Unverified quotes resolved",
    "Findings accepted or overridden",
]


def build(
    *,
    run_id: str,
    preprocess: PreprocessResult,
    doc_type: DocumentType,
    ocr: OCRResult,
    analysis: AnalysisResult,
    grounding: Optional[GroundingResult],
    segmentation: Optional[SegmentationResult],
    validation: Optional[ValidationResult],
    usage: Optional[Dict[str, float]] = None,
) -> ConsolidatedResult:
    """Assemble every stage output into the run's result."""
    return ConsolidatedResult(
        run_id=run_id,
        doc_id=preprocess.doc_id,
        filename=preprocess.filename,
        preprocess=preprocess,
        document_type=doc_type,
        ocr_provider=ocr.provider,
        ocr_cached=ocr.cached,
        failed_pages=ocr.failed_pages,
        low_confidence_pages=ocr.low_confidence_pages,
        analysis=analysis,
        grounding=grounding,
        segmentation=segmentation,
        validation=validation,
        grounding_rate=grounding.rate if grounding else 0.0,
        needs_human_review=True,
        usage=usage or {},
    )


def render_markdown(result: ConsolidatedResult, *, type_config: Optional[dict] = None) -> str:
    """Render the reviewer's packet.

    Everything that could mislead a reader into trusting an incomplete packet is
    surfaced at the top: failed pages, poor scans, an uncertain document type.
    """
    out: List[str] = []
    summary = result.segmentation
    title = summary.title if summary else result.filename

    out.append(f"# Review packet - {title}\n")
    out.append(
        f"> `{result.filename}` | type **{result.document_type.type}** "
        f"(confidence {result.document_type.confidence:.0%}) | OCR `{result.ocr_provider}`"
        f"{' (cached)' if result.ocr_cached else ''} | grounding {result.grounding_rate:.0%}"
        f" | ${result.usage.get('cost_usd', 0.0):.3f}\n"
    )

    # --- warnings that change how the packet should be read ---------------
    if result.failed_pages:
        out.append(
            f"**Pages {result.failed_pages} could not be read. Any clause on those pages is "
            f"absent from this packet - read them yourself.**\n"
        )
    if result.low_confidence_pages:
        out.append(
            f"**Pages {result.low_confidence_pages} scanned poorly; text from them may be "
            f"misread.**\n"
        )
    if result.document_type.confidence < 0.7:
        out.append(
            f"**Document type is uncertain ({result.document_type.confidence:.0%}). If "
            f"`{result.document_type.type}` is wrong, the summary sections and the playbook "
            f"are both wrong - re-run from type detection.**\n"
        )

    if summary:
        out.append(f"**Parties:** {', '.join(summary.parties) or '-'}  ")
        out.append(f"**Effective:** {summary.effective_date or '-'}  |  **Term:** {summary.term or '-'}  ")
        out.append(f"**Governing law:** {summary.governing_law or '-'}\n")
        out.append(f"## Executive summary\n\n{summary.executive_summary}\n")

        for section in summary.sections:
            if not section.bullets:
                continue
            out.append(f"### {section.heading}")
            out += [f"- {b}" for b in section.bullets]
            if section.clause_ids:
                out.append(f"\n<sub>from: {', '.join(section.clause_ids)}</sub>")
            out.append("")

    # --- findings --------------------------------------------------------
    if result.validation and result.validation.findings:
        out.append("## Playbook findings\n")
        out.append("| Rule | Status | Severity | Explanation |")
        out.append("|---|---|---|---|")
        for finding in sorted(
            result.validation.findings, key=lambda f: _SEVERITY_ORDER[f.severity]
        ):
            explanation = finding.explanation.replace("|", "\\|").replace("\n", " ")
            out.append(
                f"| {finding.rule_id} {finding.rule_title} | {finding.status.value} "
                f"| {finding.severity.value} | {explanation} |"
            )
        out.append(
            f"\n**Overall: {result.validation.overall}** "
            f"({result.validation.blocking_count} blocking)\n"
        )

        redlines = [f for f in result.validation.findings if f.suggested_redline]
        if redlines:
            out.append("## Suggested redlines\n")
            out += [f"- **{f.rule_id}**: {f.suggested_redline}" for f in redlines]
            out.append("")

    # --- evidence --------------------------------------------------------
    grounded = {g.clause_id: g for g in (result.grounding.clauses if result.grounding else [])}
    out.append("## Clause evidence\n")
    for clause in sorted(result.analysis.clauses, key=lambda c: _SEVERITY_ORDER[c.risk]):
        ground = grounded.get(clause.id)
        mark = _MATCH_MARK[ground.match] if ground else "?"
        flag = "" if (ground and ground.verified) else " :warning: UNVERIFIED"
        out.append(f"### {clause.heading} - {clause.category} ({clause.risk.value} risk){flag}")
        out.append(f"{clause.summary}\n")
        out.append(f"> [{mark} p.{clause.page}] {clause.quote}")
        if ground and ground.min_confidence < 0.8:
            out.append(
                f"\n<sub>OCR confidence on this text is {ground.min_confidence:.0%} - verify it.</sub>"
            )
        out.append("")

    if summary and summary.open_questions:
        out.append("## Open questions\n")
        out += [f"- [ ] {q}" for q in summary.open_questions]
        out.append("")

    checklist = (type_config or {}).get("review_checklist") or _DEFAULT_CHECKLIST
    out.append("---\n## Human validation\n")
    out += [f"- [ ] {item}" for item in checklist]
    out.append("\nReviewer: ____________________   Date: ____________")
    return "\n".join(out)
