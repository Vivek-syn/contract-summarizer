"""Analysis: OCR text in, important clauses out.

The model reads OCR text, never the PDF. That separation is the point of the
whole redesign: the transcript is written by one system and the quotes by
another, so the grounding stage is checking one against the other rather than
marking its own homework.

Quotes must be copied from the OCR text exactly, because grounding locates them
by character offset in that same text.
"""

from __future__ import annotations

from typing import List, Optional

from src.core.logging.structured import get_logger
from src.schemas.documents import AnalysisResult, DocumentType, OCRResult
from src.services.llm.adapter import LLMAdapter

logger = get_logger(__name__)

SYSTEM = """You are a contract analyst. You extract clauses; you never invent them.

Rules:
- Every clause carries a `quote` copied VERBATIM from the document text you are
  given, character for character. A paraphrased quote is a failure: the quote is
  located by exact search afterwards, and one that cannot be found is reported
  to the reviewer as unsupported.
- Quote the shortest span that carries the obligation, not a whole section.
- `page` is the page number shown in the `[page N]` marker above the quoted text.
- Cover the whole document, including clauses that are unfavourable, unusual or
  missing-by-implication. Omitting a bad clause is the worst error you can make.
- Judge `risk` against the document as a whole, not the clause in isolation.
- If something is genuinely absent, do not fabricate a clause for it - the
  validation stage handles missing-clause findings."""


def _type_guidance(doc_type: DocumentType, type_config: Optional[dict]) -> str:
    if not type_config:
        return ""
    focus = type_config.get("focus") or type_config.get("categories") or []
    if not focus:
        return ""
    joined = ", ".join(focus) if isinstance(focus, list) else str(focus)
    return (
        f"\n\nThis document was classified as **{type_config.get('name', doc_type.type)}** "
        f"(confidence {doc_type.confidence:.2f}). For this type, the clauses that matter most "
        f"are: {joined}. Extract others you find as well - this is emphasis, not a whitelist."
    )


async def run(
    ocr: OCRResult,
    doc_type: DocumentType,
    adapter: LLMAdapter,
    *,
    type_config: Optional[dict] = None,
    model: Optional[str] = None,
    feedback: Optional[List[str]] = None,
) -> AnalysisResult:
    """Extract the important clauses from OCR text.

    Args:
        ocr: OCR output; its ``full_text()`` is what the model reads.
        doc_type: The detected type, which steers emphasis.
        adapter: LLM adapter.
        type_config: Per-type config, if one exists for this document type.
        model: Model override.
        feedback: Correction instructions from a previous rejected attempt.

    Returns:
        An :class:`AnalysisResult` whose quotes are expected to be verbatim.
    """
    text = ocr.full_text()
    if not text.strip():
        logger.warning("analysis: OCR produced no text")
        return AnalysisResult(clauses=[])

    notes: List[str] = []
    if ocr.low_confidence_pages:
        notes.append(
            f"Pages {ocr.low_confidence_pages} scanned poorly and their text may be garbled. "
            "Quote them only where you can read the words with confidence."
        )
    if ocr.failed_pages:
        notes.append(f"Pages {ocr.failed_pages} could not be read at all and are absent below.")

    user = f"<document>\n{text}\n</document>"
    if notes:
        user += "\n\n<ocr_notes>\n" + "\n".join(f"- {n}" for n in notes) + "\n</ocr_notes>"
    user += _type_guidance(doc_type, type_config)
    if feedback:
        user += (
            "\n\n<review_feedback>\nA previous attempt was rejected. Fix these:\n"
            + "\n".join(f"- {f}" for f in feedback)
            + "\n</review_feedback>"
        )
    user += "\n\nExtract every important clause."

    result = await adapter.structured(
        AnalysisResult,
        system=SYSTEM,
        user=user,
        tool_name="record_clauses",
        tool_description="Record every important clause with its verbatim supporting quote.",
        model=model,
    )

    # Ids must be unique - grounding, segmentation and validation all key on them.
    seen: dict = {}
    for clause in result.clauses:
        base = clause.id or "clause"
        if base in seen:
            seen[base] += 1
            clause.id = f"{base}-{seen[base]}"
        else:
            seen[base] = 1

    logger.info("analysis: %d clauses", len(result.clauses))
    return result
