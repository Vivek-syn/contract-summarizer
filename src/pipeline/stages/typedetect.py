"""Document type detection.

Its own stage on purpose. This is a routing decision: get it wrong and the
segmentation schema is wrong, the playbook is wrong, and every downstream
output is confidently wrong in a way nothing detects.

Kept separate so it can be retried on its own, cached, and overridden by a user
in one click - none of which is possible when it is folded into analysis. It
reads only the first two pages and runs on the cheap model.
"""

from __future__ import annotations

from typing import List, Optional

from src.core.logging.structured import get_logger
from src.schemas.documents import DocumentType, OCRResult
from src.services.llm.adapter import LLMAdapter

logger = get_logger(__name__)

SYSTEM = """You classify legal and business documents by type.

You see only the opening pages, which is deliberate - the title, parties and
recitals are what identify a document, and they live there.

Rules:
- Answer with one of the types you are given, or `unknown`. Never invent a type.
- `confidence` is how sure you are the whole document is that type, not how sure
  you are about the words you can see.
- If the opening pages could plausibly belong to two types, say so in
  `alternatives` and lower the confidence rather than picking arbitrarily. A
  wrong type here silently corrupts everything downstream, so an honest
  `unknown` is far better than a confident guess.
- `evidence` quotes or paraphrases the specific words that decided it."""

PAGES_TO_READ = 2


async def run(
    ocr: OCRResult,
    known_types: List[dict],
    adapter: LLMAdapter,
    *,
    model: Optional[str] = None,
    override: Optional[str] = None,
) -> DocumentType:
    """Identify the document type from its opening pages.

    Args:
        ocr: OCR output for the document.
        known_types: Configured types, each with ``slug``, ``name``, ``hints``.
        adapter: LLM adapter.
        model: Override the model; defaults to the configured fast model.
        override: A user-supplied type, which short-circuits the call entirely.

    Returns:
        A :class:`DocumentType`.
    """
    if override:
        logger.info("document type overridden by user: %s", override)
        return DocumentType(
            type=override, confidence=1.0, evidence="Set by the user.", overridden_by_user=True
        )

    head = [p for p in ocr.pages if p.ok][:PAGES_TO_READ]
    if not head:
        return DocumentType(type="unknown", confidence=0.0, evidence="No readable pages.")

    catalogue = "\n".join(
        f"- {t['slug']}: {t.get('name', t['slug'])}"
        + (f" — {t['hints']}" if t.get("hints") else "")
        for t in known_types
    )
    excerpt = "\n\n".join(f"[page {p.page}]\n{p.text[:4000]}" for p in head)

    user = (
        f"<known_types>\n{catalogue}\n- unknown: none of the above\n</known_types>\n\n"
        f"<document_opening>\n{excerpt}\n</document_opening>\n\n"
        "Classify this document."
    )

    result = await adapter.structured(
        DocumentType,
        system=SYSTEM,
        user=user,
        tool_name="record_document_type",
        tool_description="Record the document's type, confidence and the evidence for it.",
        model=model or adapter.settings.fast_model,
        max_tokens=4000,
    )

    valid = {t["slug"] for t in known_types} | {"unknown"}
    if result.type not in valid:
        logger.warning("model returned unknown type %r; coercing to 'unknown'", result.type)
        result = result.model_copy(
            update={
                "type": "unknown",
                "confidence": 0.0,
                "evidence": f"Model proposed {result.type!r}, which is not a configured type.",
            }
        )

    result.evidence_page = head[0].page
    logger.info("document type: %s (confidence %.2f)", result.type, result.confidence)
    return result
