"""Preprocessing: hash, split, and scan for quality.

Deterministic, no model calls. Quality problems are *flagged and reported*, never
blocking - a low-resolution page still goes through OCR, the reviewer is just
told which pages to look at twice.

The content hash computed here is the OCR cache key, which is what makes a
rerun near-free.
"""

from __future__ import annotations

import hashlib
import io
from typing import List

from src.core.errors.exceptions import UnsupportedDocumentError, ValidationError
from src.core.logging.structured import get_logger
from src.schemas.documents import PageQuality, PreprocessResult

logger = get_logger(__name__)

# A4 at 72pt/inch is 595x842. Much smaller than this and a scan will OCR poorly.
_MIN_POINTS = 400
_MIN_TEXT_CHARS = 40


def compute_doc_id(content: bytes) -> str:
    """Content hash - the cache key OCR output is stored under."""
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def split_pdf_pages(pdf_bytes: bytes) -> List[bytes]:
    """Split a PDF into one single-page PDF per page, in memory.

    Uses pypdf rather than rasterising, so no poppler or tesseract is needed and
    each page keeps whatever it actually is - text layer, scan, or both.
    """
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(io.BytesIO(pdf_bytes))
    out: List[bytes] = []
    for page in reader.pages:
        writer = PdfWriter()
        writer.add_page(page)
        buffer = io.BytesIO()
        writer.write(buffer)
        out.append(buffer.getvalue())
    return out


def run(
    content: bytes,
    filename: str,
    *,
    max_pages: int = 300,
    max_mb: int = 50,
) -> PreprocessResult:
    """Hash, inspect and quality-scan an uploaded document.

    Args:
        content: Raw file bytes.
        filename: Original filename, kept for display.
        max_pages: Refuse documents longer than this.
        max_mb: Refuse documents larger than this.

    Returns:
        A :class:`PreprocessResult` with per-page quality flags.

    Raises:
        ValidationError: If the document exceeds a hard limit.
        UnsupportedDocumentError: If it cannot be parsed as a PDF.
    """
    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValidationError(f"document is {size_mb:.1f} MB, over the {max_mb} MB limit")

    doc_id = compute_doc_id(content)

    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        raw_pages = reader.pages
    except Exception as exc:
        raise UnsupportedDocumentError(f"could not read {filename!r} as a PDF: {exc}") from exc

    if len(raw_pages) > max_pages:
        raise ValidationError(f"document has {len(raw_pages)} pages, over the {max_pages} limit")

    quality: List[PageQuality] = []
    for index, page in enumerate(raw_pages, start=1):
        warnings: List[str] = []
        try:
            box = page.mediabox
            width = int(float(box.width))
            height = int(float(box.height))
        except Exception:
            width = height = 0
            warnings.append("unreadable_page_box")

        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""

        has_text = len(text) >= _MIN_TEXT_CHARS
        blank = not has_text and width > 0

        if width and height and min(width, height) < _MIN_POINTS:
            warnings.append("low_resolution")
        if not has_text:
            warnings.append("no_text_layer")
        if blank and width * height > 0 and not text:
            # No text layer AND nothing extractable - either a scan or genuinely
            # empty. OCR will tell us which; flag it either way.
            warnings.append("blank_or_scanned")

        quality.append(
            PageQuality(
                page=index,
                width=width,
                height=height,
                has_text_layer=has_text,
                blank=blank,
                warnings=warnings,
            )
        )

    notes: List[str] = []
    scanned = [q.page for q in quality if not q.has_text_layer]
    if len(scanned) == len(quality) and quality:
        notes.append("No page has a text layer - this is a scanned document; OCR is required.")
    elif scanned:
        notes.append(f"Pages {scanned} have no text layer and rely entirely on OCR.")

    result = PreprocessResult(
        doc_id=doc_id,
        filename=filename,
        size_bytes=len(content),
        pages=len(quality),
        page_quality=quality,
        proceed=True,
        notes=notes,
    )
    logger.info(
        "preprocess: %s, %d pages, %d flagged",
        filename, result.pages, len(result.flagged_pages),
    )
    return result
