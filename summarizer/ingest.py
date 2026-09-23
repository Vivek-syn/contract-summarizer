"""Stage 1: User Doc -> PDF parser / OCR -> text.

Order of preference, matching the pipeline design:

1. ``extracted_text`` - embedded text layer from the PDF. Cheapest and exact,
   so spans can be located by character offset.
2. ``ocr`` - only when a page has no usable text layer (scanned contract).
3. ``whole_doc`` - hand the raw PDF to Claude as a document block and let the
   model read it natively. Last resort: no character offsets, so grounding
   degrades to fuzzy quote matching.
"""
from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

from .config import SETTINGS

log = logging.getLogger("summarizer.ingest")

Mode = Literal["extracted_text", "ocr", "whole_doc"]


@dataclass
class Page:
    number: int  # 1-indexed
    text: str
    source: Literal["text_layer", "ocr"]
    offset: int = 0  # start index of this page inside Document.text


@dataclass
class Document:
    path: Path
    pages: List[Page]
    mode: Mode
    raw_bytes: bytes | None = None

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.pages)

    def page_at(self, char_index: int) -> int:
        for page in reversed(self.pages):
            if char_index >= page.offset:
                return page.number
        return 1

    def as_message_content(self) -> List[Dict[str, Any]]:
        """User-message content blocks for this document.

        In ``whole_doc`` mode LiteLLM converts the ``file`` block into an
        Anthropic ``document`` block, so Claude reads the PDF itself.
        """
        if self.mode == "whole_doc" and self.raw_bytes:
            encoded = base64.b64encode(self.raw_bytes).decode()
            return [
                {
                    "type": "file",
                    "file": {
                        "file_data": f"data:application/pdf;base64,{encoded}",
                        "filename": self.path.name,
                    },
                },
                {"type": "text", "text": "The contract is attached above."},
            ]
        return [{"type": "text", "text": f"<document name=\"{self.path.name}\">\n{self.text}\n</document>"}]


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _ocr_page(pdf_path: Path, page_number: int) -> str:
    """OCR one page. Requires poppler + tesseract on PATH."""
    from pdf2image import convert_from_path  # imported lazily: optional dependency
    import pytesseract

    images = convert_from_path(str(pdf_path), first_page=page_number, last_page=page_number, dpi=300)
    return pytesseract.image_to_string(images[0]) if images else ""


def load(path: str | Path, *, allow_ocr: bool = True, force_whole_doc: bool = False) -> Document:
    path = Path(path)
    raw = path.read_bytes()

    if path.suffix.lower() in {".txt", ".md"}:
        text = _normalise(raw.decode("utf-8", errors="replace"))
        return Document(path=path, pages=[Page(1, text, "text_layer")], mode="extracted_text")

    if force_whole_doc:
        return Document(path=path, pages=[], mode="whole_doc", raw_bytes=raw)

    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf not installed - falling back to whole-doc mode")
        return Document(path=path, pages=[], mode="whole_doc", raw_bytes=raw)

    reader = PdfReader(str(path))
    pages: List[Page] = []
    ocr_used = False

    for index, raw_page in enumerate(reader.pages, start=1):
        text = _normalise(raw_page.extract_text() or "")
        source: Literal["text_layer", "ocr"] = "text_layer"
        if len(text) < SETTINGS.min_chars_per_page and allow_ocr:
            try:
                ocr_text = _normalise(_ocr_page(path, index))
                if len(ocr_text) > len(text):
                    text, source, ocr_used = ocr_text, "ocr", True
            except Exception as exc:
                log.warning("OCR failed on page %d (%s: %s)", index, type(exc).__name__, exc)
        pages.append(Page(number=index, text=text, source=source))

    if not any(p.text.strip() for p in pages):
        log.warning("no text recovered from %s - using whole-doc mode", path.name)
        return Document(path=path, pages=[], mode="whole_doc", raw_bytes=raw)

    # Stitch pages into one addressable string with page markers.
    cursor = 0
    stitched: List[Page] = []
    for page in pages:
        block = f"\n\n[page {page.number}]\n{page.text}"
        stitched.append(Page(page.number, block, page.source, offset=cursor))
        cursor += len(block)

    mode: Mode = "ocr" if ocr_used else "extracted_text"
    log.info("ingested %s: %d pages, mode=%s", path.name, len(stitched), mode)
    return Document(path=path, pages=stitched, mode=mode, raw_bytes=raw)
