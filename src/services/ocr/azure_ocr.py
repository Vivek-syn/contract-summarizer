"""Azure Document Intelligence OCR.

Uses ``prebuilt-layout`` by default rather than ``prebuilt-read``: layout returns
tables as tables, selection marks and paragraph roles, so clause hierarchy and
signature blocks survive into the text. It costs more per page and is worth it
for contracts.

Boxes come back as 8-number polygons in inches; we normalise to [0, 1] against
the page dimensions so downstream code never has to know the unit.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Tuple

from src.core.config.settings import OCRSettings
from src.core.errors.exceptions import ConfigurationError, OCRError
from src.core.logging.structured import get_logger
from src.schemas.documents import BoundingBox, OCRPage, OCRWord
from src.services.ocr.base import OCRProvider, compute_mean_confidence

logger = get_logger(__name__)


def _polygon_to_box(polygon: Any, width: float, height: float) -> BoundingBox:
    """Normalise a Document Intelligence polygon to a [0, 1] bounding box.

    The SDK returns either a flat ``[x1, y1, x2, y2, ...]`` list or a list of
    point objects depending on version, so handle both.
    """
    xs: List[float] = []
    ys: List[float] = []
    if not polygon:
        return BoundingBox(left=0.0, top=0.0, right=0.0, bottom=0.0)

    first = polygon[0]
    if isinstance(first, (int, float)):
        coords = list(polygon)
        xs = coords[0::2]
        ys = coords[1::2]
    else:
        for point in polygon:
            xs.append(getattr(point, "x", 0.0))
            ys.append(getattr(point, "y", 0.0))

    w = width or 1.0
    h = height or 1.0
    return BoundingBox(
        left=max(min(xs) / w, 0.0),
        top=max(min(ys) / h, 0.0),
        right=min(max(xs) / w, 1.0),
        bottom=min(max(ys) / h, 1.0),
    )


class AzureOCR(OCRProvider):
    """Document Intelligence backed OCR."""

    name = "azure"

    def __init__(self, settings: OCRSettings) -> None:
        if not settings.azure_endpoint or not settings.azure_key:
            raise ConfigurationError(
                "Azure OCR needs OCR_AZURE_ENDPOINT and OCR_AZURE_KEY"
            )
        self.settings = settings
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from azure.ai.documentintelligence import DocumentIntelligenceClient
                from azure.core.credentials import AzureKeyCredential
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ConfigurationError(
                    "azure-ai-documentintelligence is not installed; pip install '.[azure]'"
                ) from exc

            self._client = DocumentIntelligenceClient(
                endpoint=self.settings.azure_endpoint,
                credential=AzureKeyCredential(self.settings.azure_key),
            )
        return self._client

    def _analyze_sync(self, pdf_bytes: bytes) -> Any:
        """Blocking SDK call - the caller runs this in a thread."""
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

        client = self._get_client()
        poller = client.begin_analyze_document(
            model_id=self.settings.azure_model_id,
            body=AnalyzeDocumentRequest(bytes_source=pdf_bytes),
        )
        return poller.result()

    def _build_pages(self, result: Any, page_offset: int = 0) -> List[OCRPage]:
        """Turn an analyze result into our page contract.

        Text is rebuilt from the word list so that every word's ``start``/``end``
        are true offsets into ``page.text`` - the grounding stage depends on that
        invariant holding exactly.
        """
        pages: List[OCRPage] = []
        tables_by_page = self._tables_by_page(result)

        for index, raw_page in enumerate(getattr(result, "pages", []) or [], start=1):
            width = float(getattr(raw_page, "width", 0.0) or 0.0)
            height = float(getattr(raw_page, "height", 0.0) or 0.0)
            page_number = int(getattr(raw_page, "page_number", index)) + page_offset

            words: List[OCRWord] = []
            parts: List[str] = []
            cursor = 0
            for raw_word in getattr(raw_page, "words", []) or []:
                text = raw_word.content or ""
                if not text:
                    continue
                if parts:
                    parts.append(" ")
                    cursor += 1
                start = cursor
                parts.append(text)
                cursor += len(text)
                words.append(
                    OCRWord(
                        text=text,
                        start=start,
                        end=cursor,
                        confidence=float(getattr(raw_word, "confidence", 1.0) or 1.0),
                        box=_polygon_to_box(getattr(raw_word, "polygon", None), width, height),
                    )
                )

            page = OCRPage(
                page=page_number,
                text="".join(parts),
                words=words,
                width=width,
                height=height,
                tables_markdown=tables_by_page.get(page_number, []),
            )
            page.mean_confidence = compute_mean_confidence(page)
            page.low_confidence = page.mean_confidence < self.settings.min_page_confidence
            pages.append(page)

        return pages

    @staticmethod
    def _tables_by_page(result: Any) -> dict:
        """Render each table as markdown, keyed by the page it starts on."""
        out: dict = {}
        for table in getattr(result, "tables", []) or []:
            regions = getattr(table, "bounding_regions", None) or []
            page_number = int(getattr(regions[0], "page_number", 1)) if regions else 1
            rows = int(getattr(table, "row_count", 0) or 0)
            cols = int(getattr(table, "column_count", 0) or 0)
            if not rows or not cols:
                continue
            grid = [["" for _ in range(cols)] for _ in range(rows)]
            for cell in getattr(table, "cells", []) or []:
                r = int(getattr(cell, "row_index", 0) or 0)
                c = int(getattr(cell, "column_index", 0) or 0)
                if r < rows and c < cols:
                    grid[r][c] = (cell.content or "").replace("|", "\\|").replace("\n", " ")
            lines = ["| " + " | ".join(grid[0]) + " |", "| " + " | ".join(["---"] * cols) + " |"]
            lines += ["| " + " | ".join(row) + " |" for row in grid[1:]]
            out.setdefault(page_number, []).append("\n".join(lines))
        return out

    async def analyze_page(self, pdf_bytes: bytes, page_number: int) -> OCRPage:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._analyze_sync, pdf_bytes),
                timeout=self.settings.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return OCRPage(page=page_number, error="azure OCR timed out")
        except Exception as exc:
            logger.warning("azure OCR failed on page %d: %s", page_number, exc)
            return OCRPage(page=page_number, error=f"{type(exc).__name__}: {exc}")

        pages = self._build_pages(result)
        if not pages:
            return OCRPage(page=page_number, error="azure OCR returned no pages")
        page = pages[0]
        page.page = page_number
        return page

    async def analyze_document(self, pdf_bytes: bytes) -> List[OCRPage]:
        """One call for the whole document - cheaper and keeps cross-page layout."""
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._analyze_sync, pdf_bytes),
                timeout=self.settings.timeout_seconds * 3,
            )
        except Exception as exc:
            raise OCRError(f"azure OCR failed for document: {exc}") from exc
        return self._build_pages(result)

    async def health(self) -> bool:
        try:
            self._get_client()
            return True
        except Exception:
            return False
