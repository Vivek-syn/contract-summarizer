"""AWS Textract OCR.

``AnalyzeDocument`` with TABLES and FORMS is the layout-aware equivalent of
Azure's ``prebuilt-layout``; with no feature types it falls back to the cheaper
``DetectDocumentText``.

Textract's synchronous API takes one page at a time, so multi-page work is
fan-out over single-page PDFs. Geometry comes back already normalised to [0, 1],
which is the same convention we use.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from src.core.config.settings import OCRSettings
from src.core.errors.exceptions import ConfigurationError
from src.core.logging.structured import get_logger
from src.schemas.documents import BoundingBox, OCRPage, OCRWord
from src.services.ocr.base import OCRProvider, compute_mean_confidence

logger = get_logger(__name__)


def _bbox(geometry: Dict[str, Any]) -> BoundingBox:
    box = (geometry or {}).get("BoundingBox", {})
    left = float(box.get("Left", 0.0))
    top = float(box.get("Top", 0.0))
    return BoundingBox(
        left=left,
        top=top,
        right=min(left + float(box.get("Width", 0.0)), 1.0),
        bottom=min(top + float(box.get("Height", 0.0)), 1.0),
    )


class AWSOCR(OCRProvider):
    """Textract backed OCR."""

    name = "aws"

    def __init__(self, settings: OCRSettings) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ConfigurationError("boto3 is not installed; pip install '.[aws]'") from exc

        self.settings = settings
        kwargs: Dict[str, str] = {"region_name": settings.aws_region}
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        self._client = boto3.client("textract", **kwargs)

    def _analyze_sync(self, pdf_bytes: bytes) -> Dict[str, Any]:
        document = {"Bytes": pdf_bytes}
        if self.settings.aws_feature_types:
            return self._client.analyze_document(
                Document=document, FeatureTypes=list(self.settings.aws_feature_types)
            )
        return self._client.detect_document_text(Document=document)

    def _build_page(self, response: Dict[str, Any], page_number: int) -> OCRPage:
        """Rebuild page text from WORD blocks, in reading order.

        Textract returns blocks unordered; we sort by line then word so the
        reconstructed text reads naturally and the offsets stay meaningful.
        """
        blocks = response.get("Blocks", [])
        by_id = {b["Id"]: b for b in blocks}

        ordered_words: List[Dict[str, Any]] = []
        line_blocks = [b for b in blocks if b.get("BlockType") == "LINE"]
        line_blocks.sort(
            key=lambda b: (
                round(b.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0.0), 3),
                b.get("Geometry", {}).get("BoundingBox", {}).get("Left", 0.0),
            )
        )
        seen: set = set()
        for line in line_blocks:
            for rel in line.get("Relationships", []) or []:
                if rel.get("Type") != "CHILD":
                    continue
                for child_id in rel.get("Ids", []):
                    child = by_id.get(child_id)
                    if child and child.get("BlockType") == "WORD" and child_id not in seen:
                        seen.add(child_id)
                        ordered_words.append(child)
            ordered_words.append({"BlockType": "_NEWLINE"})

        words: List[OCRWord] = []
        parts: List[str] = []
        cursor = 0
        for block in ordered_words:
            if block.get("BlockType") == "_NEWLINE":
                if parts and not parts[-1].endswith("\n"):
                    parts.append("\n")
                    cursor += 1
                continue
            text = block.get("Text", "")
            if not text:
                continue
            if parts and not parts[-1].endswith(("\n", " ")):
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
                    confidence=float(block.get("Confidence", 100.0)) / 100.0,
                    box=_bbox(block.get("Geometry", {})),
                )
            )

        page = OCRPage(
            page=page_number,
            text="".join(parts).strip(),
            words=words,
            tables_markdown=self._tables(blocks, by_id),
        )
        page.mean_confidence = compute_mean_confidence(page)
        page.low_confidence = page.mean_confidence < self.settings.min_page_confidence
        return page

    @staticmethod
    def _tables(blocks: List[Dict[str, Any]], by_id: Dict[str, Any]) -> List[str]:
        """Render TABLE blocks as markdown."""
        out: List[str] = []
        for table in [b for b in blocks if b.get("BlockType") == "TABLE"]:
            cells: Dict[tuple, str] = {}
            max_row = max_col = 0
            for rel in table.get("Relationships", []) or []:
                if rel.get("Type") != "CHILD":
                    continue
                for cell_id in rel.get("Ids", []):
                    cell = by_id.get(cell_id)
                    if not cell or cell.get("BlockType") != "CELL":
                        continue
                    row = int(cell.get("RowIndex", 1)) - 1
                    col = int(cell.get("ColumnIndex", 1)) - 1
                    max_row, max_col = max(max_row, row), max(max_col, col)
                    words: List[str] = []
                    for sub in cell.get("Relationships", []) or []:
                        for wid in sub.get("Ids", []):
                            word = by_id.get(wid)
                            if word and word.get("BlockType") == "WORD":
                                words.append(word.get("Text", ""))
                    cells[(row, col)] = " ".join(words).replace("|", "\\|")
            if not cells:
                continue
            grid = [
                [cells.get((r, c), "") for c in range(max_col + 1)] for r in range(max_row + 1)
            ]
            lines = ["| " + " | ".join(grid[0]) + " |",
                     "| " + " | ".join(["---"] * (max_col + 1)) + " |"]
            lines += ["| " + " | ".join(row) + " |" for row in grid[1:]]
            out.append("\n".join(lines))
        return out

    async def analyze_page(self, pdf_bytes: bytes, page_number: int) -> OCRPage:
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self._analyze_sync, pdf_bytes),
                timeout=self.settings.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return OCRPage(page=page_number, error="textract timed out")
        except Exception as exc:
            logger.warning("textract failed on page %d: %s", page_number, exc)
            return OCRPage(page=page_number, error=f"{type(exc).__name__}: {exc}")
        return self._build_page(response, page_number)

    async def health(self) -> bool:
        return self._client is not None
