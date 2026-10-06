"""Abstract OCR interface.

The contract every provider must satisfy is narrow but strict: return the page
text **verbatim**, plus a character-offset-addressable word list with boxes and
confidences. Those offsets are what lets grounding turn a quoted clause into a
coordinate on the page, so a provider that returns text without offsets is not
usable here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.schemas.documents import OCRPage


class OCRProvider(ABC):
    """Base class for OCR backends."""

    name: str = "base"

    @abstractmethod
    async def analyze_page(self, pdf_bytes: bytes, page_number: int) -> OCRPage:
        """OCR a single-page PDF.

        Args:
            pdf_bytes: A PDF containing exactly one page.
            page_number: The page's 1-indexed number in the source document,
                used only to label the result.

        Returns:
            An :class:`OCRPage`. A provider-level failure is reported by setting
            ``error`` rather than raising, so one bad page does not kill a run.
        """

    async def analyze_document(self, pdf_bytes: bytes) -> List[OCRPage]:
        """OCR a whole document in one call where the provider supports it.

        The default splits and falls back to :meth:`analyze_page`. Providers that
        can do a multi-page call should override - it is usually cheaper and
        gives the model better cross-page layout.
        """
        from src.pipeline.stages.preprocess import split_pdf_pages

        pages = split_pdf_pages(pdf_bytes)
        return [await self.analyze_page(page, i) for i, page in enumerate(pages, start=1)]

    async def health(self) -> bool:
        """Whether the provider looks reachable and configured."""
        return True


def compute_mean_confidence(page: OCRPage) -> float:
    """Mean word confidence, or 1.0 when the provider reports none."""
    if not page.words:
        return 1.0
    return sum(w.confidence for w in page.words) / len(page.words)
