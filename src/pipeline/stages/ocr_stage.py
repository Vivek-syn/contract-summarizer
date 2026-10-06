"""OCR stage: PDF to text with offsets, boxes and confidence.

Cached by document content hash. Since OCR is the expensive, deterministic,
irreversible step, a cache hit makes an entire rerun near-free - which is what
makes scoped retry affordable.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from src.core.config.settings import get_settings
from src.core.errors.exceptions import NotFoundError, OCRError
from src.core.logging.structured import get_logger
from src.schemas.documents import OCRPage, OCRResult
from src.services.ocr.base import OCRProvider
from src.services.ocr.factory import get_ocr_provider
from src.services.storage.base import StorageService

logger = get_logger(__name__)


async def _load_cached(storage: StorageService, doc_id: str, provider: str) -> Optional[OCRResult]:
    key = StorageService.ocr_key(doc_id, provider)
    try:
        payload = await storage.load_json(key)
    except NotFoundError:
        return None
    except Exception as exc:
        logger.warning("OCR cache read failed for %s: %s", key, exc)
        return None
    result = OCRResult.model_validate(payload)
    result.cached = True
    return result


async def _ocr_pages_parallel(
    provider: OCRProvider, page_pdfs: List[bytes], concurrency: int
) -> List[OCRPage]:
    """Fan out across pages, preserving page order in the result."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _one(index: int, data: bytes) -> OCRPage:
        async with semaphore:
            return await provider.analyze_page(data, index + 1)

    pages = await asyncio.gather(
        *(_one(i, data) for i, data in enumerate(page_pdfs)), return_exceptions=True
    )

    out: List[OCRPage] = []
    for index, page in enumerate(pages, start=1):
        if isinstance(page, BaseException):
            out.append(OCRPage(page=index, error=f"{type(page).__name__}: {page}"))
        else:
            out.append(page)
    out.sort(key=lambda p: p.page)
    return out


async def run(
    pdf_bytes: bytes,
    doc_id: str,
    storage: StorageService,
    *,
    provider: Optional[OCRProvider] = None,
    use_cache: Optional[bool] = None,
    whole_document: bool = True,
) -> OCRResult:
    """OCR a document, using the content-hash cache where possible.

    Args:
        pdf_bytes: The source PDF.
        doc_id: Content hash, the cache key.
        storage: Where cached OCR lives.
        provider: Override the configured provider (used by tests).
        use_cache: Override the cache setting.
        whole_document: Try a single multi-page call first. Providers that
            support it are cheaper that way and keep cross-page layout; we fall
            back to per-page fan-out if it fails.

    Returns:
        An :class:`OCRResult`. Pages that failed carry ``error`` rather than
        raising, so a bad page does not lose the document.

    Raises:
        OCRError: If no provider is configured, or every page failed.
    """
    settings = get_settings()
    ocr_provider = provider or get_ocr_provider()
    if ocr_provider is None:
        raise OCRError("no OCR provider configured (OCR_PROVIDER=none)")

    cache_on = settings.ocr.cache_enabled if use_cache is None else use_cache
    if cache_on:
        cached = await _load_cached(storage, doc_id, ocr_provider.name)
        if cached is not None:
            logger.info("OCR cache hit for %s (%d pages)", doc_id[:16], len(cached.pages))
            return cached

    pages: List[OCRPage] = []
    if whole_document:
        try:
            pages = await ocr_provider.analyze_document(pdf_bytes)
        except Exception as exc:
            logger.warning("whole-document OCR failed (%s); falling back to per-page", exc)
            pages = []

    if not pages:
        from src.pipeline.stages.preprocess import split_pdf_pages

        page_pdfs = split_pdf_pages(pdf_bytes)
        logger.info("OCR: %d pages, %d concurrent", len(page_pdfs), settings.ocr.max_concurrency)
        pages = await _ocr_pages_parallel(ocr_provider, page_pdfs, settings.ocr.max_concurrency)

    result = OCRResult(doc_id=doc_id, provider=ocr_provider.name, pages=pages)

    if pages and not any(p.ok for p in pages):
        raise OCRError(
            f"every page failed OCR ({len(pages)} pages)",
            details={"first_error": pages[0].error},
        )

    if result.failed_pages:
        logger.warning("OCR: pages %s failed", result.failed_pages)
    if result.low_confidence_pages:
        logger.warning("OCR: pages %s are low confidence", result.low_confidence_pages)

    if cache_on:
        try:
            await storage.save_json(result, StorageService.ocr_key(doc_id, ocr_provider.name))
        except Exception as exc:
            logger.warning("failed to cache OCR output: %s", exc)

    return result
