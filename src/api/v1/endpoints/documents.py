"""Document upload.

Upload is its own step rather than part of starting a run, so the same document
can be re-run under different configs without re-uploading, and so the quality
scan comes back immediately - the reviewer learns a document is a poor scan
before paying for a full run.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from src.api.dependencies.services import settings_dep, storage_dep
from src.api.v1.models.responses import UploadResponse
from src.core.config.settings import Settings
from src.core.errors.exceptions import ValidationError
from src.core.logging.structured import get_logger
from src.pipeline.stages import preprocess as preprocess_stage
from src.services.storage.base import StorageService

router = APIRouter(prefix="/documents", tags=["documents"])
logger = get_logger(__name__)

_ALLOWED_SUFFIXES = {".pdf"}


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(..., description="The PDF to analyse"),
    storage: StorageService = Depends(storage_dep),
    settings: Settings = Depends(settings_dep),
) -> UploadResponse:
    """Store a document, hash it, and return its quality scan.

    The returned ``doc_id`` is a content hash, so uploading the same file twice
    yields the same id - and the second run gets a free OCR cache hit.
    """
    filename = file.filename or "document.pdf"
    if not any(filename.lower().endswith(s) for s in _ALLOWED_SUFFIXES):
        raise ValidationError(f"only PDF uploads are supported, got {filename!r}")

    content = await file.read()
    if not content:
        raise ValidationError("uploaded file is empty")

    result = preprocess_stage.run(
        content,
        filename,
        max_pages=settings.pipeline.max_pages,
        max_mb=settings.pipeline.max_upload_mb,
    )

    uri = await storage.save_bytes(
        content,
        StorageService.source_key(result.doc_id),
        content_type="application/pdf",
        metadata={"filename": filename},
    )

    logger.info(
        "uploaded %s as %s (%d pages, %d flagged)",
        filename, result.doc_id[:16], result.pages, len(result.flagged_pages),
    )
    return UploadResponse(
        doc_id=result.doc_id,
        filename=filename,
        pages=result.pages,
        size_bytes=result.size_bytes,
        flagged_pages=result.flagged_pages,
        page_quality=result.page_quality,
        notes=result.notes,
        storage_uri=uri,
    )
