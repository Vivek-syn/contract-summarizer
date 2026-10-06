"""Domain exceptions.

Each carries an HTTP status so the API layer can translate without a pile of
isinstance checks, and a ``code`` the frontend can branch on.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ContractIDPError(Exception):
    """Base for everything this service raises deliberately."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigurationError(ContractIDPError):
    """A provider is selected but not configured - a deployment mistake, not a user one."""

    status_code = 500
    code = "configuration_error"


class ValidationError(ContractIDPError):
    status_code = 422
    code = "validation_error"


class NotFoundError(ContractIDPError):
    status_code = 404
    code = "not_found"


class StorageError(ContractIDPError):
    status_code = 502
    code = "storage_error"


class OCRError(ContractIDPError):
    """OCR failed for a page or a document."""

    status_code = 502
    code = "ocr_error"


class LLMError(ContractIDPError):
    status_code = 502
    code = "llm_error"


class SchemaRepairExhausted(LLMError):
    """The model would not return a schema-valid tool call after every repair turn."""

    code = "schema_repair_exhausted"


class QuotaExceededError(ContractIDPError):
    status_code = 429
    code = "quota_exceeded"


class StageFailedError(ContractIDPError):
    """A pipeline stage failed. Carries the stage key so a retry can target it."""

    status_code = 500
    code = "stage_failed"

    def __init__(self, stage: str, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details={**(details or {}), "stage": stage})
        self.stage = stage


class UnsupportedDocumentError(ContractIDPError):
    status_code = 415
    code = "unsupported_document"
