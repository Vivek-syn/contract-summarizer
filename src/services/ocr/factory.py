"""OCR provider selection."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, Optional

from src.core.config.settings import OCRSettings, get_settings
from src.core.errors.exceptions import ConfigurationError
from src.core.logging.structured import get_logger
from src.services.ocr.base import OCRProvider

logger = get_logger(__name__)


def _build_azure(settings: OCRSettings) -> OCRProvider:
    from src.services.ocr.azure_ocr import AzureOCR

    return AzureOCR(settings)


def _build_aws(settings: OCRSettings) -> OCRProvider:
    from src.services.ocr.aws_ocr import AWSOCR

    return AWSOCR(settings)


_BUILDERS: Dict[str, Callable[[OCRSettings], OCRProvider]] = {
    "azure": _build_azure,
    "aws": _build_aws,
}


@lru_cache(maxsize=1)
def get_ocr_provider() -> Optional[OCRProvider]:
    """The configured OCR provider, or None when OCR is disabled.

    Returns None rather than raising when ``OCR_PROVIDER=none`` so the pipeline
    can fall back to model transcription without special-casing the exception.
    """
    settings = get_settings().ocr
    if settings.provider == "none":
        logger.warning("OCR disabled (OCR_PROVIDER=none); model transcription fallback will be used")
        return None

    builder = _BUILDERS.get(settings.provider)
    if builder is None:
        raise ConfigurationError(
            f"unknown OCR_PROVIDER {settings.provider!r}; expected one of {sorted(_BUILDERS)} or 'none'"
        )

    try:
        provider = builder(settings)
    except ConfigurationError:
        # A misconfigured provider is a deployment error worth surfacing, but it
        # should not take the whole service down at import time.
        logger.error("OCR provider %s is selected but not configured", settings.provider)
        raise

    logger.info("OCR provider: %s (model=%s)", settings.provider,
                settings.azure_model_id if settings.provider == "azure" else "textract")
    return provider
