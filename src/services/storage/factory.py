"""Storage provider selection.

Call sites ask for ``get_storage()`` and never branch on the provider. Adding a
backend means one module plus one entry in ``_BUILDERS``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict

from src.core.config.settings import StorageSettings, get_settings
from src.core.errors.exceptions import ConfigurationError
from src.core.logging.structured import get_logger
from src.services.storage.base import StorageService

logger = get_logger(__name__)


def _build_local(settings: StorageSettings) -> StorageService:
    from src.services.storage.local_storage import LocalStorage

    return LocalStorage(root=settings.local_root)


def _build_azure(settings: StorageSettings) -> StorageService:
    from src.services.storage.azure_storage import AzureBlobStorage

    return AzureBlobStorage(
        container=settings.azure_container,
        connection_string=settings.azure_connection_string,
        account_url=settings.azure_account_url,
    )


def _build_s3(settings: StorageSettings) -> StorageService:
    from src.services.storage.s3_storage import S3Storage

    return S3Storage(
        bucket=settings.s3_bucket or "",
        region=settings.s3_region,
        prefix=settings.s3_prefix,
    )


_BUILDERS: Dict[str, Callable[[StorageSettings], StorageService]] = {
    "local": _build_local,
    "azure": _build_azure,
    "s3": _build_s3,
}


@lru_cache(maxsize=1)
def get_storage() -> StorageService:
    """The configured storage backend, built once per process."""
    settings = get_settings().storage
    builder = _BUILDERS.get(settings.provider)
    if builder is None:
        raise ConfigurationError(
            f"unknown STORAGE_PROVIDER {settings.provider!r}; expected one of {sorted(_BUILDERS)}"
        )
    logger.info("storage provider: %s", settings.provider)
    return builder(settings)
