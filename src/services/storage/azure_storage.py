"""Azure Blob Storage backend.

Auth is either a connection string or ``DefaultAzureCredential`` against an
account URL - the latter is what you want in AKS with a managed identity, since
it keeps a key out of the environment entirely.

The SDK is imported lazily so the package is only required when this provider
is actually selected.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.core.errors.exceptions import ConfigurationError, NotFoundError, StorageError
from src.core.logging.structured import get_logger
from src.services.storage.base import StorageService

logger = get_logger(__name__)


class AzureBlobStorage(StorageService):
    """Stores objects as blobs in one container."""

    def __init__(
        self,
        *,
        container: str,
        connection_string: Optional[str] = None,
        account_url: Optional[str] = None,
    ) -> None:
        if not connection_string and not account_url:
            raise ConfigurationError(
                "Azure storage needs STORAGE_AZURE_CONNECTION_STRING or STORAGE_AZURE_ACCOUNT_URL"
            )
        self._container_name = container
        self._connection_string = connection_string
        self._account_url = account_url
        self._service = None  # built on first use, then reused

    async def _client(self):
        """Build the container client once, creating the container if needed."""
        if self._service is None:
            try:
                from azure.storage.blob.aio import BlobServiceClient
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ConfigurationError(
                    "azure-storage-blob is not installed; pip install '.[azure]'"
                ) from exc

            if self._connection_string:
                self._service = BlobServiceClient.from_connection_string(self._connection_string)
            else:
                from azure.identity.aio import DefaultAzureCredential

                self._service = BlobServiceClient(
                    account_url=self._account_url, credential=DefaultAzureCredential()
                )

            container = self._service.get_container_client(self._container_name)
            try:
                await container.create_container()
                logger.info("created blob container %s", self._container_name)
            except Exception:
                pass  # already exists, which is the normal case
        return self._service.get_container_client(self._container_name)

    async def save_bytes(
        self,
        content: bytes,
        key: str,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        from azure.storage.blob import ContentSettings

        container = await self._client()
        try:
            await container.upload_blob(
                name=key,
                data=content,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
                metadata=metadata or {},
            )
        except Exception as exc:
            raise StorageError(f"azure blob write failed for {key}: {exc}") from exc
        return f"azure://{self._container_name}/{key}"

    async def load_bytes(self, key: str) -> bytes:
        from azure.core.exceptions import ResourceNotFoundError

        container = await self._client()
        try:
            stream = await container.download_blob(key)
            return await stream.readall()
        except ResourceNotFoundError as exc:
            raise NotFoundError(f"no blob at {key}") from exc
        except Exception as exc:
            raise StorageError(f"azure blob read failed for {key}: {exc}") from exc

    async def exists(self, key: str) -> bool:
        container = await self._client()
        try:
            return await container.get_blob_client(key).exists()
        except Exception as exc:
            raise StorageError(f"azure blob exists check failed for {key}: {exc}") from exc

    async def delete(self, key: str) -> bool:
        from azure.core.exceptions import ResourceNotFoundError

        container = await self._client()
        try:
            await container.delete_blob(key)
            return True
        except ResourceNotFoundError:
            return False
        except Exception as exc:
            raise StorageError(f"azure blob delete failed for {key}: {exc}") from exc

    async def list_keys(self, prefix: str) -> List[str]:
        container = await self._client()
        try:
            return [blob.name async for blob in container.list_blobs(name_starts_with=prefix)]
        except Exception as exc:
            raise StorageError(f"azure blob list failed for {prefix}: {exc}") from exc

    async def aclose(self) -> None:
        """Release the SDK's connection pool. Called on app shutdown."""
        if self._service is not None:
            await self._service.close()
            self._service = None
