"""AWS S3 storage backend.

boto3 is synchronous, so every call is pushed to a worker thread. Credentials
come from the standard chain (env, shared config, instance role) unless keys are
set explicitly - prefer the role.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from src.core.errors.exceptions import ConfigurationError, NotFoundError, StorageError
from src.core.logging.structured import get_logger
from src.services.storage.base import StorageService

logger = get_logger(__name__)


class S3Storage(StorageService):
    """Stores objects in one bucket under an optional key prefix."""

    def __init__(
        self,
        *,
        bucket: str,
        region: str,
        prefix: str = "",
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
    ) -> None:
        if not bucket:
            raise ConfigurationError("S3 storage needs STORAGE_S3_BUCKET")
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ConfigurationError("boto3 is not installed; pip install '.[aws]'") from exc

        self.bucket = bucket
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
        session_kwargs: Dict[str, str] = {"region_name": region}
        if access_key_id and secret_access_key:
            session_kwargs["aws_access_key_id"] = access_key_id
            session_kwargs["aws_secret_access_key"] = secret_access_key
        self._client = boto3.client("s3", **session_kwargs)
        logger.info("s3 storage on bucket %s prefix %r", bucket, self.prefix)

    def _full(self, key: str) -> str:
        return f"{self.prefix}{key}"

    async def save_bytes(
        self,
        content: bytes,
        key: str,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        def _put() -> None:
            self._client.put_object(
                Bucket=self.bucket,
                Key=self._full(key),
                Body=content,
                ContentType=content_type,
                Metadata=metadata or {},
            )

        try:
            await asyncio.to_thread(_put)
        except Exception as exc:
            raise StorageError(f"s3 write failed for {key}: {exc}") from exc
        return f"s3://{self.bucket}/{self._full(key)}"

    async def load_bytes(self, key: str) -> bytes:
        def _get() -> bytes:
            response = self._client.get_object(Bucket=self.bucket, Key=self._full(key))
            return response["Body"].read()

        try:
            return await asyncio.to_thread(_get)
        except self._client.exceptions.NoSuchKey as exc:
            raise NotFoundError(f"no object at {key}") from exc
        except Exception as exc:
            if getattr(exc, "response", {}).get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
                raise NotFoundError(f"no object at {key}") from exc
            raise StorageError(f"s3 read failed for {key}: {exc}") from exc

    async def exists(self, key: str) -> bool:
        def _head() -> bool:
            try:
                self._client.head_object(Bucket=self.bucket, Key=self._full(key))
                return True
            except Exception as exc:
                if getattr(exc, "response", {}).get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
                    return False
                raise

        try:
            return await asyncio.to_thread(_head)
        except Exception as exc:
            raise StorageError(f"s3 exists check failed for {key}: {exc}") from exc

    async def delete(self, key: str) -> bool:
        def _delete() -> bool:
            self._client.delete_object(Bucket=self.bucket, Key=self._full(key))
            return True

        try:
            return await asyncio.to_thread(_delete)
        except Exception as exc:
            raise StorageError(f"s3 delete failed for {key}: {exc}") from exc

    async def list_keys(self, prefix: str) -> List[str]:
        def _list() -> List[str]:
            keys: List[str] = []
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self._full(prefix)):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    keys.append(key[len(self.prefix):] if self.prefix else key)
            return keys

        try:
            return await asyncio.to_thread(_list)
        except Exception as exc:
            raise StorageError(f"s3 list failed for {prefix}: {exc}") from exc
