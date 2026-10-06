"""Abstract storage interface.

A unified surface over local disk, Azure Blob and S3. All methods are async so
a provider doing network I/O does not block the event loop; the local backend
runs its blocking calls in a thread.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class StorageService(ABC):
    """Base class for storage backends."""

    @abstractmethod
    async def save_bytes(
        self,
        content: bytes,
        key: str,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Write bytes at ``key``.

        Args:
            content: Payload to store.
            key: Storage key, forward-slash separated (e.g. ``docs/<hash>/source.pdf``).
            content_type: MIME type recorded where the backend supports it.
            metadata: Optional key/value metadata.

        Returns:
            A URI identifying the stored object.

        Raises:
            StorageError: If the write fails.
        """

    @abstractmethod
    async def load_bytes(self, key: str) -> bytes:
        """Read the object at ``key``.

        Raises:
            NotFoundError: If no object exists at ``key``.
            StorageError: If the read fails.
        """

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Whether an object exists at ``key``."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete ``key``. Returns False if it was not there."""

    @abstractmethod
    async def list_keys(self, prefix: str) -> List[str]:
        """Every key under ``prefix``."""

    # ---- JSON convenience, shared by all backends ----------------------

    async def save_json(self, payload: Any, key: str) -> str:
        """Serialise ``payload`` and store it as UTF-8 JSON."""
        if hasattr(payload, "model_dump_json"):
            body = payload.model_dump_json(indent=2).encode("utf-8")
        else:
            body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        return await self.save_bytes(body, key, content_type="application/json")

    async def load_json(self, key: str) -> Any:
        """Read and parse JSON stored at ``key``."""
        return json.loads((await self.load_bytes(key)).decode("utf-8"))

    # ---- key layout, shared so every backend agrees ---------------------

    @staticmethod
    def safe(segment: str) -> str:
        """Make an id safe as a path segment.

        Ids carry a colon (``sha256:...``, ``ocr:7``). Windows rejects ':' in
        paths outright, and it is awkward in S3 and Azure key listings, so every
        key builder normalises it rather than each backend coping separately.
        """
        return segment.replace(":", "_").replace("/", "_").replace("\\", "_")

    @classmethod
    def source_key(cls, doc_id: str, filename: str = "source.pdf") -> str:
        return f"docs/{cls.safe(doc_id)}/{filename}"

    @classmethod
    def ocr_key(cls, doc_id: str, provider: str) -> str:
        """OCR output keyed by content hash - this is what makes reruns free."""
        return f"ocr/{cls.safe(provider)}/{cls.safe(doc_id)}.json"

    @classmethod
    def stage_key(cls, run_id: str, stage: str) -> str:
        return f"runs/{cls.safe(run_id)}/stages/{cls.safe(stage)}.json"

    @classmethod
    def run_key(cls, run_id: str) -> str:
        return f"runs/{cls.safe(run_id)}/state.json"

    @classmethod
    def result_key(cls, run_id: str) -> str:
        return f"runs/{cls.safe(run_id)}/result.json"

    @classmethod
    def review_key(cls, run_id: str) -> str:
        return f"runs/{cls.safe(run_id)}/review.md"
