"""Local filesystem storage.

The default, so the service runs on a laptop with no cloud account. Blocking
file I/O is pushed to a worker thread to keep the event loop free.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from src.core.errors.exceptions import NotFoundError, StorageError
from src.core.logging.structured import get_logger
from src.services.storage.base import StorageService

logger = get_logger(__name__)

# Windows can refuse an atomic rename transiently; retry before giving up.
_REPLACE_ATTEMPTS = 5
_REPLACE_BACKOFF = 0.02


class LocalStorage(StorageService):
    """Stores objects as files under ``root``."""

    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info("local storage rooted at %s", self.root)

    def _path(self, key: str) -> Path:
        # Keys are service-generated, but refuse traversal rather than trust that.
        candidate = (self.root / key).resolve()
        if not str(candidate).startswith(str(self.root)):
            raise StorageError(f"key escapes storage root: {key!r}")
        return candidate

    async def save_bytes(
        self,
        content: bytes,
        key: str,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        path = self._path(key)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so a reader never sees a half-written file.
            # The temp name carries a uuid because concurrent stages write the
            # same key (run state, most of all) and a shared temp name collides.
            tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
            try:
                tmp.write_bytes(content)
                # On Windows the rename fails with WinError 5 if anything else
                # holds the destination open - a concurrent writer, an AV
                # scanner, the indexer. It is transient, so retry briefly rather
                # than failing the stage.
                for attempt in range(_REPLACE_ATTEMPTS):
                    try:
                        tmp.replace(path)
                        return
                    except PermissionError:
                        if attempt == _REPLACE_ATTEMPTS - 1:
                            raise
                        time.sleep(_REPLACE_BACKOFF * (attempt + 1))
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

        try:
            await asyncio.to_thread(_write)
        except OSError as exc:
            raise StorageError(f"failed to write {key}: {exc}") from exc
        return f"file://{path}"

    async def load_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise NotFoundError(f"no object at {key}")
        try:
            return await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise StorageError(f"failed to read {key}: {exc}") from exc

    async def exists(self, key: str) -> bool:
        return await asyncio.to_thread(self._path(key).exists)

    async def delete(self, key: str) -> bool:
        path = self._path(key)

        def _delete() -> bool:
            if not path.exists():
                return False
            path.unlink()
            return True

        return await asyncio.to_thread(_delete)

    async def list_keys(self, prefix: str) -> List[str]:
        base = self._path(prefix)

        def _list() -> List[str]:
            if not base.exists():
                return []
            if base.is_file():
                return [base.relative_to(self.root).as_posix()]
            return sorted(
                p.relative_to(self.root).as_posix()
                for p in base.rglob("*")
                if p.is_file() and not p.name.endswith(".tmp")
            )

        return await asyncio.to_thread(_list)
