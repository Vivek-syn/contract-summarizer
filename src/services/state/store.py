"""Run state persistence.

Backed by the same storage service as everything else, so a run survives a
process restart and can be picked up by another worker later. An in-memory
index keeps listing cheap without scanning storage on every request.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from src.core.errors.exceptions import NotFoundError
from src.core.logging.structured import get_logger
from src.schemas.state import RunState
from src.services.storage.base import StorageService

logger = get_logger(__name__)


class StateStore:
    """Reads and writes :class:`RunState`."""

    def __init__(self, storage: StorageService) -> None:
        self.storage = storage
        self._cache: Dict[str, RunState] = {}
        # Keyed by event loop, not a single lock: the store is a process-wide
        # singleton, and an asyncio.Lock is bound to the loop that created it.
        # A lock built under one loop raises "bound to a different event loop"
        # under another - which happens with TestClient, and with any host that
        # runs more than one loop over the process's life.
        self._locks: Dict[int, asyncio.Lock] = {}

    def _lock_for_loop(self) -> asyncio.Lock:
        loop_id = id(asyncio.get_running_loop())
        lock = self._locks.get(loop_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[loop_id] = lock
        return lock

    async def save(self, state: RunState) -> None:
        """Persist a run's state.

        The lock is held across the storage write, not just the cache update:
        the three parallel branches finish at arbitrary times and all save the
        same run, and two concurrent writes to one key race on the rename
        (WinError 5 on Windows). Serialising them here is cheap - a state blob
        is small and writes are infrequent.
        """
        state.touch()
        async with self._lock_for_loop():
            self._cache[state.run_id] = state
            await self.storage.save_json(state, StorageService.run_key(state.run_id))

    async def get(self, run_id: str) -> RunState:
        """Load a run, preferring the in-memory copy.

        Raises:
            NotFoundError: If the run does not exist.
        """
        async with self._lock_for_loop():
            cached = self._cache.get(run_id)
        if cached is not None:
            return cached

        payload = await self.storage.load_json(StorageService.run_key(run_id))
        state = RunState.model_validate(payload)
        async with self._lock_for_loop():
            self._cache[run_id] = state
        return state

    async def try_get(self, run_id: str) -> Optional[RunState]:
        try:
            return await self.get(run_id)
        except NotFoundError:
            return None

    async def list_runs(self, limit: int = 50) -> List[RunState]:
        """Recent runs, newest first.

        Reads from the cache; a fresh process backfills it from storage.
        """
        async with self._lock_for_loop():
            if not self._cache:
                await self._backfill()
            runs = list(self._cache.values())
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs[:limit]

    async def _backfill(self) -> None:
        """Populate the cache from storage. Caller holds the lock."""
        try:
            keys = await self.storage.list_keys("runs/")
        except Exception as exc:
            logger.warning("could not list runs: %s", exc)
            return
        for key in keys:
            if not key.endswith("/state.json"):
                continue
            try:
                state = RunState.model_validate(await self.storage.load_json(key))
                self._cache[state.run_id] = state
            except Exception as exc:
                logger.warning("skipping unreadable run state %s: %s", key, exc)
