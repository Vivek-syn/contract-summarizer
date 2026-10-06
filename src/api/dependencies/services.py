"""FastAPI dependency providers.

Services are process-wide singletons, so these are thin accessors rather than
per-request construction.
"""

from __future__ import annotations

from functools import lru_cache

from src.core.config.settings import Settings, get_settings
from src.pipeline.orchestrator import Orchestrator
from src.services.state.store import StateStore
from src.services.storage.base import StorageService
from src.services.storage.factory import get_storage


@lru_cache(maxsize=1)
def _state_store() -> StateStore:
    return StateStore(get_storage())


def storage_dep() -> StorageService:
    return get_storage()


def state_store_dep() -> StateStore:
    return _state_store()


def settings_dep() -> Settings:
    return get_settings()


def orchestrator_dep() -> Orchestrator:
    """A fresh orchestrator per run, so usage tracking is per-run."""
    return Orchestrator(get_storage(), _state_store())
