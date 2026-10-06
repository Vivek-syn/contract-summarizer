"""Run lifecycle: start, inspect, fetch results, and scoped retry."""

from __future__ import annotations

import uuid
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Response

from src.api.dependencies.services import state_store_dep, storage_dep
from src.api.v1.models.requests import CreateRunRequest, RetryRequest
from src.api.v1.models.responses import (
    RetryResponse,
    RunListResponse,
    RunResponse,
)
from src.core.errors.exceptions import NotFoundError, ValidationError
from src.core.logging.structured import get_logger
from src.pipeline.orchestrator import Orchestrator, build_dag
from src.schemas.state import RunState, RunStatus
from src.services.state.store import StateStore
from src.services.storage.base import StorageService

router = APIRouter(prefix="/runs", tags=["runs"])
logger = get_logger(__name__)


async def _execute(run_id: str, storage: StorageService, store: StateStore) -> None:
    """Background execution. Failures land on the run record, not the response."""
    state = await store.get(run_id)
    try:
        pdf = await storage.load_bytes(StorageService.source_key(state.doc_id))
    except Exception as exc:
        state.status = RunStatus.failed
        state.error = f"could not load source document: {exc}"
        await store.save(state)
        return

    orchestrator = Orchestrator(storage, store)
    try:
        await orchestrator.execute(state, pdf)
    except Exception as exc:
        logger.error("run %s failed: %s", run_id[:8], exc)


@router.post("", response_model=RunResponse, status_code=202)
async def create_run(
    request: CreateRunRequest,
    background: BackgroundTasks,
    storage: StorageService = Depends(storage_dep),
    store: StateStore = Depends(state_store_dep),
) -> RunResponse:
    """Start a run over an uploaded document.

    Returns immediately with the DAG in ``queued``; poll the run to watch stages
    move. Execution happens in the background so a long OCR does not hold the
    request open.
    """
    if not await storage.exists(StorageService.source_key(request.doc_id)):
        raise NotFoundError(f"no document with id {request.doc_id}; upload it first")

    run_id = str(uuid.uuid4())
    config: Dict[str, Any] = request.config.model_dump(exclude_none=True)
    state = build_dag(
        RunState(
            run_id=run_id,
            doc_id=request.doc_id,
            filename=request.config.document_type or request.doc_id[:16],
            config=config,
        )
    )
    await store.save(state)

    background.add_task(_execute, run_id, storage, store)
    logger.info("queued run %s for doc %s", run_id[:8], request.doc_id[:16])
    return RunResponse.from_state(state)


@router.get("", response_model=RunListResponse)
async def list_runs(
    limit: int = Query(default=50, ge=1, le=200),
    store: StateStore = Depends(state_store_dep),
) -> RunListResponse:
    """Recent runs, newest first."""
    runs = await store.list_runs(limit=limit)
    return RunListResponse(runs=[RunResponse.from_state(r) for r in runs], count=len(runs))


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    store: StateStore = Depends(state_store_dep),
) -> RunResponse:
    """The run's stage DAG, with per-stage status, tokens and cost."""
    return RunResponse.from_state(await store.get(run_id))


@router.get("/{run_id}/result")
async def get_result(
    run_id: str,
    store: StateStore = Depends(state_store_dep),
    storage: StorageService = Depends(storage_dep),
) -> Any:
    """The consolidated result JSON - everything the frontend needs."""
    state = await store.get(run_id)
    if not state.result_ref:
        raise NotFoundError(
            f"run {run_id} has no result yet (status: {state.status.value})"
        )
    return await storage.load_json(state.result_ref)


@router.get("/{run_id}/review")
async def get_review(
    run_id: str,
    store: StateStore = Depends(state_store_dep),
    storage: StorageService = Depends(storage_dep),
) -> Response:
    """The rendered review packet, as markdown."""
    state = await store.get(run_id)
    if state.status != RunStatus.completed:
        raise NotFoundError(
            f"run {run_id} is {state.status.value}; the review packet is not ready"
        )
    body = await storage.load_bytes(StorageService.review_key(run_id))
    return Response(content=body, media_type="text/markdown")


@router.post("/{run_id}/retry", response_model=RetryResponse, status_code=202)
async def retry_run(
    run_id: str,
    request: RetryRequest,
    background: BackgroundTasks,
    storage: StorageService = Depends(storage_dep),
    store: StateStore = Depends(state_store_dep),
) -> RetryResponse:
    """Invalidate one stage and everything downstream, then re-run.

    This is the human gate. Naming ``typedetect`` after spotting a wrong type
    re-runs three stages and leaves OCR alone; only naming ``ocr`` pays for OCR
    again. A blind full re-run of a near-deterministic pipeline mostly buys the
    same output at full price, which is what scoping avoids.
    """
    state = await store.get(run_id)
    if request.stage not in state.stages:
        raise ValidationError(
            f"unknown stage {request.stage!r}; expected one of {sorted(state.stages)}"
        )

    invalidated = state.invalidate_from(request.stage)
    if request.config:
        state.config.update(request.config.model_dump(exclude_none=True))
    if request.reason:
        # Nothing reads this yet. It is recorded because a labelled failure case
        # cannot be reconstructed after the fact.
        record = state.get(request.stage)
        if record:
            record.meta.setdefault("retry_reasons", []).append(request.reason)
    await store.save(state)

    background.add_task(_execute, run_id, storage, store)
    logger.info("retry %s from %s: invalidated %s", run_id[:8], request.stage, invalidated)
    return RetryResponse(run_id=run_id, invalidated=invalidated, status=state.status)
