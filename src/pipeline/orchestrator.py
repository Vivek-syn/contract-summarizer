"""Pipeline orchestration.

Builds the run DAG, executes stages in dependency order, and persists each
stage's output so a scoped retry can resume from any node instead of starting
over.

Stages are pure functions of JSON in to JSON out; the orchestrator owns all the
state. That separation is what keeps the eventual move to a worker queue a
transport change rather than a rewrite - the stage bodies do not know whether
they are running in-process or on a worker.

The three branches after analysis (grounding, segmentation, validation) run
concurrently because none reads another's output.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from src.core.config.settings import get_settings
from src.core.errors.exceptions import ContractIDPError, QuotaExceededError, StageFailedError
from src.core.logging.structured import get_logger
from src.pipeline.stages import (
    analysis as analysis_stage,
    consolidate as consolidate_stage,
    grounding as grounding_stage,
    ocr_stage,
    preprocess as preprocess_stage,
    segmentation as segmentation_stage,
    typedetect as typedetect_stage,
    validation as validation_stage,
)
from src.schemas.documents import (
    AnalysisResult,
    ConsolidatedResult,
    DocumentType,
    GroundingResult,
    OCRResult,
    PreprocessResult,
    SegmentationResult,
    ValidationResult,
)
from src.schemas.state import RunState, RunStatus, StageStatus, Usage
from src.services import doctypes
from src.services.llm.adapter import LLMAdapter, UsageTracker
from src.services.state.store import StateStore
from src.services.storage.base import StorageService

logger = get_logger(__name__)

# Stage keys. Fan-out stages append their unit (``ocr:7``) at runtime.
S_PREPROCESS = "preprocess"
S_OCR = "ocr"
S_TYPE = "typedetect"
S_ANALYSIS = "analysis"
S_GROUNDING = "grounding"
S_SEGMENTATION = "segmentation"
S_VALIDATION = "validation"
S_CONSOLIDATE = "consolidate"


def build_dag(state: RunState) -> RunState:
    """Declare the stage graph for a run."""
    state.add_stage(S_PREPROCESS, "Preprocess and quality scan")
    state.add_stage(S_OCR, "OCR", [S_PREPROCESS])
    state.add_stage(S_TYPE, "Document type detection", [S_OCR])
    state.add_stage(S_ANALYSIS, "Clause analysis", [S_OCR, S_TYPE])
    state.add_stage(S_GROUNDING, "Grounding", [S_ANALYSIS])
    state.add_stage(S_SEGMENTATION, "Segmentation", [S_ANALYSIS])
    state.add_stage(S_VALIDATION, "Playbook validation", [S_ANALYSIS])
    state.add_stage(
        S_CONSOLIDATE, "Consolidate", [S_GROUNDING, S_SEGMENTATION, S_VALIDATION]
    )
    return state


class Orchestrator:
    """Runs one document through the pipeline."""

    def __init__(
        self,
        storage: StorageService,
        store: StateStore,
        *,
        adapter: Optional[LLMAdapter] = None,
    ) -> None:
        self.settings = get_settings()
        self.storage = storage
        self.store = store
        self.tracker = UsageTracker()
        self.adapter = adapter or LLMAdapter(self.tracker)

    # ---- helpers -------------------------------------------------------

    async def _persist(self, state: RunState, key: str, payload: Any) -> str:
        ref = StorageService.stage_key(state.run_id, key)
        await self.storage.save_json(payload, ref)
        return ref

    async def _load_stage(self, state: RunState, key: str, model: Any) -> Optional[Any]:
        """Reload a completed stage's output, so a retry can skip it."""
        record = state.get(key)
        if record is None or record.status != StageStatus.done or not record.output_ref:
            return None
        try:
            return model.model_validate(await self.storage.load_json(record.output_ref))
        except Exception as exc:
            logger.warning("could not reload %s output: %s", key, exc)
            return None

    def _check_quota(self, state: RunState) -> None:
        """Enforce the budget at stage boundaries, the natural checkpoints."""
        quota = self.settings.quota
        if not quota.enabled:
            return
        usage = state.usage
        if usage.total_tokens > quota.max_tokens_per_run:
            raise QuotaExceededError(
                f"run used {usage.total_tokens:,} tokens, over the "
                f"{quota.max_tokens_per_run:,} limit"
            )
        if usage.cost_usd > quota.max_usd_per_run:
            raise QuotaExceededError(
                f"run cost ${usage.cost_usd:.2f}, over the ${quota.max_usd_per_run:.2f} limit"
            )

    async def _run_stage(self, state: RunState, key: str, coro_factory) -> Any:
        """Execute one stage with bookkeeping, persistence and error capture."""
        record = state.get(key)
        if record is None:
            raise StageFailedError(key, f"stage {key} is not in the DAG")

        self._check_quota(state)
        before = self.tracker.usage
        record.mark_running()
        await self.store.save(state)
        logger.info("[%s] %s: running", state.run_id[:8], key)

        try:
            output = await coro_factory()
        except ContractIDPError as exc:
            record.mark_failed(exc.message)
            await self.store.save(state)
            raise StageFailedError(key, exc.message, details=exc.details) from exc
        except Exception as exc:
            record.mark_failed(f"{type(exc).__name__}: {exc}")
            await self.store.save(state)
            raise StageFailedError(key, f"{type(exc).__name__}: {exc}") from exc

        after = self.tracker.usage
        delta = Usage(
            input_tokens=after.input_tokens - before.input_tokens,
            output_tokens=after.output_tokens - before.output_tokens,
            calls=after.calls - before.calls,
            cost_usd=after.cost_usd - before.cost_usd,
        )
        ref = await self._persist(state, key, output) if output is not None else None
        record.mark_done(output_ref=ref, usage=delta)
        state.touch()
        await self.store.save(state)
        logger.info(
            "[%s] %s: done in %.1fs ($%.4f)",
            state.run_id[:8], key, record.duration_seconds or 0.0, delta.cost_usd,
        )
        return output

    # ---- the run -------------------------------------------------------

    async def execute(self, state: RunState, pdf_bytes: bytes) -> ConsolidatedResult:
        """Run every stage that is not already done.

        Stages already marked ``done`` are reloaded from storage rather than
        re-executed, which is what makes a scoped retry cheap: invalidating
        ``typedetect`` re-runs three stages, not the OCR of every page.
        """
        state.status = RunStatus.running
        state.touch()
        await self.store.save(state)

        pipeline_cfg = self.settings.pipeline
        config = state.config or {}

        try:
            # 1. preprocess ------------------------------------------------
            pre: Optional[PreprocessResult] = await self._load_stage(
                state, S_PREPROCESS, PreprocessResult
            )
            if pre is None:
                pre = await self._run_stage(
                    state,
                    S_PREPROCESS,
                    lambda: asyncio.to_thread(
                        preprocess_stage.run,
                        pdf_bytes,
                        state.filename,
                        max_pages=pipeline_cfg.max_pages,
                        max_mb=pipeline_cfg.max_upload_mb,
                    ),
                )

            # 2. ocr -------------------------------------------------------
            ocr: Optional[OCRResult] = await self._load_stage(state, S_OCR, OCRResult)
            if ocr is None:
                ocr = await self._run_stage(
                    state,
                    S_OCR,
                    lambda: ocr_stage.run(pdf_bytes, pre.doc_id, self.storage),
                )

            # 3. type detection -------------------------------------------
            doc_type: Optional[DocumentType] = await self._load_stage(
                state, S_TYPE, DocumentType
            )
            if doc_type is None:
                doc_type = await self._run_stage(
                    state,
                    S_TYPE,
                    lambda: typedetect_stage.run(
                        ocr,
                        doctypes.catalogue(pipeline_cfg.doctype_config_dir),
                        self.adapter,
                        override=config.get("document_type"),
                    ),
                )

            type_config = doctypes.get_doctype(
                pipeline_cfg.doctype_config_dir, doc_type.type
            )

            # 4. analysis --------------------------------------------------
            analysis: Optional[AnalysisResult] = await self._load_stage(
                state, S_ANALYSIS, AnalysisResult
            )
            if analysis is None:
                analysis = await self._run_stage(
                    state,
                    S_ANALYSIS,
                    lambda: analysis_stage.run(
                        ocr, doc_type, self.adapter, type_config=type_config
                    ),
                )

            # 5. the three parallel branches -------------------------------
            grounding_on = config.get("enable_grounding", pipeline_cfg.enable_grounding)
            validation_on = config.get("enable_validation", pipeline_cfg.enable_validation)

            async def _grounding() -> Optional[GroundingResult]:
                if not grounding_on:
                    self._skip(state, S_GROUNDING)
                    return None
                existing = await self._load_stage(state, S_GROUNDING, GroundingResult)
                if existing is not None:
                    return existing
                return await self._run_stage(
                    state,
                    S_GROUNDING,
                    lambda: asyncio.to_thread(grounding_stage.run, analysis, ocr),
                )

            async def _segmentation() -> Optional[SegmentationResult]:
                existing = await self._load_stage(state, S_SEGMENTATION, SegmentationResult)
                if existing is not None:
                    return existing
                return await self._run_stage(
                    state,
                    S_SEGMENTATION,
                    lambda: segmentation_stage.run(
                        analysis, doc_type, self.adapter, type_config=type_config
                    ),
                )

            async def _validation() -> Optional[ValidationResult]:
                if not validation_on:
                    self._skip(state, S_VALIDATION)
                    return None
                existing = await self._load_stage(state, S_VALIDATION, ValidationResult)
                if existing is not None:
                    return existing

                # Rules are rows, scoped by doctype. A type with none of its own
                # still gets the global ones (doctype IS NULL).
                rules = await self._load_rules(doc_type.type, config.get("rule_group"))
                if not rules:
                    logger.info("no enabled rules for type %r; skipping validation", doc_type.type)
                    self._skip(state, S_VALIDATION)
                    return None

                return await self._run_stage(
                    state,
                    S_VALIDATION,
                    lambda: validation_stage.run(
                        analysis,
                        rules,
                        self.adapter,
                        concurrency=pipeline_cfg.validation_concurrency,
                        top_k=pipeline_cfg.retrieval_top_k,
                    ),
                )

            grounding, segmentation, validation = await asyncio.gather(
                _grounding(), _segmentation(), _validation()
            )

            # 6. consolidate ------------------------------------------------
            usage = state.usage
            result = consolidate_stage.build(
                run_id=state.run_id,
                preprocess=pre,
                doc_type=doc_type,
                ocr=ocr,
                analysis=analysis,
                grounding=grounding,
                segmentation=segmentation,
                validation=validation,
                usage={
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "calls": usage.calls,
                    "cost_usd": round(usage.cost_usd, 4),
                },
            )

            record = state.get(S_CONSOLIDATE)
            if record:
                record.mark_running()
            await self.storage.save_json(result, StorageService.result_key(state.run_id))
            markdown = consolidate_stage.render_markdown(result, type_config=type_config)
            await self.storage.save_bytes(
                markdown.encode("utf-8"),
                StorageService.review_key(state.run_id),
                content_type="text/markdown",
            )
            if record:
                record.mark_done(output_ref=StorageService.result_key(state.run_id))

            state.result_ref = StorageService.result_key(state.run_id)
            state.status = RunStatus.completed
            state.touch()
            await self.store.save(state)
            logger.info(
                "[%s] complete: %d clauses, grounding %.0f%%, $%.3f",
                state.run_id[:8], len(analysis.clauses), result.grounding_rate * 100,
                usage.cost_usd,
            )
            return result

        except Exception as exc:
            state.status = RunStatus.failed
            state.error = str(exc)
            state.touch()
            await self.store.save(state)
            raise

    async def _load_rules(
        self, doctype: str, group: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Fetch the enabled rules for a document type from the database.

        Returns them in the same dict shape the validation stage has always
        taken, so moving rules out of YAML did not touch the stage.
        """
        from src.core.database.connection import get_session_factory
        from src.repositories.validation_rule_repository import ValidationRuleRepository

        async with get_session_factory()() as session:
            repo = ValidationRuleRepository(session)
            rows = await repo.get_for_doctype(doctype, enabled_only=True)
            if group:
                rows = [r for r in rows if r.group_name == group]
            return [row.to_prompt_dict() for row in rows]

    @staticmethod
    def _skip(state: RunState, key: str) -> None:
        record = state.get(key)
        if record and record.status in (StageStatus.pending, StageStatus.stale):
            record.status = StageStatus.skipped
