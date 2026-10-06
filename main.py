"""FastAPI application entry point.

    uvicorn main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.v1.endpoints import documents, health, rules, runs
from src.core.config.settings import get_settings
from src.core.errors.exceptions import ContractIDPError
from src.core.logging.structured import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.logging.level, settings.logging.json_output)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Resolve providers at startup so misconfiguration fails loudly and early."""
    logger.info(
        "starting %s in %s (ocr=%s, storage=%s, model=%s)",
        settings.app.name,
        settings.app.env,
        settings.ocr.provider,
        settings.storage.provider,
        settings.llm.model,
    )
    try:
        from src.core.database.connection import create_all

        if settings.database.auto_create:
            await create_all()
    except Exception as exc:
        logger.error("database initialisation failed: %s", exc)

    try:
        from src.services.storage.factory import get_storage

        get_storage()
    except Exception as exc:
        logger.error("storage provider failed to initialise: %s", exc)

    try:
        from src.services.ocr.factory import get_ocr_provider

        get_ocr_provider()
    except Exception as exc:
        # Not fatal: the service still serves health and uploads, and the error
        # is visible on /health rather than only at the first run.
        logger.error("OCR provider failed to initialise: %s", exc)

    yield

    try:
        from src.core.database.connection import dispose_engine

        await dispose_engine()
    except Exception:
        pass
    logger.info("shutting down")


app = FastAPI(
    title="Contract IDP",
    description=(
        "Document understanding pipeline: OCR, clause analysis, grounding, "
        "type-aware segmentation and playbook validation."
    ),
    version=health.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ContractIDPError)
async def domain_error_handler(request: Request, exc: ContractIDPError) -> JSONResponse:
    """Translate domain errors to their declared status codes."""
    if exc.status_code >= 500:
        logger.error("%s on %s: %s", exc.code, request.url.path, exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


app.include_router(health.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(runs.router, prefix="/api/v1")
app.include_router(rules.router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {"service": settings.app.name, "docs": "/docs", "health": "/api/v1/health"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.env == "local",
    )
