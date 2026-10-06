"""Health and configuration introspection.

The health check reports which providers are actually reachable rather than just
whether the process is up - a misconfigured OCR key is the failure you want to
find at deploy time, not on the first document.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from src.api.dependencies.services import settings_dep
from src.api.v1.models.responses import HealthResponse, ProviderHealth
from src.core.config.settings import Settings
from src.services import doctypes
from src.services.llm.adapter import missing_keys

router = APIRouter(tags=["health"])

VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(settings_dep)) -> HealthResponse:
    """Liveness plus per-provider configuration status."""
    providers: List[ProviderHealth] = []

    # OCR
    ocr = settings.ocr
    if ocr.provider == "none":
        providers.append(
            ProviderHealth(name="ocr", configured=False, detail="disabled (OCR_PROVIDER=none)")
        )
    elif ocr.provider == "azure":
        ok = bool(ocr.azure_endpoint and ocr.azure_key)
        providers.append(
            ProviderHealth(
                name="ocr:azure",
                configured=ok,
                detail=f"model={ocr.azure_model_id}"
                if ok
                else "missing OCR_AZURE_ENDPOINT or OCR_AZURE_KEY",
            )
        )
    else:
        providers.append(
            ProviderHealth(name="ocr:aws", configured=True, detail=f"region={ocr.aws_region}")
        )

    # Storage
    storage = settings.storage
    if storage.provider == "local":
        detail = f"root={storage.local_root}"
        ok = True
    elif storage.provider == "azure":
        ok = bool(storage.azure_connection_string or storage.azure_account_url)
        detail = f"container={storage.azure_container}" if ok else "missing credentials"
    else:
        ok = bool(storage.s3_bucket)
        detail = f"bucket={storage.s3_bucket}" if ok else "missing STORAGE_S3_BUCKET"
    providers.append(
        ProviderHealth(name=f"storage:{storage.provider}", configured=ok, detail=detail)
    )

    # Models - the key needed depends on the model string.
    for label, model in (("llm", settings.llm.model), ("llm:fast", settings.llm.fast_model)):
        missing = missing_keys(model)
        providers.append(
            ProviderHealth(
                name=f"{label} ({model})",
                configured=not missing,
                detail="ready" if not missing else f"missing {', '.join(missing)}",
            )
        )

    degraded = any(not p.configured for p in providers if not p.name.startswith("ocr:none"))
    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=VERSION,
        environment=settings.app.env,
        providers=providers,
    )


@router.get("/config/doctypes")
async def list_doctypes(settings: Settings = Depends(settings_dep)) -> Dict[str, Any]:
    """The configured document types, as the detector sees them."""
    configured = doctypes.load_doctypes(settings.pipeline.doctype_config_dir)
    return {
        "count": len(configured),
        "types": [
            {
                "slug": t["slug"],
                "name": t.get("name"),
                "sections": t.get("sections", []),
                "focus": t.get("focus", []),
            }
            for t in configured.values()
        ],
    }
