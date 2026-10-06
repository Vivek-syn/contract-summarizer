"""Application configuration settings.

One ``BaseSettings`` class per concern, each with its own env prefix, resolved
once through ``get_settings()``. Provider choices (OCR, storage) are flags here
rather than branches at call sites - see the factories in ``src/services``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = ".env"


class AppSettings(BaseSettings):
    """Top-level service settings."""

    name: str = Field(default="contract-idp", description="Service name in logs and traces")
    env: Literal["local", "dev", "staging", "prod"] = Field(default="local")
    debug: bool = Field(default=False, description="Enable debug behaviour and verbose errors")
    host: str = Field(default="0.0.0.0", description="Bind address")
    port: int = Field(default=8000, description="Bind port")
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"], description="Allowed CORS origins"
    )

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=_ENV_FILE, extra="ignore")


class DatabaseSettings(BaseSettings):
    """Where validation rules live.

    SQLite by default so the service runs with no database server; Postgres in
    deployment. Both are async drivers - the URL must carry ``+aiosqlite`` or
    ``+asyncpg``.
    """

    url: str = Field(
        default="sqlite+aiosqlite:///./data/contract_idp.db",
        description="Async SQLAlchemy URL, e.g. postgresql+asyncpg://user:pw@host/db",
    )
    echo: bool = Field(default=False, description="Log SQL statements")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    pool_timeout: int = Field(default=30)
    pool_recycle: int = Field(default=1800)
    auto_create: bool = Field(
        default=True,
        description=(
            "Create tables at startup if missing. Convenient locally; deployments "
            "should run Alembic and set this false."
        ),
    )

    model_config = SettingsConfigDict(env_prefix="DB_", env_file=_ENV_FILE, extra="ignore")


class StorageSettings(BaseSettings):
    """Where source PDFs, OCR output and stage results are persisted.

    A run outlives the request that started it, so three things must be durable:
    the source PDF (the reviewer checks the packet against it), the OCR output
    (the cache that makes reruns near-free), and each stage's output (what a
    scoped retry resumes from).
    """

    provider: Literal["local", "azure", "s3"] = Field(
        default="local", description="Storage backend"
    )
    local_root: str = Field(default="./data/storage", description="Root dir for the local backend")

    # Azure Blob
    azure_connection_string: Optional[str] = Field(default=None)
    azure_account_url: Optional[str] = Field(
        default=None, description="Used with DefaultAzureCredential when no connection string"
    )
    azure_container: str = Field(default="contract-idp")

    # AWS S3
    s3_bucket: Optional[str] = Field(default=None)
    s3_region: str = Field(default="ap-south-1")
    s3_prefix: str = Field(default="contract-idp/")

    model_config = SettingsConfigDict(env_prefix="STORAGE_", env_file=_ENV_FILE, extra="ignore")


class OCRSettings(BaseSettings):
    """OCR provider configuration.

    ``azure`` uses Document Intelligence ``prebuilt-layout`` rather than
    ``prebuilt-read``: layout returns tables as tables, selection marks and
    paragraph roles, so clause hierarchy survives. Costs more per page and is
    worth it for contracts.
    """

    provider: Literal["azure", "aws", "none"] = Field(
        default="azure", description="OCR backend; 'none' disables OCR entirely"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Cache OCR output by document content hash - the highest-value cache here",
    )
    max_concurrency: int = Field(default=5, description="Concurrent page OCR calls")
    timeout_seconds: int = Field(default=180)
    max_attempts: int = Field(default=3, description="Retries on transient provider errors")

    # Azure Document Intelligence
    azure_endpoint: Optional[str] = Field(default=None)
    azure_key: Optional[str] = Field(default=None)
    azure_model_id: str = Field(
        default="prebuilt-layout", description="prebuilt-layout | prebuilt-read"
    )

    # AWS Textract
    aws_region: str = Field(default="ap-south-1")
    aws_access_key_id: Optional[str] = Field(default=None)
    aws_secret_access_key: Optional[str] = Field(default=None)
    aws_feature_types: List[str] = Field(
        default_factory=lambda: ["TABLES", "FORMS"],
        description="Textract AnalyzeDocument features; empty list uses DetectDocumentText",
    )

    # Quality thresholds applied to whatever provider returns
    min_page_confidence: float = Field(
        default=0.80, description="Below this a page is flagged low-confidence, not dropped"
    )

    model_config = SettingsConfigDict(env_prefix="OCR_", env_file=_ENV_FILE, extra="ignore")


class LLMSettings(BaseSettings):
    """Model routing. Ids are LiteLLM-qualified: ``<provider>/<model>``."""

    model: str = Field(default="anthropic/claude-opus-5", description="Primary reasoning model")
    fast_model: str = Field(
        default="anthropic/claude-haiku-4-5",
        description="Cheap model for routing-style calls such as type detection",
    )
    max_tokens: int = Field(default=16000)
    effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(default="high")
    thinking: bool = Field(default=True, description="Anthropic adaptive thinking; ignored elsewhere")
    max_repairs: int = Field(default=2, description="Schema-repair turns before giving up")
    request_timeout: int = Field(default=600)
    num_retries: int = Field(default=3)

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=_ENV_FILE, extra="ignore")


class QuotaSettings(BaseSettings):
    """Token budget enforcement, checked at stage boundaries."""

    enabled: bool = Field(default=False)
    max_tokens_per_run: int = Field(default=2_000_000)
    max_usd_per_run: float = Field(default=10.0)
    max_usd_per_day: float = Field(default=100.0)

    model_config = SettingsConfigDict(env_prefix="QUOTA_", env_file=_ENV_FILE, extra="ignore")


class PipelineSettings(BaseSettings):
    """Stage behaviour and fan-out limits."""

    playbook_seed_dir: str = Field(
        default="./seeds/playbooks",
        description=(
            "Legacy YAML playbooks, used ONLY by the seed script. Rules are read from "
            "the database at runtime."
        ),
    )
    doctype_config_dir: str = Field(default="./doctypes")

    enable_grounding: bool = Field(default=True)
    enable_validation: bool = Field(default=True)

    validation_concurrency: int = Field(
        default=4, description="Rules validated in parallel - validation is rule-major"
    )
    analysis_chunk_chars: int = Field(default=1800)
    retrieval_top_k: int = Field(default=8)

    max_upload_mb: int = Field(default=50)
    max_pages: int = Field(default=300)

    model_config = SettingsConfigDict(env_prefix="PIPELINE_", env_file=_ENV_FILE, extra="ignore")

    @field_validator("validation_concurrency", "analysis_chunk_chars", "retrieval_top_k")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v


class LoggingSettings(BaseSettings):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    json_output: bool = Field(default=False, description="Structured JSON logs for prod")

    model_config = SettingsConfigDict(env_prefix="LOG_", env_file=_ENV_FILE, extra="ignore")


class Settings(BaseSettings):
    """Aggregate settings object."""

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    quota: QuotaSettings = Field(default_factory=QuotaSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, resolved once."""
    return Settings()
