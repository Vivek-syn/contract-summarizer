"""Runtime configuration.

Model IDs are LiteLLM-qualified: ``anthropic/<anthropic-model-id>``.
Do not append date suffixes - ``claude-opus-5`` is the complete id.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Load .env explicitly. Importing litellm happens to call load_dotenv() too,
    # but relying on another library's import side effect to read your config is
    # a silent failure waiting to happen.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # python-dotenv is optional; real env vars still work
    pass


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # Primary reasoning model for extraction, summarization and validation.
    model: str = field(default_factory=lambda: os.getenv("SUMMARIZER_MODEL", "anthropic/claude-opus-5"))
    # The judge runs on its own model so you can A/B it independently.
    judge_model: str = field(default_factory=lambda: os.getenv("SUMMARIZER_JUDGE_MODEL", "anthropic/claude-opus-5"))

    max_tokens: int = field(default_factory=lambda: int(os.getenv("SUMMARIZER_MAX_TOKENS", "16000")))
    # Anthropic effort levels: low | medium | high | xhigh | max
    effort: str = field(default_factory=lambda: os.getenv("SUMMARIZER_EFFORT", "high"))
    # Adaptive thinking. budget_tokens is rejected on Opus 5 - never set it.
    thinking: bool = field(default_factory=lambda: _env_flag("SUMMARIZER_THINKING", True))

    # Judge gate: below this score the pipeline repairs and retries.
    judge_threshold: float = field(default_factory=lambda: float(os.getenv("SUMMARIZER_JUDGE_THRESHOLD", "0.75")))
    max_repairs: int = field(default_factory=lambda: int(os.getenv("SUMMARIZER_MAX_REPAIRS", "2")))

    # Ingest: below this many extracted chars per page we assume a scan and OCR it.
    min_chars_per_page: int = 120
    # Retrieval
    chunk_chars: int = 1800
    chunk_overlap: int = 250
    top_k: int = 8

    max_agent_turns: int = 12


SETTINGS = Settings()
