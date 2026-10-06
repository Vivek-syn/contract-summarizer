"""Logging setup.

Plain formatter locally, JSON in deployed environments. ``get_logger`` is the
only thing modules should import, so the backend can be swapped later without
touching call sites.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """One JSON object per line, with any ``extra`` fields merged in."""

    _RESERVED = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
        "pathname", "process", "processName", "relativeCreated", "stack_info",
        "thread", "threadName", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Install the root handler. Safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)-28s %(message)s", "%H:%M:%S")
        )
    root.addHandler(handler)

    # These are chatty and rarely tell us anything we want at INFO.
    for noisy in ("httpx", "httpcore", "LiteLLM", "litellm", "azure.core.pipeline.policies"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
