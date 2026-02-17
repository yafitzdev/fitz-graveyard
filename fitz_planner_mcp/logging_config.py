# fitz_planner_mcp/logging_config.py
"""
Stderr-only JSON logging configuration.

CRITICAL: MCP uses stdio transport, so ALL logging must go to stderr.
No print() statements, no stdout handlers.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON line."""
        log_data: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Include exception info if present
        if record.exc_info:
            log_data["exc"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def configure_logging() -> None:
    """
    Configure logging to output JSON to stderr only.

    MUST be called before any imports that might create loggers.
    Clears existing handlers to prevent stdout pollution.
    """
    # Create stderr handler with JSON formatter
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())

    # Get root logger and clear all existing handlers
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Configure third-party loggers to use same handler
    for logger_name in ["uvicorn", "fastmcp"]:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Don't propagate to root to avoid double logging
