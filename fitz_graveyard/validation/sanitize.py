# fitz_graveyard/validation/sanitize.py
"""
Input sanitization and validation utilities.

Provides path traversal protection, input validation, and security checks.
"""

import logging
import re
from pathlib import Path

from fastmcp.exceptions import ToolError

logger = logging.getLogger(__name__)


def sanitize_project_path(user_path: str) -> Path:
    """
    Sanitize and validate project path.

    Resolves to absolute path and checks existence.
    Raises ToolError if path is invalid or doesn't exist.

    Args:
        user_path: User-provided path string

    Returns:
        Resolved absolute Path object

    Raises:
        ToolError: If path doesn't exist or is not a directory
    """
    try:
        resolved = Path(user_path).resolve()
    except (ValueError, OSError) as e:
        raise ToolError(f"Invalid path '{user_path}': {e}")

    if not resolved.exists():
        raise ToolError(f"Path does not exist: {resolved}")

    if not resolved.is_dir():
        raise ToolError(f"Path is not a directory: {resolved}")

    logger.info(f"Sanitized project path: {resolved}")
    return resolved


def sanitize_description(text: str, max_length: int = 5000) -> str:
    """
    Sanitize and validate description text.

    Strips whitespace and validates non-empty.
    Truncates to max_length if needed.

    Args:
        text: User-provided description text
        max_length: Maximum allowed length (default 5000)

    Returns:
        Cleaned description string

    Raises:
        ToolError: If description is empty after stripping
    """
    cleaned = text.strip()

    if not cleaned:
        raise ToolError("Description cannot be empty")

    if len(cleaned) > max_length:
        logger.warning(
            f"Description truncated from {len(cleaned)} to {max_length} characters"
        )
        cleaned = cleaned[:max_length]

    return cleaned


def sanitize_job_id(job_id: str) -> str:
    """
    Sanitize and validate job ID.

    Job IDs must be alphanumeric with hyphens only, 8-64 characters.

    Args:
        job_id: User-provided job ID

    Returns:
        Validated job ID

    Raises:
        ToolError: If job ID format is invalid
    """
    # Validate format: alphanumeric + hyphens, 8-64 chars
    pattern = r"^[a-zA-Z0-9-]{8,64}$"
    if not re.match(pattern, job_id):
        raise ToolError(
            f"Invalid job ID '{job_id}': must be 8-64 alphanumeric characters or hyphens"
        )

    return job_id
