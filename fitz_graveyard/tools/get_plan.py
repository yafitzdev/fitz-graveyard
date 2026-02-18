# fitz_graveyard/tools/get_plan.py
"""
get_plan tool implementation.

Retrieves completed plan content in requested format.
"""

import logging

from fastmcp.exceptions import ToolError

from fitz_graveyard.models.responses import PlanContentResponse
from fitz_graveyard.models.store import JobStore
from fitz_graveyard.validation.sanitize import sanitize_job_id

logger = logging.getLogger(__name__)

VALID_FORMATS = {"full", "summary", "roadmap_only"}


async def get_plan(
    job_id: str, format: str, store: JobStore
) -> dict:
    """
    Retrieve a completed plan.

    Supports full, summary, or roadmap_only format.

    Args:
        job_id: Job identifier from create_plan
        format: Response format (full/summary/roadmap_only)
        store: Job storage instance

    Returns:
        PlanContentResponse as dict

    Raises:
        ToolError: If job_id is invalid, not found, or job not complete
    """
    # Validate format
    if format not in VALID_FORMATS:
        raise ToolError(
            f"Invalid format '{format}'. Must be one of: {', '.join(VALID_FORMATS)}"
        )

    # Validate job ID
    try:
        sanitized_id = sanitize_job_id(job_id)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Invalid job ID: {e}")

    # Look up job
    record = await store.get(sanitized_id)
    if not record:
        raise ToolError(
            f"Job '{sanitized_id}' not found. Use list_plans to see available jobs."
        )

    # Check if job is complete
    if record.state.value != "complete":
        raise ToolError(
            f"Job '{sanitized_id}' is {record.state.value}. "
            f"Wait for completion before retrieving plan."
        )

    # Build stub content (actual plan generation comes in Phase 4)
    content = f"""# Plan for: {record.description}

**Job ID:** {record.job_id}
**Status:** Complete
**Quality Score:** {record.quality_score or 'N/A'}

## Stub Content

This is a placeholder response. Actual plan generation will be implemented in Phase 4.

**Format requested:** {format}
**Timeline:** {record.timeline or 'Not specified'}
**Context:** {record.context or 'Not provided'}
**Integration points:** {', '.join(record.integration_points) if record.integration_points else 'None'}
"""

    # Build response
    response = PlanContentResponse(
        job_id=sanitized_id,
        format=format,
        content=content,
        file_path=record.file_path,
        quality_score=record.quality_score,
    )

    return response.model_dump()
