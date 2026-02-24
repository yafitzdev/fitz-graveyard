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

    # Read plan from file
    if not record.file_path:
        raise ToolError(
            f"Job '{sanitized_id}' has no plan file. The plan may not have been written yet."
        )

    try:
        with open(record.file_path, encoding="utf-8") as f:
            raw_content = f.read()
    except OSError as e:
        raise ToolError(f"Could not read plan file '{record.file_path}': {e}")

    if format == "full":
        content = raw_content
    elif format == "summary":
        # Return everything up to and including the Design section
        lines = raw_content.splitlines(keepends=True)
        result = []
        in_roadmap = False
        for line in lines:
            if line.startswith("## Roadmap") or line.startswith("## Risk"):
                in_roadmap = True
            if in_roadmap:
                continue
            result.append(line)
        content = "".join(result)
    elif format == "roadmap_only":
        lines = raw_content.splitlines(keepends=True)
        result = []
        in_roadmap = False
        for line in lines:
            if line.startswith("## Roadmap"):
                in_roadmap = True
            if in_roadmap:
                result.append(line)
        content = "".join(result) if result else raw_content

    # Build response
    response = PlanContentResponse(
        job_id=sanitized_id,
        format=format,
        content=content,
        file_path=record.file_path,
        quality_score=record.quality_score,
    )

    return response.model_dump()
