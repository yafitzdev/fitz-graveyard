# fitz_planner_mcp/tools/check_status.py
"""
check_status tool implementation.

Retrieves job status and progress information.
"""

import logging

from fastmcp.exceptions import ToolError

from fitz_planner_mcp.models.jobs import InMemoryJobStore
from fitz_planner_mcp.models.responses import PlanStatusResponse
from fitz_planner_mcp.validation.sanitize import sanitize_job_id

logger = logging.getLogger(__name__)


def check_status(job_id: str, store: InMemoryJobStore) -> dict:
    """
    Check the status of a planning job.

    Returns state, progress, current phase, and ETA.

    Args:
        job_id: Job identifier from create_plan
        store: Job storage instance

    Returns:
        PlanStatusResponse as dict

    Raises:
        ToolError: If job_id is invalid or not found
    """
    # Validate job ID format
    try:
        sanitized_id = sanitize_job_id(job_id)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Invalid job ID: {e}")

    # Look up job
    record = store.get(sanitized_id)
    if not record:
        raise ToolError(
            f"Job '{sanitized_id}' not found. Use list_plans to see available jobs."
        )

    # Build human-readable status message
    if record.state.value == "queued":
        message = "Job is queued. Planning engine not yet implemented (Phase 4)."
    elif record.state.value == "running":
        phase_info = f" (phase: {record.current_phase})" if record.current_phase else ""
        message = f"Job is running{phase_info}. Progress: {record.progress * 100:.1f}%"
    elif record.state.value == "complete":
        message = f"Job complete. Quality score: {record.quality_score or 'N/A'}"
    else:  # failed
        message = f"Job failed: {record.error or 'Unknown error'}"

    # Build response
    response = PlanStatusResponse(
        job_id=sanitized_id,
        state=record.state.value,
        progress=record.progress,
        current_phase=record.current_phase,
        eta=None,  # Stub for Phase 4
        message=message,
    )

    return response.model_dump()
