# fitz_planner_mcp/tools/check_status.py
"""
check_status tool implementation.

Retrieves job status and progress information.
"""

import json
import logging

from fastmcp.exceptions import ToolError

from fitz_planner_mcp.models.responses import PlanStatusResponse
from fitz_planner_mcp.models.store import JobStore
from fitz_planner_mcp.validation.sanitize import sanitize_job_id

logger = logging.getLogger(__name__)


async def check_status(job_id: str, store: JobStore) -> dict:
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
    record = await store.get(sanitized_id)
    if not record:
        raise ToolError(
            f"Job '{sanitized_id}' not found. Use list_plans to see available jobs."
        )

    # Parse cost estimate if present
    cost_estimate = None
    if record.cost_estimate_json:
        try:
            cost_estimate = json.loads(record.cost_estimate_json)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse cost_estimate_json for job {sanitized_id}")

    # Build human-readable status message
    if record.state.value == "queued":
        message = "Job is queued. Planning engine not yet implemented (Phase 4)."
    elif record.state.value == "running":
        phase_info = f" (phase: {record.current_phase})" if record.current_phase else ""
        message = f"Job is running{phase_info}. Progress: {record.progress * 100:.1f}%"
    elif record.state.value == "awaiting_review":
        # Format cost display
        cost_display = ""
        if cost_estimate:
            total_cost = cost_estimate.get("total_cost_usd", 0.0)
            cost_display = f"Estimated cost: ${total_cost:.4f}"
        else:
            cost_display = "Cost estimate not yet available"

        message = (
            f"Job is awaiting API review confirmation. {cost_display}. "
            f"Call confirm_review to proceed or cancel_review to skip."
        )
    elif record.state.value == "complete":
        message = f"Job complete. Quality score: {record.quality_score or 'N/A'}"
    elif record.state.value == "interrupted":
        message = f"Job was interrupted (server restart). Error: {record.error or 'Unknown'}. Use retry_job to re-queue."
    else:  # failed
        message = f"Job failed: {record.error or 'Unknown error'}. Use retry_job to re-queue."

    # Build response
    response = PlanStatusResponse(
        job_id=sanitized_id,
        state=record.state.value,
        progress=record.progress,
        current_phase=record.current_phase,
        eta=None,  # Stub for Phase 4
        message=message,
        error=record.error,
        cost_estimate=cost_estimate,
    )

    return response.model_dump()
