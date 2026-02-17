# fitz_planner_mcp/tools/retry_job.py
"""
retry_job tool implementation.

Re-queues failed or interrupted jobs for processing.
"""

import logging

from fastmcp.exceptions import ToolError

from fitz_planner_mcp.models.jobs import JobState
from fitz_planner_mcp.models.responses import RetryJobResponse
from fitz_planner_mcp.models.store import JobStore
from fitz_planner_mcp.validation.sanitize import sanitize_job_id

logger = logging.getLogger(__name__)


async def retry_job(job_id: str, store: JobStore) -> dict:
    """
    Retry a failed or interrupted job by re-queuing it.

    Clears error state and resets progress, allowing the background worker
    to pick it up again.

    Args:
        job_id: Job identifier from create_plan
        store: Job storage instance

    Returns:
        RetryJobResponse as dict

    Raises:
        ToolError: If job_id is invalid, not found, or not in retryable state
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

    # Check if job is in a retryable state
    if record.state not in (JobState.FAILED, JobState.INTERRUPTED):
        raise ToolError(
            f"Job '{sanitized_id}' is in '{record.state.value}' state. "
            f"Only failed or interrupted jobs can be retried."
        )

    # Reset job to queued state
    await store.update(
        sanitized_id,
        state=JobState.QUEUED,
        progress=0.0,
        error=None,
        current_phase=None,
    )

    logger.info(f"Re-queued job {sanitized_id} (was {record.state.value})")

    # Build response
    response = RetryJobResponse(
        job_id=sanitized_id,
        status="re-queued",
        message="Job re-queued for processing. Use check_status to monitor.",
    )

    return response.model_dump()
