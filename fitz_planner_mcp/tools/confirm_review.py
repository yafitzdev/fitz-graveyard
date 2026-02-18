# fitz_planner_mcp/tools/confirm_review.py
"""
confirm_review tool implementation.

Confirms API review and re-queues job for worker to execute review.
"""

import logging

from fastmcp.exceptions import ToolError

from fitz_planner_mcp.models.jobs import JobState
from fitz_planner_mcp.models.responses import ConfirmReviewResponse
from fitz_planner_mcp.models.store import JobStore
from fitz_planner_mcp.validation.sanitize import sanitize_job_id

logger = logging.getLogger(__name__)


async def confirm_review(job_id: str, store: JobStore) -> dict:
    """
    Confirm API review for a job awaiting review confirmation.

    Re-queues the job for worker to execute the review.

    Args:
        job_id: Job identifier from create_plan
        store: Job storage instance

    Returns:
        ConfirmReviewResponse as dict

    Raises:
        ToolError: If job_id is invalid, not found, or not awaiting review
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

    # Verify job is awaiting review
    if record.state != JobState.AWAITING_REVIEW:
        raise ToolError(
            f"Job is not awaiting review confirmation. Current state: {record.state.value}. "
            f"Only jobs in 'awaiting_review' state can be confirmed."
        )

    # Re-queue job for worker to execute API review
    await store.update(
        sanitized_id,
        state=JobState.QUEUED,
        current_phase="api_review_confirmed",
    )

    logger.info(f"Confirmed API review for job {sanitized_id}, re-queued for execution")

    # Build response
    response = ConfirmReviewResponse(job_id=sanitized_id)

    return response.model_dump()
