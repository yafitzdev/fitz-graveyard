# fitz_graveyard/tools/cancel_review.py
"""
cancel_review tool implementation.

Cancels API review and marks job as complete without review.
"""

import logging

from fastmcp.exceptions import ToolError

from fitz_graveyard.models.jobs import JobState
from fitz_graveyard.models.responses import CancelReviewResponse
from fitz_graveyard.models.store import JobStore
from fitz_graveyard.validation.sanitize import sanitize_job_id

logger = logging.getLogger(__name__)


async def cancel_review(job_id: str, store: JobStore) -> dict:
    """
    Cancel API review for a job awaiting review confirmation.

    Marks the job as complete without performing API review.

    Args:
        job_id: Job identifier from create_plan
        store: Job storage instance

    Returns:
        CancelReviewResponse as dict

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
            f"Only jobs in 'awaiting_review' state can be cancelled."
        )

    # Mark job as complete without review
    await store.update(
        sanitized_id,
        state=JobState.COMPLETE,
        current_phase="review_cancelled",
    )

    logger.info(f"Cancelled API review for job {sanitized_id}, marked as complete")

    # Build response
    response = CancelReviewResponse(job_id=sanitized_id)

    return response.model_dump()
