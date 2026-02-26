# fitz_graveyard/tools/list_plans.py
"""
list_plans tool implementation.

Lists all planning jobs with status and quality scores.
"""

import logging

from fitz_graveyard.models.responses import ListPlansResponse, PlanSummary
from fitz_graveyard.models.store import JobStore

logger = logging.getLogger(__name__)


async def list_plans(store: JobStore) -> dict:
    """
    List all planning jobs.

    Returns all jobs with status, quality scores, and descriptions.

    Args:
        store: Job storage instance

    Returns:
        ListPlansResponse as dict
    """
    # Get all jobs (sorted by creation time, newest first)
    records = await store.list_all()

    # Convert to summaries
    summaries = []
    for record in records:
        # Truncate description to 80 chars
        desc = record.description
        if len(desc) > 80:
            desc = desc[:77] + "..."

        summary = PlanSummary(
            job_id=record.job_id,
            description=desc,
            state=record.state.value,
            quality_score=record.quality_score,
            created_at=record.created_at.isoformat(),
            file_path=record.file_path,
        )
        summaries.append(summary)

    # Build response
    response = ListPlansResponse(plans=summaries, total=len(summaries))

    logger.info(f"Listed {len(summaries)} planning jobs")
    return response.model_dump()
