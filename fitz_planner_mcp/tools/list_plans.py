# fitz_planner_mcp/tools/list_plans.py
"""
list_plans tool implementation.

Lists all planning jobs with status and quality scores.
"""

import logging

from fitz_planner_mcp.models.jobs import InMemoryJobStore
from fitz_planner_mcp.models.responses import ListPlansResponse, PlanSummary

logger = logging.getLogger(__name__)


def list_plans(store: InMemoryJobStore) -> dict:
    """
    List all planning jobs.

    Returns all jobs with status, quality scores, and descriptions.

    Args:
        store: Job storage instance

    Returns:
        ListPlansResponse as dict
    """
    # Get all jobs (sorted by creation time, newest first)
    records = store.list_all()

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
        )
        summaries.append(summary)

    # Build response
    response = ListPlansResponse(plans=summaries, total=len(summaries))

    logger.info(f"Listed {len(summaries)} planning jobs")
    return response.model_dump()
