# fitz_planner_mcp/tools/create_plan.py
"""
create_plan tool implementation.

Validates inputs, generates job ID, and queues planning work.
"""

import logging
from datetime import datetime, timezone

from fastmcp.exceptions import ToolError

from fitz_planner_mcp.config.schema import FitzPlannerConfig
from fitz_planner_mcp.models.jobs import (
    InMemoryJobStore,
    JobRecord,
    JobState,
    generate_job_id,
)
from fitz_planner_mcp.models.responses import CreatePlanResponse
from fitz_planner_mcp.validation.sanitize import sanitize_description

logger = logging.getLogger(__name__)


def create_plan(
    description: str,
    timeline: str | None,
    context: str | None,
    integration_points: list[str] | None,
    store: InMemoryJobStore,
    config: FitzPlannerConfig,
) -> dict:
    """
    Create a new architectural planning job.

    Validates description, generates unique job ID, and queues work.

    Args:
        description: What you want to build or accomplish
        timeline: Optional timeline constraints (e.g., "2 weeks", "Q1 2026")
        context: Optional additional context or constraints
        integration_points: Optional list of systems/APIs to integrate with
        store: Job storage instance
        config: Configuration instance

    Returns:
        CreatePlanResponse as dict

    Raises:
        ToolError: If description is invalid
    """
    # Validate description
    try:
        cleaned_description = sanitize_description(description)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Invalid description: {e}")

    # Generate unique job ID
    job_id = generate_job_id()

    # Create job record
    record = JobRecord(
        job_id=job_id,
        description=cleaned_description,
        timeline=timeline.strip() if timeline else None,
        context=context.strip() if context else None,
        integration_points=integration_points or [],
        state=JobState.QUEUED,
        progress=0.0,
        current_phase=None,
        quality_score=None,
        created_at=datetime.now(timezone.utc),
        file_path=None,
    )

    # Add to store
    try:
        store.add(record)
    except ValueError as e:
        # This should never happen with UUIDs, but handle gracefully
        logger.error(f"Job ID collision: {e}")
        raise ToolError(f"Internal error creating job: {e}")

    logger.info(
        f"Created planning job {job_id} for: {cleaned_description[:80]}..."
    )

    # Build response
    response = CreatePlanResponse(
        job_id=job_id,
        status="queued",
        eta="Not yet implemented - planning engine coming in Phase 4",
        model=config.model,
    )

    return response.model_dump()
