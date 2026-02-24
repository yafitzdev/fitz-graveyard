# fitz_graveyard/tools/create_plan.py
"""
create_plan tool implementation.

Validates inputs, generates job ID, and queues planning work.
"""

import logging
from datetime import datetime, timezone

from fastmcp.exceptions import ToolError

from fitz_graveyard.config.schema import FitzPlannerConfig
from fitz_graveyard.models.jobs import (
    JobRecord,
    JobState,
    generate_job_id,
)
from fitz_graveyard.models.responses import CreatePlanResponse
from fitz_graveyard.models.store import JobStore
from fitz_graveyard.validation.sanitize import sanitize_description, sanitize_project_path

logger = logging.getLogger(__name__)


async def create_plan(
    description: str,
    timeline: str | None,
    context: str | None,
    integration_points: list[str] | None,
    api_review: bool,
    store: JobStore,
    config: FitzPlannerConfig,
    source_dir: str | None = None,
) -> dict:
    """
    Create a new architectural planning job.

    Validates description, generates unique job ID, and queues work.

    Args:
        description: What you want to build or accomplish
        timeline: Optional timeline constraints (e.g., "2 weeks", "Q1 2026")
        context: Optional additional context or constraints
        integration_points: Optional list of systems/APIs to integrate with
        api_review: Whether to enable API review for this job (default: False)
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

    # Validate source_dir if provided
    validated_source_dir: str | None = None
    if source_dir:
        try:
            validated_source_dir = str(sanitize_project_path(source_dir))
        except ToolError:
            raise

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
        api_review=api_review,
        source_dir=validated_source_dir,
    )

    # Add to store
    try:
        await store.add(record)
    except ValueError as e:
        # This should never happen with UUIDs, but handle gracefully
        logger.error(f"Job ID collision: {e}")
        raise ToolError(f"Internal error creating job: {e}")

    logger.info(
        f"Created planning job {job_id} for: {cleaned_description[:80]}... (api_review={api_review})"
    )

    # Build next_steps message based on api_review
    if api_review:
        next_steps = (
            "Job will pause for cost confirmation before API review. "
            "Use check_status to see estimate, then confirm_review or cancel_review."
        )
    else:
        next_steps = "Use check_status with job_id to monitor progress"

    # Build response
    response = CreatePlanResponse(
        job_id=job_id,
        status="queued",
        eta="Not yet implemented - planning engine coming in Phase 4",
        model=config.ollama.model,
        api_review=api_review,
        next_steps=next_steps,
    )

    return response.model_dump()
