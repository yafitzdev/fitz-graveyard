# fitz_graveyard/models/responses.py
"""
Pydantic response models for MCP tool outputs.

All tools return structured responses using these models for consistency.
"""

from pydantic import BaseModel, Field


class CreatePlanResponse(BaseModel):
    """Response from create_plan tool."""

    job_id: str = Field(description="Unique job identifier for tracking")
    status: str = Field(description="Job status (always 'queued' for new jobs)")
    eta: str = Field(description="Estimated time to completion")
    model: str = Field(description="LLM model that will be used")
    api_review: bool = Field(
        default=False, description="Whether API review is enabled for this job"
    )
    next_steps: str = Field(
        description="Instructions for monitoring job progress",
        default="Use check_status with job_id to monitor progress",
    )


class PlanStatusResponse(BaseModel):
    """Response from check_status tool."""

    job_id: str = Field(description="Job identifier")
    state: str = Field(description="Current job state (queued/running/awaiting_review/complete/failed/interrupted)")
    progress: float = Field(
        ge=0.0, le=1.0, description="Completion progress as fraction (0.0-1.0)"
    )
    current_phase: str | None = Field(
        default=None, description="Current planning phase if running"
    )
    eta: str | None = Field(default=None, description="Estimated time remaining")
    message: str | None = Field(
        default=None, description="Human-readable status message"
    )
    error: str | None = Field(
        default=None, description="Error message if job failed or interrupted"
    )
    cost_estimate: dict | None = Field(
        default=None, description="Cost estimate details when in awaiting_review state"
    )


class PlanContentResponse(BaseModel):
    """Response from get_plan tool."""

    job_id: str = Field(description="Job identifier")
    format: str = Field(description="Response format (full/summary/roadmap_only)")
    content: str = Field(description="Plan content in requested format")
    file_path: str | None = Field(
        default=None, description="Path to saved plan file on disk"
    )
    quality_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Plan quality score if available"
    )


class PlanSummary(BaseModel):
    """Summary information for a single plan (used in list_plans)."""

    job_id: str = Field(description="Job identifier")
    description: str = Field(description="Plan description (truncated to 80 chars)")
    state: str = Field(description="Current job state")
    quality_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Plan quality score if available"
    )
    created_at: str = Field(description="Creation timestamp (ISO format)")


class ListPlansResponse(BaseModel):
    """Response from list_plans tool."""

    plans: list[PlanSummary] = Field(
        default_factory=list, description="List of all plans"
    )
    total: int = Field(description="Total number of plans")


class RetryJobResponse(BaseModel):
    """Response from retry_job tool."""

    job_id: str = Field(description="Job identifier that was retried")
    status: str = Field(description="New status after retry (always 're-queued')")
    message: str = Field(
        description="Human-readable confirmation message",
        default="Job re-queued for processing. Use check_status to monitor.",
    )


class ConfirmReviewResponse(BaseModel):
    """Response from confirm_review tool."""

    job_id: str = Field(description="Job identifier")
    status: str = Field(
        default="review_started", description="Status after confirmation"
    )
    message: str = Field(
        default="API review started. Use check_status to monitor.",
        description="Human-readable confirmation message",
    )


class CancelReviewResponse(BaseModel):
    """Response from cancel_review tool."""

    job_id: str = Field(description="Job identifier")
    status: str = Field(
        default="review_skipped", description="Status after cancellation"
    )
    message: str = Field(
        default="API review cancelled. Plan finalized without API review.",
        description="Human-readable confirmation message",
    )
