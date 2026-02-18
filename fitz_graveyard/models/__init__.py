# fitz_graveyard/models/__init__.py
"""
Data models for fitz-graveyard.

Provides Pydantic response models and internal job tracking.
"""

from fitz_graveyard.models.jobs import (
    InMemoryJobStore,
    JobRecord,
    JobState,
    generate_job_id,
)
from fitz_graveyard.models.responses import (
    CreatePlanResponse,
    ListPlansResponse,
    PlanContentResponse,
    PlanStatusResponse,
    PlanSummary,
)

__all__ = [
    # Response models
    "CreatePlanResponse",
    "PlanStatusResponse",
    "PlanContentResponse",
    "PlanSummary",
    "ListPlansResponse",
    # Job tracking
    "JobState",
    "JobRecord",
    "InMemoryJobStore",
    "generate_job_id",
]
