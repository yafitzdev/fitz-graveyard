# fitz_graveyard/planning/schemas/plan_output.py
"""Aggregated output schema combining all planning stages."""

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from fitz_graveyard.planning.schemas.context import ContextOutput
from fitz_graveyard.planning.schemas.architecture import ArchitectureOutput
from fitz_graveyard.planning.schemas.design import DesignOutput
from fitz_graveyard.planning.schemas.roadmap import RoadmapOutput
from fitz_graveyard.planning.schemas.risk import RiskOutput


class PlanOutput(BaseModel):
    """Complete planning output aggregating all stages.

    This is the final structured output from the multi-stage planning pipeline.
    """

    model_config = ConfigDict(extra="ignore")

    # Stage outputs
    context: ContextOutput = Field(
        ...,
        description="Context understanding stage output",
    )

    architecture: ArchitectureOutput = Field(
        ...,
        description="Architecture exploration stage output",
    )

    design: DesignOutput = Field(
        ...,
        description="Design decisions stage output",
    )

    roadmap: RoadmapOutput = Field(
        ...,
        description="Implementation roadmap stage output",
    )

    risk: RiskOutput = Field(
        ...,
        description="Risk analysis stage output",
    )

    # Quality metadata
    section_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Quality scores per stage (0.0-1.0), if available",
    )

    overall_quality_score: float = Field(
        default=0.0,
        description="Aggregate quality score for the entire plan (0.0-1.0)",
    )

    # Provenance
    git_sha: str = Field(
        default="",
        description="Git SHA of codebase when plan was generated (if available)",
    )

    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when plan was generated",
    )

    # API Review metadata
    api_review_requested: bool = Field(
        default=False,
        description="Whether API review was requested for this plan",
    )

    api_review_cost: dict | None = Field(
        default=None,
        description="Cost breakdown from API review (CostBreakdown as dict), if review was performed",
    )

    api_review_feedback: dict[str, str] | None = Field(
        default=None,
        description="API review feedback per section (section_name -> feedback text)",
    )
