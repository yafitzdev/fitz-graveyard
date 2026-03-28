# benchmarks/eval_schemas.py
"""
Pydantic schemas for Sonnet-as-Judge plan evaluation.

Scores plans on 6 dimensions (1-10 each, max 60) measuring whether
the plan would actually work if implemented against the target codebase.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    model_config = ConfigDict(extra="ignore")

    score: int = Field(ge=1, le=10, description="Score from 1-10")
    justification: str = Field(description="Brief evidence-based justification")


class PlanScore(BaseModel):
    """Full evaluation result for a single plan."""

    model_config = ConfigDict(extra="ignore")

    plan_file: str
    query: str
    file_identification: DimensionScore
    contract_preservation: DimensionScore
    internal_consistency: DimensionScore
    codebase_alignment: DimensionScore
    implementability: DimensionScore
    scope_calibration: DimensionScore
    overall_notes: str = ""
    total_score: int = Field(description="Sum of 6 dimensions, max 60")
    normalized_score: float = Field(description="total / 60")
    scored_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @property
    def dimensions(self) -> dict[str, DimensionScore]:
        return {
            "file_identification": self.file_identification,
            "contract_preservation": self.contract_preservation,
            "internal_consistency": self.internal_consistency,
            "codebase_alignment": self.codebase_alignment,
            "implementability": self.implementability,
            "scope_calibration": self.scope_calibration,
        }


class ConsistencyCheck(BaseModel):
    """Result of scoring the same plan twice to check scorer stability."""

    model_config = ConfigDict(extra="ignore")

    plan_file: str
    run_1_total: int
    run_2_total: int
    dimension_deltas: dict[str, int]
    max_delta: int
    acceptable: bool = Field(description="True if max_delta <= 2")


class BatchScore(BaseModel):
    """Aggregate results from scoring multiple plans."""

    model_config = ConfigDict(extra="ignore")

    query: str
    model: str
    scored_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    plans_scored: int
    dimension_averages: dict[str, float]
    total_average: float
    total_std_dev: float
    total_min: int
    total_max: int
    total_cost_usd: float
    scores: list[PlanScore]
    consistency_check: ConsistencyCheck | None = None
