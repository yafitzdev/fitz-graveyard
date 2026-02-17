# fitz_planner_mcp/planning/schemas/risk.py
"""Schema for risk analysis stage output."""

from pydantic import BaseModel, Field, ConfigDict


class Risk(BaseModel):
    """A single project risk with mitigation strategy."""

    model_config = ConfigDict(extra="ignore")

    category: str = Field(
        ...,
        description="Risk category (technical, resource, schedule, etc.)",
    )

    description: str = Field(
        ...,
        description="What the risk is",
    )

    impact: str = Field(
        ...,
        description="Impact level if risk materializes: low, medium, high, critical",
    )

    likelihood: str = Field(
        ...,
        description="Probability of occurrence: low, medium, high",
    )

    mitigation: str = Field(
        ...,
        description="How to prevent or reduce this risk",
    )

    contingency: str = Field(
        default="",
        description="Backup plan if risk materializes despite mitigation",
    )

    affected_phases: list[int] = Field(
        default_factory=list,
        description="Phase numbers most affected by this risk",
    )


class RiskOutput(BaseModel):
    """Output from risk analysis stage.

    Identifies project risks and mitigation strategies.
    """

    model_config = ConfigDict(extra="ignore")

    risks: list[Risk] = Field(
        default_factory=list,
        description="Identified risks with mitigations",
    )

    overall_risk_level: str = Field(
        default="medium",
        description="Overall project risk level: low, medium, high, critical",
    )

    recommended_contingencies: list[str] = Field(
        default_factory=list,
        description="High-level contingency plans for major risk scenarios",
    )
