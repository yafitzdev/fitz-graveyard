# fitz_graveyard/planning/schemas/risk.py
"""Schema for risk analysis stage output."""

from pydantic import BaseModel, Field, ConfigDict, model_validator


class Risk(BaseModel):
    """A single project risk with mitigation strategy."""

    model_config = ConfigDict(extra="ignore")

    category: str = Field(
        default="technical",
        description="Risk category (technical, resource, schedule, etc.)",
    )

    description: str = Field(
        default="",
        description="What the risk is",
    )

    impact: str = Field(
        default="medium",
        description="Impact level if risk materializes: low, medium, high, critical",
    )

    likelihood: str = Field(
        default="medium",
        description="Probability of occurrence: low, medium, high",
    )

    mitigation: str = Field(
        default="",
        description="How to prevent or reduce this risk",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_field_names(cls, data: dict) -> dict:
        """Handle LLM field name variations."""
        if not isinstance(data, dict):
            return data
        if "desc" in data and "description" not in data:
            data["description"] = data.pop("desc")
        if "phases" in data and "affected_phases" not in data:
            data["affected_phases"] = data.pop("phases")
        return data

    contingency: str = Field(
        default="",
        description="Backup plan if risk materializes despite mitigation",
    )

    affected_phases: list[int] = Field(
        default_factory=list,
        description="Phase numbers most affected by this risk",
    )

    verification: str = Field(
        default="",
        description="Exact test, assertion, or command that catches this risk",
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
