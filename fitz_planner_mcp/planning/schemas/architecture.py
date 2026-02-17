# fitz_planner_mcp/planning/schemas/architecture.py
"""Schema for architecture exploration stage output."""

from pydantic import BaseModel, Field, ConfigDict


class Approach(BaseModel):
    """A single architectural approach option."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(
        ...,
        description="Short name for the approach (e.g., 'Microservices', 'Monolith')",
    )

    description: str = Field(
        ...,
        description="What this approach entails",
    )

    pros: list[str] = Field(
        default_factory=list,
        description="Advantages of this approach for the given requirements",
    )

    cons: list[str] = Field(
        default_factory=list,
        description="Disadvantages or risks",
    )

    complexity: str = Field(
        default="medium",
        description="Relative complexity: low, medium, high",
    )

    best_for: list[str] = Field(
        default_factory=list,
        description="Scenarios where this approach shines",
    )


class ArchitectureOutput(BaseModel):
    """Output from architecture exploration stage.

    Explores multiple architectural approaches before committing to one.
    """

    model_config = ConfigDict(extra="ignore")

    approaches: list[Approach] = Field(
        default_factory=list,
        description="List of architectural approaches considered",
    )

    recommended: str = Field(
        ...,
        description="Name of the recommended approach",
    )

    reasoning: str = Field(
        ...,
        description="Why the recommended approach is best for this project",
    )

    key_tradeoffs: dict[str, str] = Field(
        default_factory=dict,
        description="Major tradeoffs being made (e.g., 'simplicity vs scalability')",
    )

    technology_considerations: list[str] = Field(
        default_factory=list,
        description="Key technologies or patterns this architecture requires",
    )
