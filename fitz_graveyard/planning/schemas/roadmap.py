# fitz_graveyard/planning/schemas/roadmap.py
"""Schema for implementation roadmap stage output."""

from pydantic import BaseModel, Field, ConfigDict


class Phase(BaseModel):
    """A single implementation phase in the roadmap."""

    model_config = ConfigDict(extra="ignore")

    number: int = Field(
        ...,
        description="Phase number (1-indexed)",
    )

    name: str = Field(
        ...,
        description="Short descriptive name for this phase",
    )

    objective: str = Field(
        ...,
        description="What this phase achieves",
    )

    deliverables: list[str] = Field(
        default_factory=list,
        description="Concrete outputs from this phase",
    )

    dependencies: list[int] = Field(
        default_factory=list,
        description="Phase numbers that must complete before this one starts",
    )

    estimated_complexity: str = Field(
        default="medium",
        description="Relative complexity: low, medium, high",
    )

    key_risks: list[str] = Field(
        default_factory=list,
        description="Risks specific to this phase",
    )


class RoadmapOutput(BaseModel):
    """Output from implementation roadmap stage.

    Defines the sequence of phases to build the system.
    """

    model_config = ConfigDict(extra="ignore")

    phases: list[Phase] = Field(
        default_factory=list,
        description="Ordered list of implementation phases",
    )

    critical_path: list[int] = Field(
        default_factory=list,
        description="Phase numbers that form the critical path (must be sequential)",
    )

    parallel_opportunities: list[list[int]] = Field(
        default_factory=list,
        description="Groups of phase numbers that can be executed in parallel",
    )

    total_phases: int = Field(
        default=0,
        description="Total number of phases in the roadmap",
    )
