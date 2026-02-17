# fitz_planner_mcp/planning/schemas/design.py
"""Schema for design decisions stage output."""

from pydantic import BaseModel, Field, ConfigDict


class ADR(BaseModel):
    """Architectural Decision Record.

    Documents a significant design decision with context and rationale.
    """

    model_config = ConfigDict(extra="ignore")

    title: str = Field(
        ...,
        description="Short title for the decision",
    )

    context: str = Field(
        ...,
        description="What problem or question this decision addresses",
    )

    decision: str = Field(
        ...,
        description="What was decided",
    )

    rationale: str = Field(
        ...,
        description="Why this decision was made",
    )

    consequences: list[str] = Field(
        default_factory=list,
        description="Expected positive and negative consequences",
    )

    alternatives_considered: list[str] = Field(
        default_factory=list,
        description="Other options that were evaluated",
    )


class ComponentDesign(BaseModel):
    """Design specification for a system component."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(
        ...,
        description="Component name",
    )

    purpose: str = Field(
        ...,
        description="What this component does",
    )

    responsibilities: list[str] = Field(
        default_factory=list,
        description="Key responsibilities of this component",
    )

    interfaces: list[str] = Field(
        default_factory=list,
        description="Public interfaces or APIs this component exposes",
    )

    dependencies: list[str] = Field(
        default_factory=list,
        description="Other components or external systems this depends on",
    )


class DesignOutput(BaseModel):
    """Output from design decisions stage.

    Captures concrete design choices and their rationale.
    """

    model_config = ConfigDict(extra="ignore")

    adrs: list[ADR] = Field(
        default_factory=list,
        description="Architectural Decision Records for key choices",
    )

    components: list[ComponentDesign] = Field(
        default_factory=list,
        description="Major system components and their designs",
    )

    data_model: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Key entities and their primary attributes",
    )

    integration_points: list[str] = Field(
        default_factory=list,
        description="External systems or APIs that will be integrated",
    )
