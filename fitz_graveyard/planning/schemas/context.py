# fitz_graveyard/planning/schemas/context.py
"""Schema for context understanding stage output."""

from pydantic import BaseModel, Field, ConfigDict


class Assumption(BaseModel):
    """An explicit assumption made due to ambiguity in the request."""

    model_config = ConfigDict(extra="ignore")

    assumption: str = Field(..., description="What the model assumed")
    impact: str = Field(..., description="What changes if this assumption is wrong")
    confidence: str = Field(default="medium", description="How confident: low, medium, high")


class ContextOutput(BaseModel):
    """Output from context understanding stage.

    Captures the essential understanding of what needs to be built,
    who it's for, and what constraints exist.
    """

    model_config = ConfigDict(extra="ignore")

    project_description: str = Field(
        ...,
        description="Clear summary of what is being built",
    )

    key_requirements: list[str] = Field(
        default_factory=list,
        description="Essential functional and non-functional requirements",
    )

    constraints: list[str] = Field(
        default_factory=list,
        description="Technical, resource, or policy constraints that limit design choices",
    )

    existing_context: str = Field(
        default="",
        description="Summary of existing codebase, systems, or integrations (if any)",
    )

    stakeholders: list[str] = Field(
        default_factory=list,
        description="Key stakeholders and their concerns (user types, teams, etc.)",
    )

    scope_boundaries: dict[str, list[str]] = Field(
        default_factory=dict,
        description="What's in scope vs out of scope, helps prevent scope creep",
    )

    existing_files: list[str] = Field(
        default_factory=list,
        description="Key existing files relevant to this task (from codebase analysis)",
    )

    needed_artifacts: list[str] = Field(
        default_factory=list,
        description="Concrete files this project must produce (e.g. config.yaml, schema.sql)",
    )

    assumptions: list[Assumption] = Field(
        default_factory=list,
        description="Explicit assumptions made due to ambiguity in the request",
    )
