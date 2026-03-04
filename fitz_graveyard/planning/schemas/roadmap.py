# fitz_graveyard/planning/schemas/roadmap.py
"""Schema for implementation roadmap stage output."""

import re
from typing import Annotated

from pydantic import BaseModel, Field, ConfigDict, BeforeValidator, model_validator


def _coerce_phase_number(value: object) -> int:
    """Coerce LLM phase number variants to int.

    Handles: 1, "1", "Phase 1", "phase_0", etc.
    """
    if isinstance(value, int):
        return value
    m = re.search(r"\d+", str(value))
    if m:
        return int(m.group())
    raise ValueError(f"Cannot extract phase number from {value!r}")


PhaseRef = Annotated[int, BeforeValidator(_coerce_phase_number)]
"""Int that auto-coerces LLM variants like ``"Phase 1"`` to ``1``."""


class Phase(BaseModel):
    """A single implementation phase in the roadmap."""

    model_config = ConfigDict(extra="ignore")

    number: PhaseRef = Field(
        ...,
        description="Phase number (1-indexed)",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_field_names(cls, data: dict) -> dict:
        """Handle LLM field name variations (e.g. 'num' instead of 'number')."""
        if isinstance(data, dict) and "num" in data and "number" not in data:
            data["number"] = data.pop("num")
        return data

    name: str = Field(
        ...,
        description="Short descriptive name for this phase",
    )

    objective: str = Field(
        default="",
        description="What this phase achieves",
    )

    deliverables: list[str] = Field(
        default_factory=list,
        description="Concrete outputs from this phase",
    )

    dependencies: list[PhaseRef] = Field(
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

    verification_command: str = Field(
        default="",
        description="Exact command or check to verify phase completion",
    )

    estimated_effort: str = Field(
        default="",
        description="Rough time estimate (e.g., '~1 hour', '~30 min')",
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

    critical_path: list[PhaseRef] = Field(
        default_factory=list,
        description="Phase numbers that form the critical path (must be sequential)",
    )

    parallel_opportunities: list[list[PhaseRef]] = Field(
        default_factory=list,
        description="Groups of phase numbers that can be executed in parallel",
    )

    total_phases: int = Field(
        default=0,
        description="Total number of phases in the roadmap",
    )
