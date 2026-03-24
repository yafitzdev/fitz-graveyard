# fitz_graveyard/planning/schemas/decisions.py
"""Schemas for decomposed decision pipeline."""

from pydantic import BaseModel, Field, ConfigDict


class AtomicDecision(BaseModel):
    """A single atomic decision to be resolved."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(
        ...,
        description="Short unique identifier (e.g., 'd1', 'd2')",
    )

    question: str = Field(
        ...,
        description=(
            "The specific question to answer. Must be a concrete decision, "
            "not an open-ended exploration."
        ),
    )

    relevant_files: list[str] = Field(
        default_factory=list,
        description="File paths from the call graph needed to resolve this decision (1-3 files)",
    )

    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of decisions that must be resolved before this one. "
            "Each dependency's resolution will be injected as a constraint."
        ),
    )

    category: str = Field(
        default="technical",
        description=(
            "Decision category: 'interface' (API/contract change), "
            "'pattern' (architecture pattern choice), "
            "'integration' (how components connect), "
            "'scope' (what's in/out), "
            "'technical' (implementation detail)"
        ),
    )


class DecisionResolution(BaseModel):
    """The committed resolution of an atomic decision."""

    model_config = ConfigDict(extra="ignore")

    decision_id: str = Field(
        ...,
        description="ID of the AtomicDecision this resolves",
    )

    decision: str = Field(
        ...,
        description="What was decided -- a concrete, specific answer",
    )

    reasoning: str = Field(
        ...,
        description="Why this decision was made, citing specific code evidence",
    )

    evidence: list[str] = Field(
        default_factory=list,
        description=(
            "Specific code citations supporting this decision "
            "(e.g., 'synthesizer.py:generate() returns str, not Iterator')"
        ),
    )

    constraints_for_downstream: list[str] = Field(
        default_factory=list,
        description=(
            "Constraints this decision imposes on downstream decisions. "
            "These will be injected into the prompts of depending decisions."
        ),
    )


class DecisionDecompositionOutput(BaseModel):
    """Output of the decision decomposition stage."""

    model_config = ConfigDict(extra="ignore")

    decisions: list[AtomicDecision] = Field(
        default_factory=list,
        description="Ordered list of atomic decisions to resolve",
    )


class DecisionResolutionOutput(BaseModel):
    """Output of the per-decision resolution stage (all decisions)."""

    model_config = ConfigDict(extra="ignore")

    resolutions: list[DecisionResolution] = Field(
        default_factory=list,
        description="Committed resolutions for each atomic decision",
    )
