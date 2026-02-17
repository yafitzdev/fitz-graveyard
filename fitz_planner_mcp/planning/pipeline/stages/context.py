# fitz_planner_mcp/planning/pipeline/stages/context.py
"""
Context understanding stage: Extract project requirements and constraints.
"""

from typing import Any

from fitz_planner_mcp.planning.pipeline.stages.base import PipelineStage, extract_json
from fitz_planner_mcp.planning.prompts import load_prompt
from fitz_planner_mcp.planning.schemas import ContextOutput


class ContextStage(PipelineStage):
    """
    First stage: Understand project context.

    Extracts:
    - Project description
    - Key requirements
    - Constraints
    - Stakeholders
    - Scope boundaries
    """

    @property
    def name(self) -> str:
        return "context"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.10, 0.25)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any]
    ) -> list[dict]:
        """Build context understanding prompt."""
        prompt_template = load_prompt("context")
        prompt = prompt_template.format(description=job_description)

        return [{"role": "user", "content": prompt}]

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse context output into ContextOutput schema."""
        data = extract_json(raw_output)
        # Validate with Pydantic
        context = ContextOutput(**data)
        # Return as dict for checkpoint serialization
        return context.model_dump()
