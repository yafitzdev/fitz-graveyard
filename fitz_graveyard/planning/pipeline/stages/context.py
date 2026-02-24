# fitz_graveyard/planning/pipeline/stages/context.py
"""
Context understanding stage: Extract project requirements and constraints.
"""

from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import PipelineStage, extract_json
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas import ContextOutput


class ContextStage(PipelineStage):
    """
    First stage: Understand project context.

    Extracts:
    - Project description
    - Key requirements
    - Constraints
    - Stakeholders
    - Scope boundaries

    Reads pre-gathered codebase context from prior_outputs['_gathered_context']
    (set by AgentContextGatherer before stage execution begins).
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
        """Build context understanding prompt with pre-gathered agent context."""
        gathered_context = self._get_gathered_context(prior_outputs)

        prompt_template = load_prompt("context")
        prompt = prompt_template.format(
            description=job_description, krag_context=gathered_context
        )

        return [{"role": "user", "content": prompt}]

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse context output into ContextOutput schema."""
        data = extract_json(raw_output)
        context = ContextOutput(**data)
        return context.model_dump()
