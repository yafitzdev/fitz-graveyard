# fitz_graveyard/planning/pipeline/stages/context.py
"""
Context understanding stage: Extract project requirements and constraints.
"""

import logging
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import PipelineStage, StageResult, extract_json
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas import ContextOutput

logger = logging.getLogger(__name__)

_SCHEMA = """{
  "project_description": "1-3 sentence specific description of what is being built",
  "key_requirements": ["concrete testable requirement 1", "requirement 2"],
  "constraints": ["real binding constraint 1", "constraint 2"],
  "existing_context": "existing codebase or tech context, or empty string if none",
  "stakeholders": ["stakeholder with specific concern, not just job title"],
  "scope_boundaries": {
    "in_scope": ["specific feature or capability"],
    "out_of_scope": ["explicitly excluded feature"]
  }
}"""


class ContextStage(PipelineStage):
    """
    First stage: Understand project context.

    Uses two-pass prompting: reason about requirements first, then extract structured output.
    """

    @property
    def name(self) -> str:
        return "context"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.10, 0.25)

    def build_prompt(self, job_description: str, prior_outputs: dict[str, Any]) -> list[dict]:
        gathered_context = self._get_gathered_context(prior_outputs)
        prompt_template = load_prompt("context")
        prompt = prompt_template.format(description=job_description, krag_context=gathered_context)
        return self._make_messages(prompt)

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        data = extract_json(raw_output)
        context = ContextOutput(**data)
        return context.model_dump()

    async def execute(self, client: Any, job_description: str, prior_outputs: dict[str, Any]) -> StageResult:
        try:
            reasoning_messages = self.build_prompt(job_description, prior_outputs)
            reasoning, json_output = await self._two_pass(client, reasoning_messages, _SCHEMA)
            parsed = self.parse_output(json_output)
            raw = f"=== REASONING ===\n{reasoning}\n\n=== STRUCTURED ===\n{json_output}"
            return StageResult(stage_name=self.name, success=True, output=parsed, raw_output=raw)
        except Exception as e:
            logger.error(f"Stage '{self.name}' failed: {e}", exc_info=True)
            return StageResult(stage_name=self.name, success=False, output={}, raw_output="", error=str(e))
