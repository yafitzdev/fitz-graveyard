# fitz_graveyard/planning/pipeline/stages/architecture.py
"""
Architecture exploration stage: Two-stage reasoning (free-form then structured).
"""

import json
import logging
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import (
    PipelineStage,
    StageResult,
    extract_json,
)
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas import ArchitectureOutput

logger = logging.getLogger(__name__)

_SCHEMA = """{
  "approaches": [
    {
      "name": "Approach Name",
      "description": "What it looks like in production with specific technologies",
      "pros": ["specific advantage 1"],
      "cons": ["specific disadvantage or failure mode 1"],
      "complexity": "low|medium|high",
      "best_for": ["scenario where this is the right choice"]
    }
  ],
  "recommended": "must match one approach name exactly",
  "reasoning": "why this approach is right and the others are wrong for this project",
  "key_tradeoffs": {"tradeoff_name": "specific description of the tradeoff"},
  "technology_considerations": ["specific technology or library with reason"]
}"""


class ArchitectureStage(PipelineStage):
    """
    Second stage: Explore architectural approaches.

    Uses two-stage prompting:
    1. Free-form reasoning about approaches (no JSON constraint)
    2. Format reasoning into structured JSON via multi-turn continuation
    """

    @property
    def name(self) -> str:
        return "architecture"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.25, 0.45)

    def build_prompt(self, job_description: str, prior_outputs: dict[str, Any]) -> list[dict]:
        prompt_template = load_prompt("architecture")

        context_str = job_description
        if "context" in prior_outputs:
            context = prior_outputs["context"]
            context_str = f"""Project: {job_description}

Description: {context.get('project_description', '')}
Requirements: {', '.join(context.get('key_requirements', []))}
Constraints: {', '.join(context.get('constraints', []))}"""

        krag_context = self._get_gathered_context(prior_outputs)
        prompt = prompt_template.format(context=context_str.strip(), krag_context=krag_context)
        return self._make_messages(prompt)

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        data = extract_json(raw_output)
        architecture = ArchitectureOutput(**data)
        return architecture.model_dump()

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
