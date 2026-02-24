# fitz_graveyard/planning/pipeline/stages/design.py
"""
Design decisions stage: Document ADRs and component designs.
"""

import json
import logging
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import PipelineStage, StageResult, extract_json
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas import DesignOutput

logger = logging.getLogger(__name__)

_SCHEMA = """{
  "adrs": [
    {
      "title": "ADR: Decision Title",
      "context": "What problem this decision solves",
      "decision": "What was decided",
      "rationale": "Why this is the right choice for this project",
      "consequences": ["positive consequence", "negative consequence or tradeoff"],
      "alternatives_considered": ["Alternative X — rejected because specific reason"]
    }
  ],
  "components": [
    {
      "name": "ComponentName",
      "purpose": "What it does in one sentence",
      "responsibilities": ["specific responsibility"],
      "interfaces": ["methodName(param: Type) -> ReturnType"],
      "dependencies": ["OtherComponent", "ExternalService"]
    }
  ],
  "data_model": {
    "EntityName": ["field_name: type", "other_field: type"]
  },
  "integration_points": ["ExternalSystem — what we use it for and how"]
}"""


class DesignStage(PipelineStage):
    """
    Third stage: Design decisions.

    Uses two-pass prompting: reason about design first, then extract structured output.
    """

    @property
    def name(self) -> str:
        return "design"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.45, 0.65)

    def build_prompt(self, job_description: str, prior_outputs: dict[str, Any]) -> list[dict]:
        prompt_template = load_prompt("design")

        context_str = ""
        if "context" in prior_outputs:
            ctx = prior_outputs["context"]
            context_str = f"Project: {ctx.get('project_description', '')}\nRequirements: {', '.join(ctx.get('key_requirements', [])[:5])}\nConstraints: {', '.join(ctx.get('constraints', []))}"

        architecture_str = ""
        if "architecture" in prior_outputs:
            arch = prior_outputs["architecture"]
            architecture_str = f"Recommended: {arch.get('recommended', '')}\nReasoning: {arch.get('reasoning', '')}\nKey Tradeoffs: {json.dumps(arch.get('key_tradeoffs', {}), indent=2)}"

        krag_context = self._get_gathered_context(prior_outputs)
        prompt = prompt_template.format(
            description=job_description,
            context=context_str.strip(),
            architecture=architecture_str.strip(),
            krag_context=krag_context,
        )
        return self._make_messages(prompt)

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        data = extract_json(raw_output)
        design = DesignOutput(**data)
        return design.model_dump()

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
