# fitz_graveyard/planning/pipeline/stages/roadmap.py
"""
Implementation roadmap stage: Phase planning with dependencies.
"""

import json
import logging
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import PipelineStage, StageResult, extract_json
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas import RoadmapOutput

logger = logging.getLogger(__name__)

_SCHEMA = """{
  "phases": [
    {
      "number": 1,
      "name": "Phase Name",
      "objective": "What this phase achieves — outcome, not activity",
      "deliverables": ["specific demo-able or testable behavior"],
      "dependencies": [],
      "estimated_complexity": "low|medium|high",
      "key_risks": ["specific risk that could delay this phase"]
    }
  ],
  "critical_path": [1, 2, 4],
  "parallel_opportunities": [[3, 5]],
  "total_phases": 5
}"""


class RoadmapStage(PipelineStage):
    """
    Fourth stage: Implementation roadmap.

    Uses two-pass prompting: reason about sequencing first, then extract structured output.
    """

    @property
    def name(self) -> str:
        return "roadmap"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.65, 0.80)

    def build_prompt(self, job_description: str, prior_outputs: dict[str, Any]) -> list[dict]:
        prompt_template = load_prompt("roadmap")

        context_str = ""
        if "context" in prior_outputs:
            ctx = prior_outputs["context"]
            context_str = f"Project: {ctx.get('project_description', '')}\nScope: {json.dumps(ctx.get('scope_boundaries', {}), indent=2)}"

        architecture_str = ""
        if "architecture" in prior_outputs:
            arch = prior_outputs["architecture"]
            architecture_str = f"Approach: {arch.get('recommended', '')}"

        design_str = ""
        if "design" in prior_outputs:
            design = prior_outputs["design"]
            components = [c.get("name", "") for c in design.get("components", [])]
            adrs = [a.get("title", "") for a in design.get("adrs", [])]
            design_str = f"Components: {', '.join(components)}\nKey decisions: {', '.join(adrs)}"

        krag_context = self._get_gathered_context(prior_outputs)
        prompt = prompt_template.format(
            description=job_description,
            context=context_str.strip(),
            architecture=architecture_str.strip(),
            design=design_str.strip(),
            krag_context=krag_context,
        )
        return self._make_messages(prompt)

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        data = extract_json(raw_output)
        roadmap = RoadmapOutput(**data)
        return roadmap.model_dump()

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
