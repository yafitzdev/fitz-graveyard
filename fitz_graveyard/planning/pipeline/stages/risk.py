# fitz_graveyard/planning/pipeline/stages/risk.py
"""
Risk analysis stage: Identify risks with mitigation and contingency.
"""

import json
import logging
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import PipelineStage, StageResult, extract_json
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas import RiskOutput

logger = logging.getLogger(__name__)

_SCHEMA = """{
  "risks": [
    {
      "category": "technical|external|resource|schedule|quality|security",
      "description": "Specific description of what could go wrong in this project",
      "impact": "low|medium|high|critical",
      "likelihood": "low|medium|high",
      "mitigation": "Specific mitigation action — not generic advice",
      "contingency": "What to do if it happens despite mitigation",
      "affected_phases": [1, 3]
    }
  ],
  "overall_risk_level": "low|medium|high",
  "recommended_contingencies": ["specific contingency action"]
}"""


class RiskStage(PipelineStage):
    """
    Fifth stage: Risk analysis.

    Uses two-pass prompting: reason about risks first, then extract structured output.
    """

    @property
    def name(self) -> str:
        return "risk"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.80, 0.95)

    def build_prompt(self, job_description: str, prior_outputs: dict[str, Any]) -> list[dict]:
        prompt_template = load_prompt("risk")

        context_str = ""
        if "context" in prior_outputs:
            ctx = prior_outputs["context"]
            context_str = f"Constraints: {', '.join(ctx.get('constraints', []))}\nStakeholders: {', '.join(ctx.get('stakeholders', []))}"

        architecture_str = ""
        if "architecture" in prior_outputs:
            arch = prior_outputs["architecture"]
            tradeoffs = list(arch.get("key_tradeoffs", {}).keys())
            architecture_str = f"Approach: {arch.get('recommended', '')}\nKey Tradeoffs: {', '.join(tradeoffs)}"

        design_str = ""
        if "design" in prior_outputs:
            design = prior_outputs["design"]
            integrations = design.get("integration_points", [])
            design_str = f"Integration Points: {', '.join(integrations)}"

        roadmap_str = ""
        if "roadmap" in prior_outputs:
            roadmap = prior_outputs["roadmap"]
            roadmap_str = f"Total Phases: {roadmap.get('total_phases', 0)}\nCritical Path: {roadmap.get('critical_path', [])}"

        krag_context = self._get_gathered_context(prior_outputs)
        prompt = prompt_template.format(
            description=job_description,
            context=context_str.strip(),
            architecture=architecture_str.strip(),
            design=design_str.strip(),
            roadmap=roadmap_str.strip(),
            krag_context=krag_context,
        )
        return self._make_messages(prompt)

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        data = extract_json(raw_output)
        risk = RiskOutput(**data)
        return risk.model_dump()

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
