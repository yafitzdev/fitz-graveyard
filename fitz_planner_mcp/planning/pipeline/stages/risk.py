# fitz_planner_mcp/planning/pipeline/stages/risk.py
"""
Risk analysis stage: Identify risks with mitigation and contingency.
"""

import json
from typing import Any

from fitz_planner_mcp.planning.pipeline.stages.base import PipelineStage, extract_json
from fitz_planner_mcp.planning.prompts import load_prompt
from fitz_planner_mcp.planning.schemas import RiskOutput


class RiskStage(PipelineStage):
    """
    Fifth stage: Risk analysis.

    Outputs:
    - Identified risks with impact/likelihood
    - Mitigation strategies
    - Contingency plans
    - Overall risk level
    """

    @property
    def name(self) -> str:
        return "risk"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.80, 0.95)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any]
    ) -> list[dict]:
        """Build risk analysis prompt with prior stage outputs."""
        prompt_template = load_prompt("risk")

        # Include all prior stages
        context_str = ""
        if "context" in prior_outputs:
            context = prior_outputs["context"]
            context_str = f"""
Constraints: {', '.join(context.get('constraints', []))}
Stakeholders: {', '.join(context.get('stakeholders', []))}
"""

        architecture_str = ""
        if "architecture" in prior_outputs:
            arch = prior_outputs["architecture"]
            tradeoffs = list(arch.get("key_tradeoffs", {}).keys())
            architecture_str = f"""
Approach: {arch.get('recommended', '')}
Key Tradeoffs: {', '.join(tradeoffs)}
"""

        design_str = ""
        if "design" in prior_outputs:
            design = prior_outputs["design"]
            integrations = design.get("integration_points", [])
            design_str = f"Integration Points: {', '.join(integrations)}"

        roadmap_str = ""
        if "roadmap" in prior_outputs:
            roadmap = prior_outputs["roadmap"]
            roadmap_str = f"""
Total Phases: {roadmap.get('total_phases', 0)}
Critical Path: {roadmap.get('critical_path', [])}
"""

        # krag_context will be populated in Phase 5-02; empty for now
        prompt = prompt_template.format(
            description=job_description,
            context=context_str.strip(),
            architecture=architecture_str.strip(),
            design=design_str.strip(),
            roadmap=roadmap_str.strip(),
            krag_context="",
        )

        return [{"role": "user", "content": prompt}]

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse risk output into RiskOutput schema."""
        data = extract_json(raw_output)
        # Validate with Pydantic
        risk = RiskOutput(**data)
        # Return as dict for checkpoint serialization
        return risk.model_dump()
