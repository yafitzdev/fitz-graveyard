# fitz_graveyard/planning/pipeline/stages/roadmap.py
"""
Implementation roadmap stage: Phase planning with dependencies.
"""

import json
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import PipelineStage, extract_json
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas import RoadmapOutput


class RoadmapStage(PipelineStage):
    """
    Fourth stage: Implementation roadmap.

    Outputs:
    - Phases with deliverables
    - Dependencies between phases
    - Critical path
    - Parallel opportunities
    """

    @property
    def name(self) -> str:
        return "roadmap"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.65, 0.80)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any]
    ) -> list[dict]:
        """Build roadmap prompt with prior stage outputs and KRAG context."""
        prompt_template = load_prompt("roadmap")

        # Include context, architecture, and design
        context_str = ""
        if "context" in prior_outputs:
            context = prior_outputs["context"]
            context_str = f"""
Project: {context.get('project_description', '')}
Scope: {json.dumps(context.get('scope_boundaries', {}), indent=2)}
"""

        architecture_str = ""
        if "architecture" in prior_outputs:
            arch = prior_outputs["architecture"]
            architecture_str = f"Approach: {arch.get('recommended', '')}"

        design_str = ""
        if "design" in prior_outputs:
            design = prior_outputs["design"]
            components = [c.get("name", "") for c in design.get("components", [])]
            design_str = f"""
Components: {', '.join(components)}
ADR Count: {len(design.get('adrs', []))}
"""

        krag_context = self._get_gathered_context(prior_outputs)

        prompt = prompt_template.format(
            description=job_description,
            context=context_str.strip(),
            architecture=architecture_str.strip(),
            design=design_str.strip(),
            krag_context=krag_context,
        )

        return [{"role": "user", "content": prompt}]

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse roadmap output into RoadmapOutput schema."""
        data = extract_json(raw_output)
        # Validate with Pydantic
        roadmap = RoadmapOutput(**data)
        # Return as dict for checkpoint serialization
        return roadmap.model_dump()
