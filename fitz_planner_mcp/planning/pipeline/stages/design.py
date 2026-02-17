# fitz_planner_mcp/planning/pipeline/stages/design.py
"""
Design decisions stage: Document ADRs and component designs.
"""

import json
from typing import Any

from fitz_planner_mcp.planning.pipeline.stages.base import PipelineStage, extract_json
from fitz_planner_mcp.planning.prompts import load_prompt
from fitz_planner_mcp.planning.schemas import DesignOutput


class DesignStage(PipelineStage):
    """
    Third stage: Design decisions.

    Outputs:
    - Architectural Decision Records (ADRs)
    - Component designs
    - Data model
    - Integration points
    """

    @property
    def name(self) -> str:
        return "design"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.45, 0.65)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any]
    ) -> list[dict]:
        """Build design decisions prompt with prior stage outputs."""
        prompt_template = load_prompt("design")

        # Include context and architecture
        context_str = ""
        if "context" in prior_outputs:
            context = prior_outputs["context"]
            context_str = f"""
Project: {context.get('project_description', '')}
Requirements: {', '.join(context.get('key_requirements', [])[:5])}
"""

        architecture_str = ""
        if "architecture" in prior_outputs:
            arch = prior_outputs["architecture"]
            architecture_str = f"""
Recommended Approach: {arch.get('recommended', '')}
Reasoning: {arch.get('reasoning', '')}
Key Tradeoffs: {json.dumps(arch.get('key_tradeoffs', {}), indent=2)}
"""

        prompt = prompt_template.format(
            description=job_description,
            context=context_str.strip(),
            architecture=architecture_str.strip(),
        )

        return [{"role": "user", "content": prompt}]

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse design output into DesignOutput schema."""
        data = extract_json(raw_output)
        # Validate with Pydantic
        design = DesignOutput(**data)
        # Return as dict for checkpoint serialization
        return design.model_dump()
