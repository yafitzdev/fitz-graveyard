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


class ArchitectureStage(PipelineStage):
    """
    Second stage: Explore architectural approaches.

    Uses two-stage prompting:
    1. Free-form reasoning about approaches (no JSON constraint)
    2. Format reasoning into structured JSON

    This improves reasoning quality by avoiding JSON Schema constraints
    during the exploration phase.
    """

    @property
    def name(self) -> str:
        return "architecture"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.25, 0.45)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any]
    ) -> list[dict]:
        """Build free-form architecture reasoning prompt with KRAG context."""
        # First stage: free-form reasoning (no JSON)
        prompt_template = load_prompt("architecture")

        # Include context if available
        context_str = job_description  # Default to job description
        if "context" in prior_outputs:
            context = prior_outputs["context"]
            context_str = f"""
Project: {job_description}

Project Description: {context.get('project_description', '')}
Key Requirements: {', '.join(context.get('key_requirements', []))}
Constraints: {', '.join(context.get('constraints', []))}
"""

        # Query KRAG for design patterns and architectural decisions
        krag_queries = [
            "What design patterns and abstractions are used in this codebase?",
            "What architectural decisions have been made and why?",
        ]
        krag_context = self._get_krag_context(krag_queries, prior_outputs)

        prompt = prompt_template.format(context=context_str.strip(), krag_context=krag_context)

        return [{"role": "user", "content": prompt}]

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse architecture output into ArchitectureOutput schema.

        Note: This is called by the custom execute() after formatting stage.
        """
        data = extract_json(raw_output)
        # Validate with Pydantic
        architecture = ArchitectureOutput(**data)
        # Return as dict for checkpoint serialization
        return architecture.model_dump()

    async def execute(
        self,
        client: Any,
        job_description: str,
        prior_outputs: dict[str, Any],
    ) -> StageResult:
        """
        Override execute to implement two-stage prompting.

        Stage 1: Free-form reasoning (no JSON constraint)
        Stage 2: Format reasoning into structured JSON
        """
        try:
            # Stage 1: Free-form reasoning
            messages_reasoning = self.build_prompt(job_description, prior_outputs)
            logger.info(
                f"Stage '{self.name}': Phase 1 - Free-form reasoning ({len(messages_reasoning)} messages)"
            )

            response_reasoning = await client.generate_chat(messages=messages_reasoning)
            reasoning_output = response_reasoning.content

            logger.info(
                f"Stage '{self.name}': Phase 1 complete ({len(reasoning_output)} chars)"
            )

            # Stage 2: Format into JSON
            format_template = load_prompt("architecture_format")

            # Create schema example
            schema_example = """{
  "approaches": [
    {
      "name": "string",
      "description": "string",
      "pros": ["string"],
      "cons": ["string"],
      "complexity": "low|medium|high",
      "best_for": ["string"]
    }
  ],
  "recommended": "string (must match an approach name)",
  "reasoning": "string",
  "key_tradeoffs": {"tradeoff_name": "description"},
  "technology_considerations": ["string"]
}"""

            format_prompt = format_template.format(
                reasoning=reasoning_output,
                schema=schema_example
            )

            messages_format = [{"role": "user", "content": format_prompt}]

            logger.info(
                f"Stage '{self.name}': Phase 2 - JSON formatting ({len(messages_format)} messages)"
            )

            response_format = await client.generate_chat(messages=messages_format)
            formatted_output = response_format.content

            logger.info(
                f"Stage '{self.name}': Phase 2 complete ({len(formatted_output)} chars)"
            )

            # Parse formatted output
            parsed = self.parse_output(formatted_output)

            # Store both reasoning and formatted output in raw_output
            combined_raw = f"=== REASONING ===\n{reasoning_output}\n\n=== STRUCTURED ===\n{formatted_output}"

            logger.info(f"Stage '{self.name}': Parsed output successfully")

            return StageResult(
                stage_name=self.name,
                success=True,
                output=parsed,
                raw_output=combined_raw,
            )

        except Exception as e:
            logger.error(f"Stage '{self.name}' failed: {e}", exc_info=True)
            return StageResult(
                stage_name=self.name,
                success=False,
                output={},
                raw_output="",
                error=str(e),
            )
