# fitz_graveyard/planning/pipeline/stages/context.py
"""
Context understanding stage: Reasoning + per-field-group extraction.

Extracts project requirements, constraints, and assumptions. One free-form
reasoning call, then 4 small JSON extractions.
"""

import json
import logging
import time
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import (
    PipelineStage,
    StageResult,
    extract_json,
)
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas import ContextOutput

logger = logging.getLogger(__name__)

# Mini-schemas for per-field-group extraction
_FIELD_GROUPS = [
    {
        "label": "description",
        "fields": ["project_description", "key_requirements", "constraints", "existing_context"],
        "schema": json.dumps({
            "project_description": "1-3 sentence specific description of what is being built",
            "key_requirements": ["concrete testable requirement 1", "requirement 2"],
            "constraints": ["real binding constraint 1", "constraint 2"],
            "existing_context": "existing codebase or tech context, or empty string if none",
        }, indent=2),
    },
    {
        "label": "stakeholders",
        "fields": ["stakeholders", "scope_boundaries"],
        "schema": json.dumps({
            "stakeholders": ["stakeholder with specific concern, not just job title"],
            "scope_boundaries": {
                "in_scope": ["specific feature or capability"],
                "out_of_scope": ["explicitly excluded feature"],
            },
        }, indent=2),
    },
    {
        "label": "files",
        "fields": ["existing_files", "needed_artifacts"],
        "schema": json.dumps({
            "existing_files": ["path/to/relevant/file.py — what it does"],
            "needed_artifacts": ["config.yaml — the config file this project must produce"],
        }, indent=2),
    },
    {
        "label": "assumptions",
        "fields": ["assumptions"],
        "schema": json.dumps({
            "assumptions": [
                {"assumption": "what you assumed", "impact": "what changes if wrong", "confidence": "low|medium|high"}
            ],
        }, indent=2),
    },
]


class ContextStage(PipelineStage):
    """
    First stage: Understand project context.

    Uses reasoning + per-field-group extraction for reliability with small models.
    """

    @property
    def name(self) -> str:
        return "context"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.10, 0.25)

    def build_prompt(self, job_description: str, prior_outputs: dict[str, Any]) -> list[dict]:
        # Use raw summaries for reasoning — more detail for accurate file identification
        gathered_context = self._get_raw_summaries(prior_outputs)
        prompt_template = load_prompt("context")
        prompt = prompt_template.format(description=job_description, krag_context=gathered_context)
        return self._make_messages(prompt)

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        data = extract_json(raw_output)
        context = ContextOutput(**data)
        return context.model_dump()

    async def execute(self, client: Any, job_description: str, prior_outputs: dict[str, Any]) -> StageResult:
        try:
            # 1. Reasoning pass
            messages = self.build_prompt(job_description, prior_outputs)
            await self._report_substep("reasoning")
            t0 = time.monotonic()
            reasoning = await client.generate(messages=messages)
            t1 = time.monotonic()
            logger.info(f"Stage '{self.name}': reasoning took {t1 - t0:.1f}s ({len(reasoning)} chars)")

            # 2. Self-critique pass
            krag_context = self._get_gathered_context(prior_outputs)
            reasoning = await self._self_critique(
                client, reasoning, job_description, krag_context=krag_context,
            )

            # 3. Per-field-group extraction
            # files group needs codebase context for accurate path identification
            _CONTEXT_GROUPS = {"files", "description"}

            merged: dict[str, Any] = {}
            for group in _FIELD_GROUPS:
                extra = krag_context if group["label"] in _CONTEXT_GROUPS else ""
                partial = await self._extract_field_group(
                    client,
                    reasoning,
                    group["fields"],
                    group["schema"],
                    group["label"],
                    extra_context=extra,
                )
                merged.update(partial)

            # 4. Parse through existing parse_output (handles defaults + Pydantic validation)
            parsed = self.parse_output(json.dumps(merged))
            return StageResult(
                stage_name=self.name,
                success=True,
                output=parsed,
                raw_output=reasoning,
            )
        except Exception as e:
            logger.error(f"Stage '{self.name}' failed: {e}", exc_info=True)
            return StageResult(stage_name=self.name, success=False, output={}, raw_output="", error=str(e))
