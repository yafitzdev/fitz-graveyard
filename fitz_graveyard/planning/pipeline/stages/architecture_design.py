# fitz_graveyard/planning/pipeline/stages/architecture_design.py
"""
Merged architecture + design stage: Reasoning + per-field-group extraction.

Combines architectural exploration and design decisions. One free-form
reasoning call, then 6 small JSON extractions — each producing a tiny
schema that a small model can handle reliably.
"""

import difflib
import json
import logging
import time
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import (
    SYSTEM_PROMPT,
    PipelineStage,
    StageResult,
    extract_json,
)
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas import ArchitectureOutput, DesignOutput

logger = logging.getLogger(__name__)

_ARCH_FIELDS = {"approaches", "recommended", "reasoning", "key_tradeoffs", "technology_considerations", "scope_statement"}
_DESIGN_FIELDS = {"adrs", "components", "data_model", "integration_points", "artifacts"}

# Mini-schemas for per-field-group extraction (small enough for 3B models)
_FIELD_GROUPS = [
    {
        "label": "approaches",
        "fields": ["approaches", "recommended", "reasoning", "scope_statement"],
        "schema": json.dumps({
            "approaches": [
                {
                    "name": "Approach Name",
                    "description": "What it looks like in production",
                    "pros": ["advantage"],
                    "cons": ["disadvantage"],
                    "complexity": "low|medium|high",
                    "best_for": ["scenario"],
                }
            ],
            "recommended": "must match one approach name exactly",
            "reasoning": "why this approach is right",
            "scope_statement": "1-2 sentences characterizing the effort",
        }, indent=2),
    },
    {
        "label": "tradeoffs",
        "fields": ["key_tradeoffs", "technology_considerations"],
        "schema": json.dumps({
            "key_tradeoffs": {"tradeoff_name": "description"},
            "technology_considerations": ["technology with reason"],
        }, indent=2),
    },
    {
        "label": "adrs",
        "fields": ["adrs"],
        "schema": json.dumps({
            "adrs": [
                {
                    "title": "ADR: Decision Title",
                    "context": "What problem this solves",
                    "decision": "What was decided",
                    "rationale": "Why this is right",
                    "consequences": ["consequence"],
                    "alternatives_considered": ["Alternative — rejected because reason"],
                }
            ],
        }, indent=2),
    },
    {
        "label": "components",
        "fields": ["components", "data_model"],
        "schema": json.dumps({
            "components": [
                {
                    "name": "ComponentName",
                    "purpose": "What it does",
                    "responsibilities": ["responsibility"],
                    "interfaces": ["methodName(param: Type) -> ReturnType"],
                    "dependencies": ["OtherComponent"],
                }
            ],
            "data_model": {"EntityName": ["field: type"]},
        }, indent=2),
    },
    {
        "label": "integrations",
        "fields": ["integration_points"],
        "schema": json.dumps({
            "integration_points": ["ExternalSystem — what and how"],
        }, indent=2),
    },
    {
        "label": "artifacts",
        "fields": ["artifacts"],
        "schema": json.dumps({
            "artifacts": [
                {
                    "filename": "path/to/file",
                    "content": "complete file content",
                    "purpose": "why this artifact exists",
                }
            ],
        }, indent=2),
    },
]


class ArchitectureDesignStage(PipelineStage):
    """
    Merged stage: Architecture exploration + design decisions.

    Uses reasoning + per-field-group extraction for reliability with small models.
    Returns split output: {"architecture": {...}, "design": {...}}.
    """

    @property
    def name(self) -> str:
        return "architecture_design"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.25, 0.65)

    def build_prompt(self, job_description: str, prior_outputs: dict[str, Any]) -> list[dict]:
        prompt_template = load_prompt("architecture_design")

        context_str = job_description
        if "context" in prior_outputs:
            context = prior_outputs["context"]
            parts = [
                f"Project: {job_description}",
                f"Description: {context.get('project_description', '')}",
                f"Requirements: {', '.join(context.get('key_requirements', []))}",
                f"Constraints: {', '.join(context.get('constraints', []))}",
            ]
            if context.get("existing_files"):
                parts.append(f"Existing files: {', '.join(context['existing_files'])}")
            if context.get("needed_artifacts"):
                parts.append(f"Expected deliverable files: {', '.join(context['needed_artifacts'])}")
            context_str = "\n".join(parts)

        # Use raw summaries for reasoning prompt — more detail for accurate architecture
        krag_context = self._get_raw_summaries(prior_outputs)
        prompt = prompt_template.format(context=context_str.strip(), krag_context=krag_context)
        return self._make_messages(prompt)

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        data = extract_json(raw_output)

        # Fix recommended approach name
        approach_names = [a["name"] for a in data.get("approaches", [])]
        recommended = data.get("recommended", "")
        if recommended not in approach_names and approach_names:
            matches = difflib.get_close_matches(recommended, approach_names, n=1, cutoff=0.4)
            if matches:
                logger.warning(f"Fuzzy-matched recommended '{recommended}' → '{matches[0]}'")
                data["recommended"] = matches[0]
            else:
                logger.warning(f"recommended '{recommended}' not in approaches, using first: '{approach_names[0]}'")
                data["recommended"] = approach_names[0]

        # Split into architecture and design sub-outputs
        _arch_str_fields = {"reasoning", "recommended", "scope_statement"}
        _arch_dict_fields = {"key_tradeoffs"}
        arch_data = {
            k: data.get(k, "" if k in _arch_str_fields else {} if k in _arch_dict_fields else [])
            for k in _ARCH_FIELDS
        }
        # Ensure defaults for missing arch fields
        arch_data.setdefault("approaches", [])
        arch_data.setdefault("key_tradeoffs", {})
        arch_data.setdefault("technology_considerations", [])

        design_data = {k: data.get(k, [] if k != "data_model" else {}) for k in _DESIGN_FIELDS}
        design_data.setdefault("adrs", [])
        design_data.setdefault("components", [])
        design_data.setdefault("data_model", {})
        design_data.setdefault("integration_points", [])
        design_data.setdefault("artifacts", [])

        # Validate through Pydantic
        arch = ArchitectureOutput(**arch_data)
        design = DesignOutput(**design_data)

        return {
            "architecture": arch.model_dump(),
            "design": design.model_dump(),
        }

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
            # Selective context: only groups that need codebase evidence get it
            _CONTEXT_GROUPS = {"approaches", "adrs", "artifacts"}

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
