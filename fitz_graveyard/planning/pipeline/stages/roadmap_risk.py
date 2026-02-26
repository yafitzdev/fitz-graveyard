# fitz_graveyard/planning/pipeline/stages/roadmap_risk.py
"""
Merged roadmap + risk stage: Reasoning + per-field-group extraction.

Combines implementation roadmap and risk assessment. One free-form
reasoning call, then 3 small JSON extractions.
"""

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
from fitz_graveyard.planning.schemas import RoadmapOutput, RiskOutput

logger = logging.getLogger(__name__)

_ROADMAP_FIELDS = {"phases", "critical_path", "parallel_opportunities", "total_phases"}
_RISK_FIELDS = {"risks", "overall_risk_level", "recommended_contingencies"}

# Mini-schemas for per-field-group extraction
_FIELD_GROUPS = [
    {
        "label": "phases",
        "fields": ["phases"],
        "schema": json.dumps({
            "phases": [
                {
                    "number": 1,
                    "name": "Phase Name",
                    "objective": "What this phase achieves",
                    "deliverables": ["specific deliverable"],
                    "dependencies": [],
                    "estimated_complexity": "low|medium|high",
                    "key_risks": ["risk"],
                    "verification_command": "pytest tests/test_something.py -v",
                    "estimated_effort": "~2 hours",
                }
            ],
        }, indent=2),
    },
    {
        "label": "scheduling",
        "fields": ["critical_path", "parallel_opportunities", "total_phases"],
        "schema": json.dumps({
            "critical_path": [1, 2, 4],
            "parallel_opportunities": [[3, 5]],
            "total_phases": 5,
        }, indent=2),
    },
    {
        "label": "risks",
        "fields": ["risks", "overall_risk_level", "recommended_contingencies"],
        "schema": json.dumps({
            "risks": [
                {
                    "category": "technical|external|resource|schedule|quality|security",
                    "description": "What could go wrong",
                    "impact": "low|medium|high|critical",
                    "likelihood": "low|medium|high",
                    "mitigation": "Specific mitigation action",
                    "contingency": "What to do if it happens",
                    "affected_phases": [1, 3],
                    "verification": "assert something",
                }
            ],
            "overall_risk_level": "low|medium|high",
            "recommended_contingencies": ["contingency action"],
        }, indent=2),
    },
]


def _remove_dependency_cycles(phases: list[dict]) -> list[dict]:
    """Remove back-edges from phase dependencies.

    Dependencies should only point to earlier phases. Removes any dependency
    where dep_num >= phase_num or dep_num doesn't exist.
    """
    phase_nums = {p["number"] for p in phases}
    for phase in phases:
        original = phase.get("dependencies", [])
        cleaned = [d for d in original if d < phase["number"] and d in phase_nums]
        if cleaned != original:
            removed = set(original) - set(cleaned)
            logger.warning(f"Phase {phase['number']}: removed invalid deps {removed}")
            phase["dependencies"] = cleaned
    return phases


class RoadmapRiskStage(PipelineStage):
    """
    Merged stage: Implementation roadmap + risk assessment.

    Uses reasoning + per-field-group extraction for reliability with small models.
    Returns split output: {"roadmap": {...}, "risk": {...}}.
    """

    @property
    def name(self) -> str:
        return "roadmap_risk"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.65, 0.95)

    def build_prompt(self, job_description: str, prior_outputs: dict[str, Any]) -> list[dict]:
        prompt_template = load_prompt("roadmap_risk")

        context_str = ""
        if "context" in prior_outputs:
            ctx = prior_outputs["context"]
            context_str = (
                f"Project: {ctx.get('project_description', '')}\n"
                f"Requirements: {', '.join(ctx.get('key_requirements', []))}\n"
                f"Constraints: {', '.join(ctx.get('constraints', []))}\n"
                f"Scope: {json.dumps(ctx.get('scope_boundaries', {}), indent=2)}"
            )

        architecture_design_str = ""
        if "architecture" in prior_outputs:
            arch = prior_outputs["architecture"]
            architecture_design_str += (
                f"Recommended approach: {arch.get('recommended', '')}\n"
                f"Reasoning: {arch.get('reasoning', '')}\n"
                f"Key Tradeoffs: {json.dumps(arch.get('key_tradeoffs', {}), indent=2)}\n"
            )
        if "design" in prior_outputs:
            design = prior_outputs["design"]

            # Full ADR details so roadmap can reference decisions
            adrs = design.get("adrs", [])
            if adrs:
                architecture_design_str += "\nKey Decisions (ADRs):\n"
                for adr in adrs:
                    title = adr.get("title", "")
                    decision = adr.get("decision", "")
                    rationale = adr.get("rationale", "")
                    architecture_design_str += f"- {title}: {decision} ({rationale})\n"

            # Full component details
            components = design.get("components", [])
            if components:
                architecture_design_str += "\nComponents:\n"
                for comp in components:
                    name = comp.get("name", "")
                    purpose = comp.get("purpose", "")
                    interfaces = ", ".join(comp.get("interfaces", []))
                    deps = ", ".join(comp.get("dependencies", []))
                    architecture_design_str += f"- {name}: {purpose}"
                    if interfaces:
                        architecture_design_str += f" [interfaces: {interfaces}]"
                    if deps:
                        architecture_design_str += f" [depends on: {deps}]"
                    architecture_design_str += "\n"

            integrations = design.get("integration_points", [])
            if integrations:
                architecture_design_str += f"\nIntegration Points: {', '.join(integrations)}\n"

            # Include artifacts list so roadmap knows what files to produce
            artifacts = design.get("artifacts", [])
            if artifacts:
                artifact_names = [a.get("filename", "") for a in artifacts]
                architecture_design_str += f"\nDesign Artifacts: {', '.join(artifact_names)}\n"

        # Use raw summaries for reasoning prompt — more detail for accurate roadmap
        krag_context = self._get_raw_summaries(prior_outputs)
        prompt = prompt_template.format(
            description=job_description,
            context=context_str.strip(),
            architecture_design=architecture_design_str.strip(),
            krag_context=krag_context,
        )
        return self._make_messages(prompt)

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        data = extract_json(raw_output)

        # Normalize LLM field name variations before _remove_dependency_cycles
        if "phases" in data:
            for phase in data["phases"]:
                if "num" in phase and "number" not in phase:
                    phase["number"] = phase.pop("num")
            data["phases"] = _remove_dependency_cycles(data["phases"])

        # Split into roadmap and risk sub-outputs
        roadmap_data = {k: data.get(k) for k in _ROADMAP_FIELDS if k in data}
        roadmap_data.setdefault("phases", [])
        roadmap_data.setdefault("critical_path", [])
        roadmap_data.setdefault("parallel_opportunities", [])
        roadmap_data.setdefault("total_phases", len(roadmap_data.get("phases", [])))

        risk_data = {k: data.get(k) for k in _RISK_FIELDS if k in data}
        risk_data.setdefault("risks", [])
        risk_data.setdefault("overall_risk_level", "medium")
        risk_data.setdefault("recommended_contingencies", [])

        # Validate through Pydantic
        roadmap = RoadmapOutput(**roadmap_data)
        risk = RiskOutput(**risk_data)

        return {
            "roadmap": roadmap.model_dump(),
            "risk": risk.model_dump(),
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
            # Selective context: phases needs codebase for verification commands,
            # risks needs it to ground risks in actual code structure
            _CONTEXT_GROUPS = {"phases", "risks"}

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
