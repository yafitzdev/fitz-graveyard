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

    Phase number coercion (e.g. ``"Phase 1"`` → ``1``) is handled by
    ``PhaseRef`` in the Pydantic schema, so this only needs to validate
    dependency ordering on already-coerced ints.
    """
    from fitz_graveyard.planning.schemas.roadmap import _coerce_phase_number

    for phase in phases:
        try:
            phase["number"] = _coerce_phase_number(phase.get("number", 0))
        except ValueError:
            pass
    phase_nums = {p["number"] for p in phases}
    for phase in phases:
        original = []
        for d in phase.get("dependencies", []):
            try:
                original.append(_coerce_phase_number(d))
            except ValueError:
                continue
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

    Supports two modes:
    - Combined (default): single reasoning call with roadmap_risk.txt
    - Split (split_reasoning=True): two sequential calls — roadmap.txt
      then risk.txt with the roadmap injected. Reduces peak context.
    """

    def __init__(self, *, split_reasoning: bool = False) -> None:
        super().__init__()
        self._split_reasoning = split_reasoning

    @property
    def name(self) -> str:
        return "roadmap_risk"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.65, 0.95)

    def _build_prompt_parts(
        self, job_description: str, prior_outputs: dict[str, Any],
    ) -> tuple[str, str, str, str]:
        """Extract shared prompt components.

        Returns: (context_str, architecture_design_str, binding, krag_context)
        """
        context_str = ""
        ctx = {}
        if "context" in prior_outputs:
            ctx = prior_outputs["context"]
            if isinstance(ctx, str):
                try:
                    ctx = json.loads(ctx)
                except (json.JSONDecodeError, TypeError):
                    ctx = {}
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
            adrs = design.get("adrs", [])
            if adrs:
                architecture_design_str += "\nKey Decisions (ADRs):\n"
                for adr in adrs:
                    architecture_design_str += f"- {adr.get('title', '')}: {adr.get('decision', '')} ({adr.get('rationale', '')})\n"
            components = design.get("components", [])
            if components:
                architecture_design_str += "\nComponents:\n"
                for comp in components:
                    line = f"- {comp.get('name', '')}: {comp.get('purpose', '')}"
                    ifaces = ", ".join(comp.get("interfaces", []))
                    if ifaces:
                        line += f" [interfaces: {ifaces}]"
                    architecture_design_str += line + "\n"
            integrations = design.get("integration_points", [])
            if integrations:
                architecture_design_str += f"\nIntegration Points: {', '.join(integrations)}\n"
            artifacts = design.get("artifacts", [])
            if artifacts:
                architecture_design_str += f"\nDesign Artifacts: {', '.join(a.get('filename', '') for a in artifacts)}\n"

        binding_parts = []
        if "context" in prior_outputs:
            needed = ctx.get("needed_artifacts", [])
            if needed:
                binding_parts.append("Deliverable files (from context):\n" + "\n".join(f"  - {a}" for a in needed))
        if "architecture" in prior_outputs:
            rec = prior_outputs["architecture"].get("recommended", "")
            if rec:
                binding_parts.append(f"Chosen approach: {rec}")
        if "design" in prior_outputs:
            comps = [c.get("name", "") for c in prior_outputs["design"].get("components", []) if c.get("name")]
            if comps:
                binding_parts.append(f"Components to implement: {', '.join(comps)}")
        binding = "\n".join(binding_parts) if binding_parts else "No specific binding constraints."

        krag_context = self._get_raw_summaries(prior_outputs)
        return context_str.strip(), architecture_design_str.strip(), binding, krag_context

    def build_prompt(self, job_description: str, prior_outputs: dict[str, Any]) -> list[dict]:
        prompt_template = load_prompt("roadmap_risk")
        context_str, arch_design_str, binding, krag_context = self._build_prompt_parts(
            job_description, prior_outputs,
        )
        prompt = prompt_template.format(
            description=job_description,
            context=context_str,
            architecture_design=arch_design_str,
            krag_context=krag_context,
            binding_constraints=binding,
        )
        return self._make_messages(prompt)

    def _build_split_roadmap_prompt(
        self, job_description: str, prior_outputs: dict[str, Any],
    ) -> list[dict]:
        prompt_template = load_prompt("roadmap")
        context_str, arch_design_str, binding, krag_context = self._build_prompt_parts(
            job_description, prior_outputs,
        )
        prompt = prompt_template.format(
            context=context_str,
            architecture_design=arch_design_str,
            krag_context=krag_context,
            binding_constraints=binding,
        )
        return self._make_messages(prompt)

    def _build_split_risk_prompt(
        self, job_description: str, prior_outputs: dict[str, Any],
        roadmap_reasoning: str,
    ) -> list[dict]:
        prompt_template = load_prompt("risk")
        context_str, arch_design_str, binding, krag_context = self._build_prompt_parts(
            job_description, prior_outputs,
        )
        prompt = prompt_template.format(
            context=context_str,
            architecture_design=arch_design_str,
            krag_context=krag_context,
            binding_constraints=binding,
            roadmap=roadmap_reasoning,
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

    async def _execute_combined(
        self, client: Any, job_description: str, prior_outputs: dict[str, Any],
    ) -> str:
        """Single combined reasoning call (original behavior)."""
        messages = self.build_prompt(job_description, prior_outputs)
        await self._report_substep("reasoning")
        t0 = time.monotonic()
        reasoning = await self._reason_with_tools(client, messages, prior_outputs)
        t1 = time.monotonic()
        logger.info(f"Stage '{self.name}': reasoning took {t1 - t0:.1f}s ({len(reasoning)} chars)")

        krag_context = self._get_gathered_context(prior_outputs)
        return await self._self_critique(
            client, reasoning, job_description, krag_context=krag_context,
        )

    async def _execute_split(
        self, client: Any, job_description: str, prior_outputs: dict[str, Any],
    ) -> str:
        """Two sequential reasoning calls: roadmap then risk."""
        # Roadmap reasoning
        roadmap_messages = self._build_split_roadmap_prompt(job_description, prior_outputs)
        await self._report_substep("reasoning:roadmap")
        t0 = time.monotonic()
        roadmap_reasoning = await self._reason_with_tools(
            client, roadmap_messages, prior_outputs,
        )
        t1 = time.monotonic()
        logger.info(
            f"Stage '{self.name}': roadmap reasoning took "
            f"{t1 - t0:.1f}s ({len(roadmap_reasoning)} chars)"
        )

        # Risk reasoning (with roadmap injected)
        risk_messages = self._build_split_risk_prompt(
            job_description, prior_outputs, roadmap_reasoning,
        )
        await self._report_substep("reasoning:risk")
        t2 = time.monotonic()
        risk_reasoning = await self._reason_with_tools(
            client, risk_messages, prior_outputs,
        )
        t3 = time.monotonic()
        logger.info(
            f"Stage '{self.name}': risk reasoning took "
            f"{t3 - t2:.1f}s ({len(risk_reasoning)} chars)"
        )

        reasoning = (
            "## Roadmap\n\n" + roadmap_reasoning
            + "\n\n## Risk Assessment\n\n" + risk_reasoning
        )

        krag_context = self._get_gathered_context(prior_outputs)
        return await self._self_critique(
            client, reasoning, job_description, krag_context=krag_context,
        )

    async def execute(self, client: Any, job_description: str, prior_outputs: dict[str, Any]) -> StageResult:
        try:
            if self._split_reasoning:
                reasoning = await self._execute_split(
                    client, job_description, prior_outputs,
                )
            else:
                reasoning = await self._execute_combined(
                    client, job_description, prior_outputs,
                )

            # 3. Per-field-group extraction
            # Selective context: phases needs codebase for verification commands,
            # risks needs it to ground risks in actual code structure
            _CONTEXT_GROUPS = {"phases", "risks"}
            extract_context = self._get_gathered_context(prior_outputs)

            merged: dict[str, Any] = {}
            for group in _FIELD_GROUPS:
                extra = extract_context if group["label"] in _CONTEXT_GROUPS else ""
                partial = await self._extract_field_group(
                    client,
                    reasoning,
                    group["fields"],
                    group["schema"],
                    group["label"],
                    extra_context=extra,
                )
                merged.update(partial)

            # 4. Post-extraction validators
            from fitz_graveyard.planning.pipeline.validators import (
                ensure_phase_zero, ensure_concrete_verification, ensure_grounded_risks,
            )
            merged = ensure_phase_zero(merged, prior_outputs)
            merged = ensure_grounded_risks(merged, prior_outputs)
            merged = await ensure_concrete_verification(merged, client, reasoning)

            # 5. Parse through existing parse_output (handles defaults + Pydantic validation)
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
