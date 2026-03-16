# fitz_graveyard/planning/pipeline/stages/architecture_design.py
"""
Merged architecture + design stage: Reasoning + per-field-group extraction.

Combines architectural exploration and design decisions. One free-form
reasoning call, then 6 small JSON extractions — each producing a tiny
schema that a small model can handle reliably.
"""

import asyncio
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
                    "name": "Approach A",
                    "description": "What it looks like in production",
                    "pros": ["advantage"],
                    "cons": ["disadvantage"],
                    "complexity": "low|medium|high",
                    "best_for": ["scenario"],
                },
                {
                    "name": "Approach B (minimum 2 approaches required)",
                    "description": "A genuinely different strategy",
                    "pros": ["advantage"],
                    "cons": ["disadvantage"],
                    "complexity": "low|medium|high",
                    "best_for": ["scenario"],
                },
            ],
            "recommended": "must match one approach name exactly",
            "reasoning": "why this approach is right AND why the other is wrong",
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

    Supports two modes:
    - Combined (default): single reasoning call with architecture_design.txt
    - Split (split_reasoning=True): two sequential calls — architecture.txt
      then design.txt with the architecture decision injected. Each call
      uses ~8K tokens instead of ~29K, enabling smaller context windows.
    """

    # Budget for codebase context in the reasoning prompt (~50K tokens).
    # Leaves room for template text, response generation, and tool-use overhead.
    _REASONING_KRAG_BUDGET_CHARS = 200_000

    def __init__(self, *, split_reasoning: bool = False) -> None:
        super().__init__()
        self._split_reasoning = split_reasoning

    @property
    def name(self) -> str:
        return "architecture_design"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.25, 0.65)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any], *, findings: str = "",
    ) -> list[dict]:
        prompt_template = load_prompt("architecture_design")
        context_str, binding, krag_context = self._build_prompt_parts(
            job_description, prior_outputs, findings,
        )
        impl_check = self._get_implementation_check(prior_outputs)
        prompt = prompt_template.format(
            context=context_str.strip(),
            krag_context=krag_context,
            binding_constraints=binding,
        )
        if impl_check:
            prompt = f"{impl_check}\n\n{prompt}"
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
        arch_data.setdefault("technology_considerations", [])

        arch_data.setdefault("key_tradeoffs", {})

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

    # ------------------------------------------------------------------
    # Post-reasoning verification agents
    # ------------------------------------------------------------------

    async def _verify_contracts(
        self, client: Any, reasoning: str, krag_context: str, job_description: str,
    ) -> str:
        """Agent 1: Extract actual interface contracts for proposed integration points."""
        await self._report_substep("verifying:contracts")
        prompt = load_prompt("verify_contracts").format(
            reasoning=reasoning, krag_context=krag_context, job_description=job_description,
        )
        messages = self._make_messages(prompt)
        try:
            t0 = time.monotonic()
            result = await client.generate(messages=messages, temperature=0, max_tokens=4096)
            logger.info(f"Stage '{self.name}': contract verification took {time.monotonic() - t0:.1f}s")
            return result
        except Exception as e:
            logger.warning(f"Stage '{self.name}': contract verification failed: {e}")
            return ""

    async def _verify_data_flow(
        self, client: Any, reasoning: str, krag_context: str, job_description: str,
    ) -> str:
        """Agent 2: Trace actual data flow through proposed modification paths."""
        await self._report_substep("verifying:data_flow")
        prompt = load_prompt("verify_data_flow").format(
            reasoning=reasoning, krag_context=krag_context, job_description=job_description,
        )
        messages = self._make_messages(prompt)
        try:
            t0 = time.monotonic()
            result = await client.generate(messages=messages, temperature=0, max_tokens=4096)
            logger.info(f"Stage '{self.name}': data flow verification took {time.monotonic() - t0:.1f}s")
            return result
        except Exception as e:
            logger.warning(f"Stage '{self.name}': data flow verification failed: {e}")
            return ""

    async def _verify_patterns(
        self, client: Any, reasoning: str, krag_context: str, job_description: str,
    ) -> str:
        """Agent 4: Find existing patterns similar to proposed approaches."""
        await self._report_substep("verifying:patterns")
        prompt = load_prompt("verify_patterns").format(
            reasoning=reasoning, krag_context=krag_context, job_description=job_description,
        )
        messages = self._make_messages(prompt)
        try:
            t0 = time.monotonic()
            result = await client.generate(messages=messages, temperature=0, max_tokens=4096)
            logger.info(f"Stage '{self.name}': pattern verification took {time.monotonic() - t0:.1f}s")
            return result
        except Exception as e:
            logger.warning(f"Stage '{self.name}': pattern verification failed: {e}")
            return ""

    async def _verify_type_boundaries(
        self, client: Any, reasoning: str, krag_context: str, job_description: str,
    ) -> str:
        """Agent 6: Audit type boundaries — trace concrete runtime types, catch DATA LOST."""
        await self._report_substep("verifying:type_boundaries")
        prompt = load_prompt("verify_type_boundaries").format(
            reasoning=reasoning, krag_context=krag_context, job_description=job_description,
        )
        messages = self._make_messages(prompt)
        try:
            t0 = time.monotonic()
            result = await client.generate(messages=messages, temperature=0, max_tokens=4096)
            logger.info(f"Stage '{self.name}': type boundary audit took {time.monotonic() - t0:.1f}s")
            return result
        except Exception as e:
            logger.warning(f"Stage '{self.name}': type boundary audit failed: {e}")
            return ""

    async def _verify_sketch(
        self,
        client: Any,
        reasoning: str,
        contract_sheet: str,
        data_flow_map: str,
        job_description: str,
    ) -> str:
        """Agent 3: Write pseudocode against real interfaces, flag mismatches."""
        await self._report_substep("verifying:sketch")
        prompt = load_prompt("verify_sketch").format(
            reasoning=reasoning,
            contract_sheet=contract_sheet or "(contract extraction unavailable)",
            data_flow_map=data_flow_map or "(data flow tracing unavailable)",
            job_description=job_description,
        )
        messages = self._make_messages(prompt)
        try:
            t0 = time.monotonic()
            result = await client.generate(messages=messages, temperature=0, max_tokens=4096)
            logger.info(f"Stage '{self.name}': sketch verification took {time.monotonic() - t0:.1f}s")
            return result
        except Exception as e:
            logger.warning(f"Stage '{self.name}': sketch verification failed: {e}")
            return ""

    async def _verify_assumptions(
        self,
        client: Any,
        reasoning: str,
        contract_sheet: str,
        data_flow_map: str,
        feasibility_report: str,
        pattern_catalog: str,
        job_description: str,
    ) -> str:
        """Agent 5: Surface and verify every assumption in the proposed architecture."""
        await self._report_substep("verifying:assumptions")
        prompt = load_prompt("verify_assumptions").format(
            reasoning=reasoning,
            contract_sheet=contract_sheet or "(unavailable)",
            data_flow_map=data_flow_map or "(unavailable)",
            feasibility_report=feasibility_report or "(unavailable)",
            pattern_catalog=pattern_catalog or "(unavailable)",
            job_description=job_description,
        )
        messages = self._make_messages(prompt)
        try:
            t0 = time.monotonic()
            result = await client.generate(messages=messages, temperature=0, max_tokens=4096)
            logger.info(f"Stage '{self.name}': assumption verification took {time.monotonic() - t0:.1f}s")
            return result
        except Exception as e:
            logger.warning(f"Stage '{self.name}': assumption verification failed: {e}")
            return ""

    async def _run_verification_agents(
        self,
        client: Any,
        reasoning: str,
        prior_outputs: dict[str, Any],
        job_description: str,
    ) -> str:
        """Run 6 post-reasoning verification agents to catch architectural flaws.

        Agents 1 (contracts), 2 (data flow), 4 (patterns), 6 (type boundaries) run in parallel.
        Agent 3 (sketch test) depends on 1+2. Agent 5 (assumptions) depends on all.

        Returns formatted verification findings, or empty string if all agents fail.
        """
        krag_context = self._get_gathered_context(prior_outputs)
        if not krag_context:
            logger.info(f"Stage '{self.name}': no gathered context, skipping verification agents")
            return ""

        t0 = time.monotonic()

        # Parallel batch: agents 1, 2, 4, 6 (independent inputs)
        parallel_results = await asyncio.gather(
            self._verify_contracts(client, reasoning, krag_context, job_description),
            self._verify_data_flow(client, reasoning, krag_context, job_description),
            self._verify_patterns(client, reasoning, krag_context, job_description),
            self._verify_type_boundaries(client, reasoning, krag_context, job_description),
            return_exceptions=True,
        )

        contract_sheet = parallel_results[0] if not isinstance(parallel_results[0], Exception) else ""
        data_flow_map = parallel_results[1] if not isinstance(parallel_results[1], Exception) else ""
        pattern_catalog = parallel_results[2] if not isinstance(parallel_results[2], Exception) else ""
        type_boundary_audit = parallel_results[3] if not isinstance(parallel_results[3], Exception) else ""

        for i, r in enumerate(parallel_results):
            if isinstance(r, Exception):
                labels = ["contracts", "data_flow", "patterns", "type_boundaries"]
                logger.warning(f"Stage '{self.name}': verification agent {labels[i]} raised: {r}")

        # Sequential: agent 3 (needs agents 1+2)
        feasibility_report = await self._verify_sketch(
            client, reasoning, contract_sheet, data_flow_map, job_description,
        )

        # Sequential: agent 5 (needs all)
        assumption_register = await self._verify_assumptions(
            client, reasoning, contract_sheet, data_flow_map,
            feasibility_report, pattern_catalog, job_description,
        )

        # Assemble findings
        sections = []
        if contract_sheet:
            sections.append(f"--- INTERFACE CONTRACTS (verified against source) ---\n{contract_sheet}")
        if data_flow_map:
            sections.append(f"--- DATA FLOW MAP (traced through source) ---\n{data_flow_map}")
        if type_boundary_audit:
            sections.append(f"--- TYPE BOUNDARY AUDIT ---\n{type_boundary_audit}")
        if pattern_catalog:
            sections.append(f"--- EXISTING PATTERNS ---\n{pattern_catalog}")
        if feasibility_report:
            sections.append(f"--- FEASIBILITY REPORT ---\n{feasibility_report}")
        if assumption_register:
            sections.append(f"--- ASSUMPTION REGISTER ---\n{assumption_register}")

        elapsed = time.monotonic() - t0
        if sections:
            logger.info(
                f"Stage '{self.name}': verification agents completed in {elapsed:.1f}s "
                f"({len(sections)}/6 agents produced output)"
            )
            return "\n\n".join(sections)

        logger.warning(f"Stage '{self.name}': all 6 verification agents failed ({elapsed:.1f}s)")
        return ""

    def _build_split_architecture_prompt(
        self, job_description: str, prior_outputs: dict[str, Any], findings: str = "",
    ) -> list[dict]:
        """Build prompt for architecture-only reasoning (split mode)."""
        prompt_template = load_prompt("architecture")
        context_str, binding, krag_context = self._build_prompt_parts(
            job_description, prior_outputs, findings,
        )
        impl_check = self._get_implementation_check(prior_outputs)
        prompt = prompt_template.format(
            context=context_str.strip(),
            krag_context=krag_context,
            binding_constraints=binding,
        )
        if impl_check:
            prompt = f"{impl_check}\n\n{prompt}"
        return self._make_messages(prompt)

    def _build_split_design_prompt(
        self,
        job_description: str,
        prior_outputs: dict[str, Any],
        architecture_reasoning: str,
        findings: str = "",
    ) -> list[dict]:
        """Build prompt for design-only reasoning (split mode)."""
        prompt_template = load_prompt("design")
        context_str, binding, krag_context = self._build_prompt_parts(
            job_description, prior_outputs, findings,
        )
        prompt = prompt_template.format(
            context=context_str.strip(),
            krag_context=krag_context,
            binding_constraints=binding,
            architecture=architecture_reasoning,
        )
        return self._make_messages(prompt)

    def _build_prompt_parts(
        self, job_description: str, prior_outputs: dict[str, Any], findings: str = "",
    ) -> tuple[str, str, str]:
        """Extract shared prompt components (context, binding, krag_context)."""
        context_str = job_description
        if "context" in prior_outputs:
            context = prior_outputs["context"]
            if isinstance(context, str):
                try:
                    context = json.loads(context)
                except (json.JSONDecodeError, TypeError):
                    context = {}
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

        binding = ""
        if "context" in prior_outputs:
            ctx = context
            files = ctx.get("existing_files", [])
            artifacts = ctx.get("needed_artifacts", [])
            scope = ctx.get("scope_boundaries", {})
            bp = []
            if files:
                bp.append("Existing files to integrate with:\n" + "\n".join(f"  - {f}" for f in files))
            if artifacts:
                bp.append("Expected deliverables:\n" + "\n".join(f"  - {a}" for a in artifacts))
            if scope:
                in_scope = scope.get("in_scope", [])
                out_of_scope = scope.get("out_of_scope", [])
                if in_scope:
                    bp.append(f"In scope: {', '.join(in_scope)}")
                if out_of_scope:
                    bp.append(f"Out of scope: {', '.join(out_of_scope)}")
            binding = "\n".join(bp) if bp else "No specific constraints from context stage."

        # Inject artifact duplicate warnings if any proposed files match existing code
        dupes = prior_outputs.get("_artifact_duplicates", [])
        if dupes:
            dupe_lines = [
                "\nWARNING — EXISTING CODE DETECTED for proposed new files:"
            ]
            for d in dupes:
                dupe_lines.append(f"\n  Proposed: {d['proposed']} (keywords: {', '.join(d['keywords'])})")
                dupe_lines.append("  Already exists in codebase:")
                for match in d["existing_matches"]:
                    dupe_lines.append(f"    - {match}")
                dupe_lines.append("  → CHECK these files before creating new ones. Extend existing code if possible.")
            binding += "\n".join(dupe_lines)

        raw_summaries = self._get_raw_summaries(prior_outputs)
        gathered_context = self._get_gathered_context(prior_outputs)
        full_krag = (findings + "\n\n" + raw_summaries) if findings else raw_summaries
        if len(full_krag) <= self._REASONING_KRAG_BUDGET_CHARS:
            krag_context = full_krag
        else:
            krag_context = (findings + "\n\n" + gathered_context) if findings else gathered_context

        return context_str, binding, krag_context

    async def _execute_combined(
        self, client: Any, job_description: str,
        prior_outputs: dict[str, Any], findings: str,
    ) -> str:
        """Single combined reasoning call (original behavior)."""
        messages = self.build_prompt(job_description, prior_outputs, findings=findings)
        await self._report_substep("reasoning")
        t0 = time.monotonic()
        reasoning = await self._reason_with_tools(client, messages, prior_outputs)
        t1 = time.monotonic()
        logger.info(f"Stage '{self.name}': reasoning took {t1 - t0:.1f}s ({len(reasoning)} chars)")

        verification = await self._run_verification_agents(
            client, reasoning, prior_outputs, job_description,
        )
        if verification:
            reasoning = reasoning + "\n\n--- POST-REASONING VERIFICATION ---\n\n" + verification

        krag_context = self._get_gathered_context(prior_outputs)
        reasoning = await self._self_critique(
            client, reasoning, job_description, krag_context=krag_context,
        )
        return reasoning

    async def _execute_split(
        self, client: Any, job_description: str,
        prior_outputs: dict[str, Any], findings: str,
    ) -> str:
        """Two sequential reasoning calls: architecture then design.

        Each call has its own tool-use session. The architecture decision
        is injected into the design prompt. Reduces peak context from
        ~29K to ~8K tokens per call.
        """
        # Architecture reasoning
        arch_messages = self._build_split_architecture_prompt(
            job_description, prior_outputs, findings,
        )
        await self._report_substep("reasoning:architecture")
        t0 = time.monotonic()
        arch_reasoning = await self._reason_with_tools(client, arch_messages, prior_outputs)
        t1 = time.monotonic()
        logger.info(
            f"Stage '{self.name}': architecture reasoning took "
            f"{t1 - t0:.1f}s ({len(arch_reasoning)} chars)"
        )

        # Design reasoning (with architecture decision injected)
        design_messages = self._build_split_design_prompt(
            job_description, prior_outputs, arch_reasoning, findings,
        )
        await self._report_substep("reasoning:design")
        t2 = time.monotonic()
        design_reasoning = await self._reason_with_tools(client, design_messages, prior_outputs)
        t3 = time.monotonic()
        logger.info(
            f"Stage '{self.name}': design reasoning took "
            f"{t3 - t2:.1f}s ({len(design_reasoning)} chars)"
        )

        # Combine for downstream extraction
        reasoning = (
            "## Architecture Analysis\n\n" + arch_reasoning
            + "\n\n## Design Decisions\n\n" + design_reasoning
        )

        # Verification + critique on combined reasoning
        verification = await self._run_verification_agents(
            client, reasoning, prior_outputs, job_description,
        )
        if verification:
            reasoning = reasoning + "\n\n--- POST-REASONING VERIFICATION ---\n\n" + verification

        krag_context = self._get_gathered_context(prior_outputs)
        reasoning = await self._self_critique(
            client, reasoning, job_description, krag_context=krag_context,
        )
        return reasoning

    async def execute(self, client: Any, job_description: str, prior_outputs: dict[str, Any]) -> StageResult:
        try:
            # 1. Focused investigations (parallel) — pre-digest the source code
            findings = await self._investigate(client, job_description, prior_outputs)

            if self._split_reasoning:
                reasoning = await self._execute_split(
                    client, job_description, prior_outputs, findings,
                )
            else:
                reasoning = await self._execute_combined(
                    client, job_description, prior_outputs, findings,
                )

            # 5. Per-field-group extraction — compact structural overview for grounding
            _CONTEXT_GROUPS = {"approaches", "adrs", "artifacts", "components", "integrations"}
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

            # 7. Post-extraction validators
            from fitz_graveyard.planning.pipeline.validators import (
                ensure_correct_artifacts,
                ensure_min_adrs,
                ensure_valid_artifacts,
            )
            merged = await ensure_min_adrs(merged, client, prior_outputs, reasoning)
            merged = ensure_valid_artifacts(merged, prior_outputs)
            merged = await ensure_correct_artifacts(merged, client, prior_outputs)

            # 8. Parse through existing parse_output (handles defaults + Pydantic validation)
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
