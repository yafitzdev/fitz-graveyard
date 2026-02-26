# fitz_graveyard/planning/pipeline/orchestrator.py
"""
Multi-stage planning pipeline orchestrator.

Executes stages sequentially, passes outputs forward, handles checkpointing
and crash recovery.
"""

import json
import logging
import subprocess
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fitz_graveyard.planning.pipeline.checkpoint import CheckpointManager
from fitz_graveyard.planning.pipeline.stages.base import SYSTEM_PROMPT, PipelineStage, StageResult

if TYPE_CHECKING:
    from fitz_graveyard.planning.agent import AgentContextGatherer

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Result of executing the full planning pipeline.

    Attributes:
        success: Whether all stages completed successfully
        outputs: Dictionary mapping stage_name -> stage_output
        failed_stage: Name of the stage that failed (if success=False)
        error: Error message (if success=False)
        git_sha: Git commit SHA when pipeline ran
    """

    success: bool
    outputs: dict[str, Any]
    failed_stage: str | None = None
    error: str | None = None
    git_sha: str | None = None
    head_advanced: bool = False


def get_git_sha() -> str | None:
    """
    Get current git commit SHA.

    Returns:
        7-character SHA if in git repo, None otherwise
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Failed to get git SHA: {e}")

    return None


class PlanningPipeline:
    """
    Multi-stage planning pipeline orchestrator.

    Executes stages sequentially, with each stage receiving outputs
    from all prior stages. Supports crash recovery via checkpoints.

    Workflow:
    1. Check for existing checkpoint (resume if found)
    2. Execute each stage in sequence
    3. Save checkpoint after each stage
    4. Return aggregated outputs

    Example:
        stages = [VisionStage(), ArchitecturalStage(), ...]
        pipeline = PlanningPipeline(stages, checkpoint_mgr)
        result = await pipeline.execute(client, job_id, description)
    """

    def __init__(
        self, stages: list[PipelineStage], checkpoint_manager: CheckpointManager
    ) -> None:
        """
        Initialize planning pipeline.

        Args:
            stages: Ordered list of pipeline stages (executed in sequence)
            checkpoint_manager: CheckpointManager for persistence
        """
        self._stages = stages
        self._checkpoint_mgr = checkpoint_manager
        logger.info(f"Created PlanningPipeline with {len(stages)} stages")

    async def execute(
        self,
        client: Any,  # OllamaClient (avoiding circular import)
        job_id: str,
        job_description: str,
        resume: bool = False,
        progress_callback: Callable[[float, str], None] | Callable[[float, str], Any] | None = None,
        agent: "AgentContextGatherer | None" = None,
        pre_gathered_context: str | None = None,
    ) -> PipelineResult:
        """
        Execute the planning pipeline.

        Args:
            client: OllamaClient instance for LLM calls
            job_id: Job identifier (for checkpointing)
            job_description: User's planning request
            resume: If True, resume from checkpoint; if False, start fresh
            progress_callback: Optional callback(progress, phase) for updates
            agent: Optional AgentContextGatherer for codebase exploration
            pre_gathered_context: Pre-gathered codebase context (skips agent gathering)

        Returns:
            PipelineResult with all stage outputs or error
        """
        # Capture git SHA at start for freshness detection
        start_sha = get_git_sha()

        # Load checkpoint (if resuming)
        if resume:
            prior_outputs = await self._checkpoint_mgr.load_checkpoint(job_id)
            logger.info(
                f"Resuming pipeline for job {job_id} from stage {len(prior_outputs)}"
            )
        else:
            prior_outputs = {}
            await self._checkpoint_mgr.clear_checkpoint(job_id)
            logger.info(f"Starting fresh pipeline for job {job_id}")

        # Inject pre-gathered context if provided (skips agent gathering)
        if pre_gathered_context is not None and "_agent_context" not in prior_outputs:
            prior_outputs["_agent_context"] = {
                "synthesized": pre_gathered_context,
                "raw_summaries": pre_gathered_context,
            }
            logger.info(
                f"Using pre-gathered context for job {job_id} ({len(pre_gathered_context)} chars)"
            )

        # Run agent context gathering (once, before all stages, with checkpoint)
        if agent is not None and "_agent_context" not in prior_outputs:
            logger.info(f"Running AgentContextGatherer for job {job_id}")
            gathered = await agent.gather(
                client=client,
                job_description=job_description,
                progress_callback=progress_callback,
            )
            # gathered is {"synthesized": str, "raw_summaries": str}
            await self._checkpoint_mgr.save_stage(
                job_id, "_agent_context", gathered
            )
            prior_outputs["_agent_context"] = gathered
            synth_len = len(gathered.get("synthesized", ""))
            raw_len = len(gathered.get("raw_summaries", ""))
            logger.info(
                f"AgentContextGatherer complete: synthesized={synth_len}, raw={raw_len} chars"
            )
        elif "_agent_context" in prior_outputs:
            logger.info(
                f"Resuming: using checkpointed agent context for job {job_id}"
            )
            # Report agent stage as complete so UI shows it done
            if progress_callback:
                result_or_coro = progress_callback(0.09, "agent_exploring_complete")
                if hasattr(result_or_coro, '__await__'):
                    await result_or_coro

        # Inject gathered context for stages to consume
        # _gathered_context = synthesized (concise, for extraction calls)
        # _raw_summaries = per-file detail (for reasoning passes)
        if "_agent_context" in prior_outputs:
            agent_ctx = prior_outputs["_agent_context"]
            # Backwards compat: old checkpoints have {"text": str}
            if "text" in agent_ctx and "synthesized" not in agent_ctx:
                prior_outputs["_gathered_context"] = agent_ctx.get("text", "")
                prior_outputs["_raw_summaries"] = agent_ctx.get("text", "")
            else:
                prior_outputs["_gathered_context"] = agent_ctx.get("synthesized", "")
                prior_outputs["_raw_summaries"] = agent_ctx.get("raw_summaries", "")

        # Flatten merged stage outputs from checkpoint (same as live execution)
        # Merged stages store e.g. {"architecture_design": {"architecture": {}, "design": {}}}
        # but downstream code expects flattened keys like "architecture", "design"
        _MERGED_STAGES = {"architecture_design", "roadmap_risk"}
        for key in _MERGED_STAGES:
            if key in prior_outputs and isinstance(prior_outputs[key], dict):
                for sub_key in ("architecture", "design", "roadmap", "risk"):
                    if sub_key in prior_outputs[key] and sub_key not in prior_outputs:
                        prior_outputs[sub_key] = prior_outputs[key][sub_key]

        # Determine which stages to execute (exclude internal _ keys)
        completed_stages = {k for k in prior_outputs.keys() if not k.startswith("_")}
        remaining_stages = [s for s in self._stages if s.name not in completed_stages]

        if not remaining_stages:
            logger.info("All stages already completed (checkpoint recovery)")
            return PipelineResult(
                success=True,
                outputs=prior_outputs,
                git_sha=get_git_sha(),
            )

        logger.info(
            f"Executing {len(remaining_stages)} stages: {[s.name for s in remaining_stages]}"
        )

        # Report progress for already-completed stages (so UI shows them as done)
        if resume and progress_callback and completed_stages:
            for stage in self._stages:
                if stage.name in completed_stages:
                    result_or_coro = progress_callback(
                        stage.progress_range[1], f"{stage.name}_complete"
                    )
                    if hasattr(result_or_coro, '__await__'):
                        await result_or_coro

        # Execute stages sequentially
        for stage in remaining_stages:
            logger.info(f"Executing stage: {stage.name}")

            # Notify progress (start of stage)
            if progress_callback:
                progress = stage.progress_range[0]
                result_or_coro = progress_callback(progress, stage.name)
                if hasattr(result_or_coro, '__await__'):
                    await result_or_coro

            # Set sub-step callback so stage can report reasoning/formatting progress
            async def _substep_cb(phase_detail: str, _stage=stage) -> None:
                if progress_callback:
                    result_or_coro = progress_callback(_stage.progress_range[0], phase_detail)
                    if hasattr(result_or_coro, '__await__'):
                        await result_or_coro

            stage.set_substep_callback(_substep_cb)

            # Run stage
            result = await stage.execute(client, job_description, prior_outputs)

            # Handle failure
            if not result.success:
                logger.error(f"[{stage.name}] Stage failed: {result.error}")
                return PipelineResult(
                    success=False,
                    outputs=prior_outputs,
                    failed_stage=stage.name,
                    error=result.error,
                    git_sha=get_git_sha(),
                )

            # Save checkpoint
            await self._checkpoint_mgr.save_stage(job_id, stage.name, result.output)

            # Add to prior outputs for next stage
            prior_outputs[stage.name] = result.output

            # Flatten merged stage outputs for downstream consumption
            if isinstance(result.output, dict):
                for sub_key in ("architecture", "design", "roadmap", "risk"):
                    if sub_key in result.output:
                        prior_outputs[sub_key] = result.output[sub_key]

            # Notify progress (end of stage)
            if progress_callback:
                progress = stage.progress_range[1]
                result_or_coro = progress_callback(progress, f"{stage.name}_complete")
                if hasattr(result_or_coro, '__await__'):
                    await result_or_coro

            logger.info(f"Stage '{stage.name}' completed successfully")

        # Cross-stage coherence check
        if progress_callback:
            result_or_coro = progress_callback(0.955, "coherence_check")
            if hasattr(result_or_coro, '__await__'):
                await result_or_coro

        coherence_fixes = await self._coherence_check(client, prior_outputs)
        if coherence_fixes:
            # Apply fixes to prior_outputs
            for section, fixes in coherence_fixes.items():
                if section in prior_outputs and isinstance(prior_outputs[section], dict):
                    prior_outputs[section].update(fixes)
                    # Also update merged stage outputs
                    for stage in self._stages:
                        if stage.name in prior_outputs and isinstance(prior_outputs[stage.name], dict):
                            if section in prior_outputs[stage.name]:
                                prior_outputs[stage.name][section].update(fixes)
            logger.info(f"Applied coherence fixes to sections: {list(coherence_fixes.keys())}")

        # All stages completed — check if HEAD advanced during execution
        end_sha = get_git_sha()
        head_advanced = (
            start_sha is not None
            and end_sha is not None
            and start_sha != end_sha
        )
        if head_advanced:
            logger.warning(
                f"HEAD advanced during pipeline execution: {start_sha[:8]} → {end_sha[:8]}"
            )

        return PipelineResult(
            success=True,
            outputs=prior_outputs,
            git_sha=start_sha,
            head_advanced=head_advanced,
        )

    async def _coherence_check(
        self,
        client: Any,
        prior_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a cross-stage coherence check after all stages complete.

        Verifies that context→architecture→roadmap are consistent:
        - Architecture actually addresses the requirements from context
        - Roadmap phases implement the chosen architecture, not an alternative
        - Risk mitigations reference real phases
        - Component names are consistent across stages

        Returns dict of section→fixes to apply, or {} if coherent.
        On failure, returns {} (never crashes the pipeline).
        """
        # Build a concise summary of each stage's key outputs
        context = prior_outputs.get("context", {})
        architecture = prior_outputs.get("architecture", {})
        design = prior_outputs.get("design", {})
        roadmap = prior_outputs.get("roadmap", {})
        risk = prior_outputs.get("risk", {})

        if not all([context, architecture, roadmap]):
            return {}

        summary = (
            f"CONTEXT:\n"
            f"  Requirements: {json.dumps(context.get('key_requirements', []))}\n"
            f"  Constraints: {json.dumps(context.get('constraints', []))}\n"
            f"  Needed artifacts: {json.dumps(context.get('needed_artifacts', []))}\n\n"
            f"ARCHITECTURE:\n"
            f"  Recommended: {architecture.get('recommended', '')}\n"
            f"  Reasoning: {architecture.get('reasoning', '')}\n\n"
            f"DESIGN:\n"
            f"  Components: {json.dumps([c.get('name', '') for c in design.get('components', [])])}\n"
            f"  ADRs: {json.dumps([a.get('title', '') for a in design.get('adrs', [])])}\n\n"
            f"ROADMAP:\n"
            f"  Phases: {json.dumps([{'num': p.get('number'), 'name': p.get('name'), 'deliverables': p.get('deliverables', [])} for p in roadmap.get('phases', [])])}\n"
            f"  Critical path: {json.dumps(roadmap.get('critical_path', []))}\n\n"
            f"RISKS:\n"
            f"  Risks: {json.dumps([{'desc': r.get('description', ''), 'phases': r.get('affected_phases', [])} for r in risk.get('risks', [])])}\n"
            f"  Overall: {risk.get('overall_risk_level', 'unknown')}\n"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Review this plan for cross-section coherence issues.\n\n"
                    f"{summary}\n"
                    "Check for:\n"
                    "1. Requirements in CONTEXT that aren't addressed by any roadmap phase\n"
                    "2. Roadmap phases that implement a different approach than ARCHITECTURE recommended\n"
                    "3. Risk affected_phases referencing phase numbers that don't exist\n"
                    "4. Component names in DESIGN that don't appear in any roadmap deliverable\n"
                    "5. Needed artifacts from CONTEXT missing from roadmap deliverables\n\n"
                    "If ALL sections are coherent, respond with exactly: COHERENT\n\n"
                    "If there are issues, respond with a JSON object mapping section names to fixes:\n"
                    '{"roadmap": {"phases": [...]}, "risk": {"risks": [...]}}\n'
                    "Only include sections that need fixes. Keep fixes minimal — just correct the inconsistency."
                ),
            },
        ]

        try:
            t0 = time.monotonic()
            response = await client.generate(messages=messages)
            t1 = time.monotonic()
            logger.info(f"Coherence check took {t1 - t0:.1f}s ({len(response)} chars)")

            if "COHERENT" in response.upper()[:50]:
                logger.info("Cross-stage coherence check: all sections coherent")
                return {}

            # Try to parse fixes
            from fitz_graveyard.planning.pipeline.stages.base import extract_json
            fixes = extract_json(response)
            if isinstance(fixes, dict):
                logger.warning(f"Coherence check found issues in: {list(fixes.keys())}")
                return fixes
            return {}
        except Exception as e:
            logger.warning(f"Coherence check failed (non-fatal): {e}")
            return {}

    def get_progress(self, completed_stages: set[str]) -> float:
        """
        Calculate overall pipeline progress from completed stages.

        Args:
            completed_stages: Set of completed stage names

        Returns:
            Progress from 0.0 to 1.0
        """
        if not self._stages:
            return 0.0

        # Find the last completed stage
        last_completed_idx = -1
        for i, stage in enumerate(self._stages):
            if stage.name in completed_stages:
                last_completed_idx = i

        if last_completed_idx == -1:
            return 0.0

        # Return the end of that stage's progress range
        return self._stages[last_completed_idx].progress_range[1]
