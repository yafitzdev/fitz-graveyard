# fitz_graveyard/planning/pipeline/orchestrator.py
"""
Multi-stage planning pipeline orchestrator.

Executes stages sequentially, passes outputs forward, handles checkpointing
and crash recovery.
"""

import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fitz_graveyard.planning.pipeline.checkpoint import CheckpointManager
from fitz_graveyard.planning.pipeline.stages.base import PipelineStage, StageResult

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
            prior_outputs["_agent_context"] = {"text": pre_gathered_context}
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
            await self._checkpoint_mgr.save_stage(
                job_id, "_agent_context", {"text": gathered}
            )
            prior_outputs["_agent_context"] = {"text": gathered}
            logger.info(
                f"AgentContextGatherer complete: {len(gathered)} chars of context"
            )
        elif "_agent_context" in prior_outputs:
            logger.info(
                f"Resuming: using checkpointed agent context for job {job_id}"
            )

        # Inject gathered context for stages to consume
        if "_agent_context" in prior_outputs:
            prior_outputs["_gathered_context"] = prior_outputs["_agent_context"].get(
                "text", ""
            )

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

        # Execute stages sequentially
        for stage in remaining_stages:
            logger.info(f"Executing stage: {stage.name}")

            # Notify progress (start of stage)
            if progress_callback:
                progress = stage.progress_range[0]
                result_or_coro = progress_callback(progress, stage.name)
                if hasattr(result_or_coro, '__await__'):
                    await result_or_coro

            # Run stage
            result = await stage.execute(client, job_description, prior_outputs)

            # Handle failure
            if not result.success:
                logger.error(f"Stage '{stage.name}' failed: {result.error}")
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

            # Notify progress (end of stage)
            if progress_callback:
                progress = stage.progress_range[1]
                result_or_coro = progress_callback(progress, f"{stage.name}_complete")
                if hasattr(result_or_coro, '__await__'):
                    await result_or_coro

            logger.info(f"Stage '{stage.name}' completed successfully")

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
