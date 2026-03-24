# fitz_graveyard/background/worker.py
"""
Background worker for sequential job processing.

Polls the job queue, processes jobs one at a time, and handles graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING as _TC

from fitz_graveyard.config.schema import FitzPlannerConfig
from fitz_graveyard.llm.llama_cpp import LlamaCppClient
from fitz_graveyard.llm.lm_studio import LMStudioClient
from fitz_graveyard.llm.memory import MemoryMonitor

if _TC:
    from fitz_graveyard.llm.client import OllamaClient
from fitz_graveyard.models.jobs import JobState
from fitz_graveyard.models.store import JobStore
from fitz_graveyard.planning.pipeline.checkpoint import CheckpointManager
from fitz_graveyard.planning.pipeline.orchestrator import DecomposedPipeline, PlanningPipeline
from fitz_graveyard.planning.pipeline.output import PlanRenderer
from fitz_graveyard.planning.agent import AgentContextGatherer
from fitz_graveyard.planning.pipeline.stages import DEFAULT_STAGES, create_stages
from fitz_graveyard.planning.schemas.plan_output import PlanOutput
from fitz_graveyard.api_review.schemas import ReviewRequest, CostBreakdown
from fitz_graveyard.api_review.client import AnthropicReviewClient

logger = logging.getLogger(__name__)


class BackgroundWorker:
    """
    Sequential background job processor.

    Features:
        - Polls queue for next QUEUED job (FIFO)
        - Processes jobs one at a time
        - Marks running job as INTERRUPTED on shutdown (via CancelledError)
        - Handles exceptions and marks jobs as FAILED
    """

    def __init__(
        self,
        store: JobStore,
        config: FitzPlannerConfig | None = None,
        poll_interval: float = 1.0,
        ollama_client: OllamaClient | LMStudioClient | LlamaCppClient | None = None,
        memory_threshold: float = 80.0,
    ) -> None:
        """
        Initialize background worker.

        Args:
            store: Job store implementation (SQLite or in-memory)
            config: Optional FitzPlannerConfig for pipeline setup
            poll_interval: Seconds between queue checks (default: 1.0)
            ollama_client: Optional OllamaClient for real plan generation
            memory_threshold: RAM usage % threshold for memory monitoring (default: 80.0)
        """
        self._store = store
        self._config = config or FitzPlannerConfig()
        self._poll_interval = poll_interval
        self._ollama_client = ollama_client
        self._memory_threshold = memory_threshold
        self._current_job_id: str | None = None
        self._pre_gathered_context: str | None = None
        self._resume_from_checkpoint: bool = False
        self._task: asyncio.Task | None = None

        # Initialize pipeline components if config provided
        self._pipeline: PlanningPipeline | None = None
        self._checkpoint_mgr: CheckpointManager | None = None
        self._renderer: PlanRenderer | None = None

        if ollama_client:
            # Create checkpoint manager (needs db_path from store)
            if hasattr(store, '_db_path'):
                self._checkpoint_mgr = CheckpointManager(store._db_path)
            else:
                # Fallback for in-memory stores (no checkpointing)
                self._checkpoint_mgr = None
                self._pipeline = None
                logger.warning("Store does not support checkpointing (no db_path)")
                return

            # Create decomposed pipeline (replaces classic 3-stage pipeline)
            self._pipeline = DecomposedPipeline(self._checkpoint_mgr)

            # Create plan renderer
            self._renderer = PlanRenderer()

        logger.info(
            f"Initialized BackgroundWorker (poll_interval={poll_interval}s, "
            f"ollama={'configured' if ollama_client else 'None'}, "
            f"pipeline={'configured' if self._pipeline else 'None'})"
        )

    @property
    def current_job_id(self) -> str | None:
        """Get the currently processing job ID (None if idle)."""
        return self._current_job_id

    async def start(self) -> None:
        """
        Start the background worker loop.

        Creates an asyncio task that polls for queued jobs.
        """
        if self._task is not None:
            logger.warning("Worker already started")
            return

        self._task = asyncio.create_task(self._run_loop())
        logger.info("Background worker started")

    async def stop(self) -> None:
        """
        Stop the background worker gracefully.

        Cancels the worker task and waits for it to finish.
        If a job is running, it will be marked as INTERRUPTED.
        """
        if self._task is None:
            logger.warning("Worker not running")
            return

        logger.info("Stopping background worker...")
        self._task.cancel()

        try:
            await self._task
        except asyncio.CancelledError:
            logger.info("Worker task cancelled")

        self._task = None
        logger.info("Background worker stopped")

    async def _run_loop(self) -> None:
        """
        Main worker loop: poll queue, process jobs sequentially.

        Handles CancelledError for graceful shutdown.
        """
        logger.info("Worker loop started")

        try:
            while True:
                # Fetch next queued job
                job = await self._store.get_next_queued()

                if job is None:
                    # No jobs available - sleep and retry
                    await asyncio.sleep(self._poll_interval)
                    continue

                # Process the job
                self._current_job_id = job.job_id
                logger.info(f"Picked up job {job.job_id} for processing")

                try:
                    # Mark as running
                    await self._store.update(
                        job.job_id,
                        state=JobState.RUNNING,
                        progress=0.0,
                        current_phase="initializing",
                    )

                    # Process job (stub for Phase 4)
                    await self._process_job(job)

                    # Mark as complete
                    await self._store.update(
                        job.job_id,
                        state=JobState.COMPLETE,
                        progress=1.0,
                    )
                    logger.info(f"Job {job.job_id} completed successfully")

                except asyncio.CancelledError:
                    # Shutdown during processing - mark as interrupted
                    logger.warning(f"Job {job.job_id} interrupted by shutdown")
                    try:
                        await self._store.update(
                            job.job_id,
                            state=JobState.INTERRUPTED,
                            error="Server shutdown during processing",
                        )
                    except Exception as e:
                        logger.error(f"Failed to mark job as interrupted: {e}")
                    raise  # Re-raise to exit loop

                except Exception as e:
                    # Job failed - mark with error
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
                    tb_snippet = "".join(tb_lines[-3:])  # Last 3 lines of traceback

                    await self._store.update(
                        job.job_id,
                        state=JobState.FAILED,
                        error=f"{error_msg}\n\n{tb_snippet}",
                    )
                    logger.error(f"Job {job.job_id} failed: {error_msg}")

                finally:
                    self._current_job_id = None

        except asyncio.CancelledError:
            # Graceful shutdown - job already marked as interrupted in inner handler
            logger.info("Worker loop cancelled")
            raise

    async def _process_job(self, job) -> None:
        """
        Process a planning job using PlanningPipeline.

        Runs health check, executes multi-stage pipeline, scores outputs,
        handles API review if requested, renders markdown, and writes to file.

        Two-pass flow for API review:
        - Pass 1: Run pipeline, score, flag sections, estimate cost, pause at AWAITING_REVIEW
        - Pass 2: Resume from stored pipeline state, execute API review, incorporate feedback, complete

        Args:
            job: JobRecord to process

        Raises:
            ConnectionError: If Ollama health check fails
            MemoryError: If RAM threshold exceeded during generation
        """
        if self._ollama_client is None or self._pipeline is None:
            logger.warning(
                f"No pipeline configured, completing job {job.job_id} with stub"
            )
            await self._store.update(job.job_id, progress=0.5, current_phase="pending_engine")
            await asyncio.sleep(0.1)
            return

        # Detect if this is the second pass (API review confirmed)
        is_review_confirmed = job.current_phase == "api_review_confirmed"

        # Step 1: Health check (skip on second pass)
        if not is_review_confirmed:
            if not self._resume_from_checkpoint:
                await self._store.update(job.job_id, progress=0.05, current_phase="health_check")
            else:
                await self._store.update(job.job_id, current_phase="health_check")

            # For LM Studio: split check into connectivity + model loading
            # so the GUI shows "Loading model..." during the slow part
            if isinstance(self._ollama_client, LMStudioClient):
                if not await self._ollama_client.is_model_loaded():
                    await self._store.update(
                        job.job_id, current_phase="loading_model",
                    )
                    logger.info(
                        f"No model loaded — auto-loading {self._ollama_client.model}"
                    )

            healthy = await self._ollama_client.health_check()
            if not healthy:
                raise ConnectionError(
                    "LLM health check failed: server not available or model not found"
                )

        # Step 2: Execute pipeline with progress callback (skip on second pass)
        if not is_review_confirmed:
            async def progress_callback(progress: float, phase: str) -> None:
                await self._store.update(job.job_id, progress=progress, current_phase=phase)

            # Build agent if source_dir is available
            source_dir = (
                job.source_dir
                or self._config.agent.source_dir
                or str(Path.cwd())
            )
            agent = None
            if self._config.agent.enabled and source_dir:
                if Path(source_dir).is_dir():
                    agent = AgentContextGatherer(
                        config=self._config.agent,
                        source_dir=source_dir,
                    )
                else:
                    logger.warning(
                        f"source_dir '{source_dir}' is not a directory, "
                        "skipping agent context gathering"
                    )

            result = await self._pipeline.execute(
                client=self._ollama_client,
                job_id=job.job_id,
                job_description=job.description,
                resume=self._resume_from_checkpoint,
                progress_callback=progress_callback,
                agent=agent,
                pre_gathered_context=self._pre_gathered_context,
            )

            if not result.success:
                raise RuntimeError(
                    f"Pipeline failed at stage '{result.failed_stage}': {result.error}"
                )
        else:
            # Second pass: load pipeline state from job.pipeline_state
            import json
            from fitz_graveyard.planning.pipeline.orchestrator import PipelineResult

            if not job.pipeline_state:
                raise RuntimeError("API review confirmed but no pipeline state stored")

            # Reconstruct PipelineResult from stored JSON
            state_data = json.loads(job.pipeline_state)
            result = PipelineResult(**state_data)
            logger.info(f"Loaded pipeline state for API review continuation (job {job.job_id})")

        # Step 3: API review (if requested and configured)
        if not is_review_confirmed:
            api_review_cost_dict = None
            api_review_feedback = None

            if job.api_review:
                if self._config.anthropic.api_key is None:
                    logger.warning(
                        f"API review requested for job {job.job_id} but no Anthropic API key configured. "
                        "Skipping review and completing normally."
                    )
                else:
                    # Flag all non-internal sections for review
                    flagged_sections = [
                        ReviewRequest(
                            section_name=stage_name,
                            section_type=stage_name,
                            content=str(stage_output),
                            confidence_score=0.0,
                            flag_reason="API review requested",
                        )
                        for stage_name, stage_output in result.outputs.items()
                        if not stage_name.startswith("_")
                    ]

                    review_client = AnthropicReviewClient(
                        api_key=self._config.anthropic.api_key,
                        model=self._config.anthropic.model
                    )
                    cost_calculator = review_client.get_cost_calculator()
                    cost_estimate = await cost_calculator.estimate_review_cost(
                        flagged_sections,
                        model=self._config.anthropic.model
                    )

                    import json
                    await self._store.update(
                        job.job_id,
                        cost_estimate_json=cost_estimate.model_dump_json(),
                        pipeline_state=json.dumps(result.model_dump()),
                        state=JobState.AWAITING_REVIEW,
                        progress=0.96,
                        current_phase="awaiting_review_confirmation",
                    )

                    logger.info(
                        f"Job {job.job_id} paused at AWAITING_REVIEW. "
                        f"{len(flagged_sections)} sections flagged. "
                        f"Estimated cost: ${cost_estimate.cost_usd:.4f} USD"
                    )
                    return  # EARLY RETURN - wait for user confirmation

        else:
            # Second pass: execute API review
            import json
            from fitz_graveyard.api_review.schemas import CostEstimate

            await self._store.update(job.job_id, progress=0.96, current_phase="executing_review")

            cost_estimate = CostEstimate.model_validate_json(job.cost_estimate_json)

            flagged_sections = [
                ReviewRequest(
                    section_name=stage_name,
                    section_type=stage_name,
                    content=str(stage_output),
                    confidence_score=0.0,
                    flag_reason="API review requested",
                )
                for stage_name, stage_output in result.outputs.items()
                if not stage_name.startswith("_")
            ]

            review_client = AnthropicReviewClient(
                api_key=self._config.anthropic.api_key,
                model=self._config.anthropic.model
            )
            review_results = await review_client.review_sections(flagged_sections)

            cost_calculator = review_client.get_cost_calculator()
            actual_cost = cost_calculator.calculate_actual_cost(
                review_results,
                model=self._config.anthropic.model
            )
            actual_cost.estimate = cost_estimate
            api_review_cost_dict = actual_cost.model_dump()

            api_review_feedback = {
                r.section_name: r.feedback
                for r in review_results
                if r.success and r.feedback
            }

            await self._store.update(
                job.job_id,
                review_result_json=json.dumps([r.model_dump() for r in review_results])
            )

            logger.info(
                f"Job {job.job_id}: API review complete. "
                f"{actual_cost.sections_reviewed}/{len(flagged_sections)} sections reviewed successfully. "
                f"Actual cost: ${actual_cost.actual_cost_usd:.4f} USD"
            )

        # Step 4: Build diagnostics and create PlanOutput
        call_metrics = []
        if hasattr(self._ollama_client, "drain_call_metrics"):
            metrics = self._ollama_client.drain_call_metrics()
            # Handle both sync and async (AsyncMock returns coroutine)
            if hasattr(metrics, "__await__"):
                metrics = await metrics
            call_metrics = metrics if isinstance(metrics, list) else []
        diagnostics: dict[str, Any] = {
            "provider": self._config.provider,
            "model": self._ollama_client.model,
            "context_length": getattr(self._ollama_client, "context_size", None),
            "agent_enabled": self._config.agent.enabled,
            "stage_timings_s": result.stage_timings,
            "total_llm_calls": len(call_metrics),
            "total_generation_s": round(sum(m["elapsed_s"] for m in call_metrics), 1),
        }

        # Provider-specific model details
        if self._config.provider == "llama_cpp":
            model_cfg = self._config.llama_cpp.fast_model
            diagnostics["model_file"] = model_cfg.path
            if model_cfg.cache_type_k:
                diagnostics["cache_type_k"] = model_cfg.cache_type_k
            if model_cfg.cache_type_v:
                diagnostics["cache_type_v"] = model_cfg.cache_type_v

        # Extract quant from model path/name (e.g. "Q6_K_XL" from GGUF filename)
        model_id = diagnostics.get("model_file") or diagnostics.get("model", "")
        quant_match = re.search(r"[_-](Q\d+_K(?:_[A-Z]+)?|Q\d+_\d+|IQ\d+_[A-Z]+|F16|F32|BF16)", model_id, re.IGNORECASE)
        if quant_match:
            diagnostics["quant"] = quant_match.group(1).upper()
        agent_ctx = result.outputs.get("_agent_context", {})
        if isinstance(agent_ctx, dict) and "agent_files" in agent_ctx:
            diagnostics["agent_files"] = agent_ctx["agent_files"]

        # Pipeline outputs are dicts from model_dump(), need to reconstruct Pydantic models
        await self._store.update(job.job_id, progress=0.97, current_phase="rendering")

        from fitz_graveyard.planning.schemas.context import ContextOutput
        from fitz_graveyard.planning.schemas.architecture import ArchitectureOutput
        from fitz_graveyard.planning.schemas.design import DesignOutput
        from fitz_graveyard.planning.schemas.roadmap import RoadmapOutput
        from fitz_graveyard.planning.schemas.risk import RiskOutput

        plan_output = PlanOutput(
            context=ContextOutput(**result.outputs["context"]),
            architecture=ArchitectureOutput(**result.outputs["architecture"]),
            design=DesignOutput(**result.outputs["design"]),
            roadmap=RoadmapOutput(**result.outputs["roadmap"]),
            risk=RiskOutput(**result.outputs["risk"]),
            job_description=job.description,
            git_sha=result.git_sha or "",
            generated_at=datetime.now(),
            api_review_requested=job.api_review,
            api_review_cost=api_review_cost_dict,
            api_review_feedback=api_review_feedback,
            diagnostics=diagnostics,
        )

        # Step 5: Render to markdown
        if self._renderer:
            markdown = self._renderer.render(plan_output, head_advanced=result.head_advanced)
        else:
            markdown = "# Plan output\n\n(Renderer not configured)"

        # Step 6: Write to file
        await self._store.update(job.job_id, progress=0.98, current_phase="writing_file")
        plans_dir = Path(self._config.output.plans_dir)
        if not plans_dir.is_absolute():
            base = Path(job.source_dir).resolve() if job.source_dir else Path.cwd()
            output_dir = base / plans_dir
        else:
            output_dir = plans_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_name = f"plan_{job.job_id}_{timestamp}.md"
        file_path = output_dir / file_name

        file_path.write_text(markdown, encoding="utf-8")

        # Step 7: Update job with file path
        await self._store.update(
            job.job_id,
            progress=0.99,
            current_phase="finalizing",
            file_path=str(file_path),
        )

        # Step 8: Eject model from LM Studio to free VRAM
        unload = getattr(self._ollama_client, "unload_model", None)
        if unload is not None:
            logger.info("Pipeline complete — ejecting model to free VRAM")
            await unload()

        logger.info(f"Job {job.job_id} completed: plan written to {file_path}")

    async def process_job_direct(
        self, job_id: str, pre_gathered_context: str | None = None,
        resume: bool = False,
    ) -> None:
        """
        Process a single job directly (for inline CLI execution).

        Unlike the polling loop, this method processes one specific job and returns.
        Handles state transitions (RUNNING → COMPLETE/FAILED) directly.

        Args:
            job_id: Job to process
            pre_gathered_context: Optional pre-gathered codebase context to skip agent re-run
            resume: If True, resume from checkpoint (skip completed stages)

        Raises:
            ValueError: If job not found
            Exception: Re-raises any processing error after marking job FAILED
        """
        job = await self._store.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        self._current_job_id = job_id
        self._pre_gathered_context = pre_gathered_context
        self._resume_from_checkpoint = resume
        try:
            if resume:
                await self._store.update(
                    job_id, state=JobState.RUNNING, current_phase="resuming"
                )
            else:
                await self._store.update(
                    job_id, state=JobState.RUNNING, progress=0.0, current_phase="starting"
                )
            await self._process_job(job)
            # Check if job ended at AWAITING_REVIEW (pipeline paused for API review)
            final_job = await self._store.get(job_id)
            if final_job and final_job.state != JobState.AWAITING_REVIEW:
                await self._store.update(job_id, state=JobState.COMPLETE, progress=1.0)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            try:
                await self._store.update(job_id, state=JobState.FAILED, error=error_msg)
            except Exception:
                pass
            raise
        finally:
            self._current_job_id = None
            self._pre_gathered_context = None
            self._resume_from_checkpoint = False

    def _build_messages(self, job) -> list[dict]:
        """
        Build chat messages from job metadata.

        DEPRECATED: This method exists for backward compatibility with tests.
        Production code now uses PlanningPipeline instead.

        Args:
            job: JobRecord with description, timeline, context, integration_points

        Returns:
            List of chat messages in format [{"role": "user", "content": "..."}]
        """
        prompt_parts = [f"Create an architectural plan for: {job.description}"]

        if job.timeline:
            prompt_parts.append(f"\nTimeline: {job.timeline}")

        if job.context:
            prompt_parts.append(f"\nContext: {job.context}")

        if job.integration_points:
            prompt_parts.append(f"\nIntegration points: {', '.join(job.integration_points)}")

        prompt = "".join(prompt_parts)

        return [{"role": "user", "content": prompt}]

