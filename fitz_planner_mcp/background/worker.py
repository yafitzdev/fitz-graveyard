# fitz_planner_mcp/background/worker.py
"""
Background worker for sequential job processing.

Polls the job queue, processes jobs one at a time, and handles graceful shutdown.
"""

import asyncio
import logging
import traceback
from datetime import datetime
from pathlib import Path

from fitz_planner_mcp.config.schema import FitzPlannerConfig
from fitz_planner_mcp.llm.client import OllamaClient
from fitz_planner_mcp.llm.memory import MemoryMonitor
from fitz_planner_mcp.models.jobs import JobState
from fitz_planner_mcp.models.store import JobStore
from fitz_planner_mcp.planning.confidence.flagging import SectionFlagger
from fitz_planner_mcp.planning.confidence.scorer import ConfidenceScorer
from fitz_planner_mcp.planning.pipeline.checkpoint import CheckpointManager
from fitz_planner_mcp.planning.pipeline.orchestrator import PlanningPipeline
from fitz_planner_mcp.planning.pipeline.output import PlanRenderer
from fitz_planner_mcp.planning.pipeline.stages import DEFAULT_STAGES
from fitz_planner_mcp.planning.schemas.plan_output import PlanOutput

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
        ollama_client: OllamaClient | None = None,
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
        self._task: asyncio.Task | None = None

        # Initialize pipeline components if config provided
        self._pipeline: PlanningPipeline | None = None
        self._checkpoint_mgr: CheckpointManager | None = None
        self._scorer: ConfidenceScorer | None = None
        self._flagger: SectionFlagger | None = None
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

            # Create pipeline with stages
            self._pipeline = PlanningPipeline(DEFAULT_STAGES, self._checkpoint_mgr)

            # Create confidence scorer and flagger
            self._scorer = ConfidenceScorer(ollama_client)
            self._flagger = SectionFlagger.from_config(self._config.confidence)

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
        renders markdown, and writes to file.

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

        # Step 1: Health check
        await self._store.update(job.job_id, progress=0.05, current_phase="health_check")
        healthy = await self._ollama_client.health_check()
        if not healthy:
            raise ConnectionError(
                "Ollama health check failed: server not available or model not found"
            )

        # Step 2: Execute pipeline with progress callback
        async def progress_callback(progress: float, phase: str) -> None:
            await self._store.update(job.job_id, progress=progress, current_phase=phase)

        result = await self._pipeline.execute(
            client=self._ollama_client,
            job_id=job.job_id,
            job_description=job.description,
            resume=False,
            progress_callback=progress_callback,
        )

        if not result.success:
            raise RuntimeError(
                f"Pipeline failed at stage '{result.failed_stage}': {result.error}"
            )

        # Step 3: Score sections with confidence scorer
        await self._store.update(job.job_id, progress=0.96, current_phase="scoring")
        section_scores = {}
        if self._scorer:
            for stage_name, stage_output in result.outputs.items():
                # Score based on stage output (simplified: score description if available)
                content = str(stage_output)
                score = await self._scorer.score_section(stage_name, content)
                section_scores[stage_name] = score

        # Compute overall quality score
        overall_score = 0.0
        if self._flagger and section_scores:
            overall_score = self._flagger.compute_overall_score(section_scores)

        # Step 4: Create PlanOutput with all stage outputs
        # Pipeline outputs are dicts from model_dump(), need to reconstruct Pydantic models
        await self._store.update(job.job_id, progress=0.97, current_phase="rendering")

        from fitz_planner_mcp.planning.schemas.context import ContextOutput
        from fitz_planner_mcp.planning.schemas.architecture import ArchitectureOutput
        from fitz_planner_mcp.planning.schemas.design import DesignOutput
        from fitz_planner_mcp.planning.schemas.roadmap import RoadmapOutput
        from fitz_planner_mcp.planning.schemas.risk import RiskOutput

        plan_output = PlanOutput(
            context=ContextOutput(**result.outputs["context"]),
            architecture=ArchitectureOutput(**result.outputs["architecture"]),
            design=DesignOutput(**result.outputs["design"]),
            roadmap=RoadmapOutput(**result.outputs["roadmap"]),
            risk=RiskOutput(**result.outputs["risk"]),
            section_scores=section_scores,
            overall_quality_score=overall_score,
            git_sha=result.git_sha or "",
            generated_at=datetime.now(),
        )

        # Step 5: Render to markdown
        if self._renderer:
            markdown = self._renderer.render(plan_output, head_advanced=result.head_advanced)
        else:
            markdown = "# Plan output\n\n(Renderer not configured)"

        # Step 6: Write to file
        await self._store.update(job.job_id, progress=0.98, current_phase="writing_file")
        output_dir = Path(self._config.output.plans_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_name = f"plan_{job.job_id}_{timestamp}.md"
        file_path = output_dir / file_name

        file_path.write_text(markdown, encoding="utf-8")

        # Step 7: Update job with file path and quality score
        await self._store.update(
            job.job_id,
            progress=0.99,
            current_phase="finalizing",
            file_path=str(file_path),
            quality_score=overall_score,
        )

        logger.info(
            f"Job {job.job_id} completed: plan written to {file_path} "
            f"(quality={overall_score:.2f})"
        )

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

