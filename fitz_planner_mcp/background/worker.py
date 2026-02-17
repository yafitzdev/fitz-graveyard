# fitz_planner_mcp/background/worker.py
"""
Background worker for sequential job processing.

Polls the job queue, processes jobs one at a time, and handles graceful shutdown.
"""

import asyncio
import logging
import traceback

from fitz_planner_mcp.llm.client import OllamaClient
from fitz_planner_mcp.llm.memory import MemoryMonitor
from fitz_planner_mcp.models.jobs import JobState
from fitz_planner_mcp.models.store import JobStore

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
        poll_interval: float = 1.0,
        ollama_client: OllamaClient | None = None,
        memory_threshold: float = 80.0,
    ) -> None:
        """
        Initialize background worker.

        Args:
            store: Job store implementation (SQLite or in-memory)
            poll_interval: Seconds between queue checks (default: 1.0)
            ollama_client: Optional OllamaClient for real plan generation
            memory_threshold: RAM usage % threshold for memory monitoring (default: 80.0)
        """
        self._store = store
        self._poll_interval = poll_interval
        self._ollama_client = ollama_client
        self._memory_threshold = memory_threshold
        self._current_job_id: str | None = None
        self._task: asyncio.Task | None = None
        logger.info(
            f"Initialized BackgroundWorker (poll_interval={poll_interval}s, "
            f"ollama={'configured' if ollama_client else 'None'})"
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
        Process a planning job using OllamaClient.

        Runs health check, generates plan content with memory monitoring,
        and handles fallback to smaller model on OOM.

        Args:
            job: JobRecord to process

        Raises:
            ConnectionError: If Ollama health check fails
            MemoryError: If RAM threshold exceeded during generation
        """
        if self._ollama_client is None:
            logger.warning(
                f"No Ollama client configured, completing job {job.job_id} with stub"
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

        # Step 2: Build prompt from job metadata
        await self._store.update(job.job_id, progress=0.1, current_phase="generating")
        messages = self._build_messages(job)

        # Step 3: Generate with memory monitoring and fallback
        monitor = MemoryMonitor(threshold_percent=self._memory_threshold)
        result, model_used = await self._ollama_client.generate_with_monitoring(
            messages=messages,
            monitor=monitor,
        )

        # Step 4: Store result (Phase 4 will write to file)
        await self._store.update(
            job.job_id,
            progress=0.95,
            current_phase="finalizing",
            file_path=None,  # Phase 4 will write to file
        )

        logger.info(f"Job {job.job_id} generated {len(result)} chars using {model_used}")

    def _build_messages(self, job) -> list[dict]:
        """
        Build chat messages from job metadata.

        This is a PLACEHOLDER prompt - Phase 4 will replace with full multi-stage pipeline.

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
