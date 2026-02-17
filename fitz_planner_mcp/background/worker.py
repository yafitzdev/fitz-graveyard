# fitz_planner_mcp/background/worker.py
"""
Background worker for sequential job processing.

Polls the job queue, processes jobs one at a time, and handles graceful shutdown.
"""

import asyncio
import logging
import traceback

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

    def __init__(self, store: JobStore, poll_interval: float = 1.0) -> None:
        """
        Initialize background worker.

        Args:
            store: Job store implementation (SQLite or in-memory)
            poll_interval: Seconds between queue checks (default: 1.0)
        """
        self._store = store
        self._poll_interval = poll_interval
        self._current_job_id: str | None = None
        self._task: asyncio.Task | None = None
        logger.info(f"Initialized BackgroundWorker (poll_interval={poll_interval}s)")

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
        Process a single job (stub for Phase 4).

        This is where the planning engine will be integrated.
        For now, logs the job and simulates minimal processing.

        Args:
            job: JobRecord to process
        """
        logger.info(
            f"Processing job {job.job_id} (planning engine not yet implemented)"
        )

        # Simulate some work
        await self._store.update(
            job.job_id,
            progress=0.5,
            current_phase="pending_engine",
        )

        # Minimal delay so tests can observe state changes
        await asyncio.sleep(0.1)

        logger.info(f"Job {job.job_id} processing complete (stub)")
