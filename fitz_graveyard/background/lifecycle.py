# fitz_graveyard/background/lifecycle.py
"""
Server lifecycle management.

Coordinates startup (DB initialization + crash recovery + worker) and shutdown.
"""

import logging

from fitz_graveyard.background.signals import setup_signal_handlers
from fitz_graveyard.background.worker import BackgroundWorker
from fitz_graveyard.config.schema import FitzPlannerConfig
from fitz_graveyard.llm.client import OllamaClient
from fitz_graveyard.llm.factory import create_llm_client
from fitz_graveyard.llm.lm_studio import LMStudioClient
from fitz_graveyard.models.sqlite_store import SQLiteJobStore

logger = logging.getLogger(__name__)


class ServerLifecycle:
    """
    Server lifecycle coordinator.

    Manages:
        - Database initialization and crash recovery on startup
        - Background worker lifecycle
        - Signal handler registration
        - Graceful shutdown
    """

    def __init__(self, db_path: str, config: FitzPlannerConfig | None = None) -> None:
        """
        Initialize server lifecycle manager.

        Args:
            db_path: Path to SQLite database file
            config: Optional FitzPlannerConfig for OllamaClient creation
        """
        self._store = SQLiteJobStore(db_path)

        # Create LLM client if config provided
        self._ollama_client: OllamaClient | LMStudioClient | None = None
        if config:
            self._ollama_client = create_llm_client(config)

        # Create worker with Ollama client and memory threshold
        self._worker = BackgroundWorker(
            self._store,
            config=config,
            ollama_client=self._ollama_client,
            memory_threshold=config.ollama.memory_threshold if config else 80.0,
        )
        logger.info(f"Created ServerLifecycle with db_path={db_path}")

    @property
    def store(self) -> SQLiteJobStore:
        """Get the job store (for server.py to pass to tools)."""
        return self._store

    @property
    def worker(self) -> BackgroundWorker:
        """Get the background worker (for inspection/testing)."""
        return self._worker

    @property
    def ollama_client(self) -> OllamaClient | LMStudioClient | None:
        """Get the LLM client (for inspection/testing)."""
        return self._ollama_client

    async def startup(self) -> None:
        """
        Start the server lifecycle.

        Steps:
            1. Initialize database schema
            2. Run crash recovery (mark running → interrupted)
            3. Register signal handlers for graceful shutdown
            4. Start background worker

        Logs any interrupted jobs found during crash recovery.
        """
        logger.info("Starting server lifecycle...")

        # Initialize DB schema and run crash recovery
        await self._store.initialize()

        # Log any interrupted jobs (the initialize method already marks them)
        interrupted = [
            job
            for job in await self._store.list_all()
            if job.state.value == "interrupted"
        ]
        if interrupted:
            logger.warning(
                f"Found {len(interrupted)} interrupted job(s) from previous session"
            )
            for job in interrupted:
                logger.warning(f"  - {job.job_id}: {job.description}")

        # Register signal handlers
        setup_signal_handlers(self._worker, self._store)

        # Start background worker
        await self._worker.start()

        logger.info("Server lifecycle started: worker running, signals registered")

    async def shutdown(self) -> None:
        """
        Shut down the server lifecycle gracefully.

        Steps:
            1. Stop background worker (marks running job as interrupted)
            2. Close database (WAL checkpoint)
        """
        logger.info("Shutting down server lifecycle...")

        # Stop worker
        await self._worker.stop()

        # Close DB
        await self._store.close()

        logger.info("Server lifecycle shutdown complete")
