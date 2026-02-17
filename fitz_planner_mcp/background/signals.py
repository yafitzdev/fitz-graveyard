# fitz_planner_mcp/background/signals.py
"""
Graceful shutdown signal handling for Windows and Unix.

Registers SIGINT/SIGTERM handlers to trigger worker shutdown and DB cleanup.
"""

import asyncio
import logging
import signal

from fitz_planner_mcp.background.worker import BackgroundWorker
from fitz_planner_mcp.models.store import JobStore

logger = logging.getLogger(__name__)


def setup_signal_handlers(worker: BackgroundWorker, store: JobStore) -> None:
    """
    Set up signal handlers for graceful shutdown.

    Handles SIGINT (Ctrl+C) and SIGTERM with platform-specific fallbacks.

    On Windows (ProactorEventLoop), add_signal_handler is not supported,
    so we fall back to signal.signal().

    Args:
        worker: BackgroundWorker to stop on shutdown
        store: JobStore to close (WAL checkpoint)
    """
    loop = asyncio.get_event_loop()

    async def _shutdown(sig_name: str) -> None:
        """
        Async shutdown handler.

        Args:
            sig_name: Signal name for logging
        """
        logger.info(f"Received {sig_name}, shutting down gracefully...")

        # Stop worker (marks running job as interrupted)
        await worker.stop()

        # Checkpoint WAL and close DB
        await store.close()

        logger.info("Shutdown complete")

    def _signal_callback(sig_num, frame) -> None:
        """
        Fallback signal handler for Windows.

        Creates an asyncio task for the shutdown coroutine.
        """
        sig_name = signal.Signals(sig_num).name
        logger.info(f"Signal handler triggered: {sig_name}")
        asyncio.create_task(_shutdown(sig_name))

    # Try loop-based signal handling (Unix)
    try:
        loop.add_signal_handler(
            signal.SIGINT,
            lambda: asyncio.create_task(_shutdown("SIGINT")),
        )
        loop.add_signal_handler(
            signal.SIGTERM,
            lambda: asyncio.create_task(_shutdown("SIGTERM")),
        )
        logger.info("Signal handlers registered (loop-based)")

    except NotImplementedError:
        # Fall back to signal.signal() for Windows
        signal.signal(signal.SIGINT, _signal_callback)
        signal.signal(signal.SIGTERM, _signal_callback)
        logger.info("Signal handlers registered (fallback for Windows)")
