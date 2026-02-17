# fitz_planner_mcp/llm/memory.py
"""Memory monitoring for detecting RAM threshold violations during LLM generation."""

import asyncio
import logging

import psutil

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Monitors system RAM usage and detects when a threshold is exceeded.

    Used to abort long-running LLM generation if RAM usage becomes dangerous.
    """

    def __init__(self, threshold_percent: float = 80.0):
        """
        Initialize memory monitor.

        Args:
            threshold_percent: RAM usage % threshold (0-100). Defaults to 80%.
        """
        self.threshold_percent = threshold_percent
        self._running = False

    def check_once(self) -> tuple[float, bool]:
        """
        Check current RAM usage once.

        Returns:
            (current_percent, exceeded): Current RAM % and whether it exceeds threshold.
        """
        current_percent = psutil.virtual_memory().percent
        exceeded = current_percent >= self.threshold_percent
        return current_percent, exceeded

    async def start_monitoring(self, check_interval: float = 5.0) -> bool:
        """
        Start monitoring RAM usage at regular intervals.

        Runs until threshold is exceeded or stop() is called.

        Args:
            check_interval: Seconds between checks. Defaults to 5.0.

        Returns:
            True if threshold was exceeded, False if stopped via stop().
        """
        self._running = True
        logger.info(
            f"Memory monitor started (threshold={self.threshold_percent}%, interval={check_interval}s)"
        )

        while self._running:
            current_percent, exceeded = self.check_once()

            if exceeded:
                logger.warning(
                    f"Memory threshold exceeded: {current_percent:.1f}% >= {self.threshold_percent}%"
                )
                self._running = False
                return True

            await asyncio.sleep(check_interval)

        logger.info("Memory monitor stopped")
        return False

    def stop(self):
        """Stop monitoring (makes start_monitoring return False)."""
        self._running = False
