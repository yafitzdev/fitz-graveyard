# fitz_graveyard/planning/krag/client.py
"""
KragClient - Wrapper for fitz-ai SDK with graceful fallback and multi-query support.

Handles lazy initialization, ImportError fallback, ABSTAIN answer filtering, and
exception handling for robust KRAG integration.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fitz_graveyard.config.schema import KragConfig

from .formatter import format_krag_answer

logger = logging.getLogger(__name__)


class KragClient:
    """
    Client for querying fitz-ai SDK with graceful error handling.

    Features:
    - Lazy SDK initialization (imports fitz only when first query executes)
    - ImportError fallback (returns empty string if fitz-ai not installed)
    - ABSTAIN answer filtering (skips non-answers)
    - Exception handling (logs and returns empty string on errors)
    - Multi-query aggregation (combines multiple queries into one context block)
    """

    def __init__(
        self,
        enabled: bool = True,
        fitz_ai_config: str | None = None,
        source_dir: str | None = None,
    ):
        """
        Initialize KragClient.

        Args:
            enabled: Whether KRAG is enabled (if False, all methods return "")
            fitz_ai_config: Path to fitz-ai config file (None uses defaults)
            source_dir: Source directory to point fitz to (None skips pointing)
        """
        self._enabled = enabled
        self._fitz_ai_config = fitz_ai_config
        self._source_dir = source_dir
        self._fitz_instance = None

    def _get_fitz(self):
        """
        Lazy-initialize fitz SDK instance.

        Returns:
            fitz module or None if import failed.

        Logs debug message on ImportError (fitz-ai not installed).
        Caches instance for reuse across queries.
        """
        if not self._enabled:
            return None

        if self._fitz_instance is not None:
            return self._fitz_instance

        try:
            # Lazy import - only import when first query executes
            from fitz_ai import fitz

            self._fitz_instance = fitz

            # Point to source directory if provided
            if self._source_dir is not None:
                try:
                    fitz.point(self._source_dir)
                except Exception as e:
                    logger.warning(f"Failed to point fitz to {self._source_dir}: {e}")

            return self._fitz_instance

        except ImportError:
            logger.debug(
                "fitz-ai not installed - KRAG queries will return empty context. "
                "Install with: pip install fitz-ai"
            )
            return None

    def query(self, question: str) -> str:
        """
        Query fitz-ai SDK with a single question.

        Args:
            question: Question to ask the codebase

        Returns:
            Formatted markdown context or empty string on failure/ABSTAIN.

        Logs warning on exceptions, returns empty string instead of raising.
        """
        if not self._enabled:
            return ""

        try:
            fitz = self._get_fitz()
            if fitz is None:
                return ""

            # Import AnswerMode for ABSTAIN check (lazy import)
            from fitz_ai.core.answer_mode import AnswerMode

            answer = fitz.ask(question)

            # Skip ABSTAIN answers (fitz couldn't answer the question)
            if answer.mode == AnswerMode.ABSTAIN:
                logger.debug(f"fitz ABSTAIN for query: {question}")
                return ""

            return format_krag_answer(answer)

        except Exception as e:
            logger.warning(f"KRAG query failed: {e}")
            return ""

    def multi_query(self, queries: list[str]) -> str:
        """
        Execute multiple queries and aggregate results into structured markdown.

        Args:
            queries: List of questions to ask

        Returns:
            Aggregated markdown with section headers, or empty string if all queries fail.

        Format:
            ## Codebase Context

            ### Query 1 text
            answer 1

            ### Query 2 text
            answer 2
        """
        if not self._enabled:
            return ""

        results = []
        for query in queries:
            result = self.query(query)
            if result:
                results.append((query, result))

        if not results:
            return ""

        sections = [f"### {query}\n\n{result}\n" for query, result in results]
        return "## Codebase Context\n\n" + "\n".join(sections)

    @classmethod
    def from_config(cls, config: "KragConfig", source_dir: str | None = None) -> "KragClient":
        """
        Create KragClient from KragConfig.

        Args:
            config: KragConfig instance
            source_dir: Source directory to point fitz to (overrides config)

        Returns:
            Configured KragClient instance.
        """
        return cls(
            enabled=config.enabled,
            fitz_ai_config=config.fitz_ai_config,
            source_dir=source_dir,
        )
