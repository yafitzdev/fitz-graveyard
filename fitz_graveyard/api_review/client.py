# fitz_graveyard/api_review/client.py
"""
Anthropic review client for async section review.

Uses AsyncAnthropic for concurrent review of multiple plan sections.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from .cost_calculator import CostCalculator
from .schemas import ReviewRequest, ReviewResult, build_review_prompt

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)


class AnthropicReviewClient:
    """Async client for reviewing plan sections using Anthropic API."""

    SYSTEM_PROMPT = (
        "You are an expert technical reviewer for software project plans. "
        "Provide actionable, substantive feedback that helps improve plan quality."
    )

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        """
        Initialize Anthropic review client.

        Args:
            api_key: Anthropic API key
            model: Model to use for review (default: Claude Sonnet 4.5)
        """
        self._api_key = api_key
        self._model = model
        self._client: "anthropic.AsyncAnthropic | None" = None

    @property
    def client(self) -> "anthropic.AsyncAnthropic":
        """Lazy-loaded Anthropic client."""
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def review_sections(self, sections: list[ReviewRequest]) -> list[ReviewResult]:
        """
        Review multiple sections concurrently.

        Args:
            sections: Sections to review

        Returns:
            Review results (one per section, including failures)

        Note:
            Each section is reviewed independently. Failures in one section
            do not affect others.
        """
        # Review all sections concurrently
        review_tasks = [self._review_single_section(section) for section in sections]
        results = await asyncio.gather(*review_tasks, return_exceptions=True)

        # Convert exceptions to ReviewResult with error
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Review failed for section '{sections[i].section_name}': {result}",
                    exc_info=result,
                )
                processed_results.append(
                    ReviewResult(
                        section_name=sections[i].section_name,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _review_single_section(self, section: ReviewRequest) -> ReviewResult:
        """
        Review a single section using Anthropic API.

        Args:
            section: Section to review

        Returns:
            Review result with feedback and token usage

        Raises:
            Exception: On API errors (handled by review_sections)
        """
        try:
            # Build review prompt
            prompt = build_review_prompt(section)

            # Call Anthropic API
            response = await self.client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract feedback from response
            feedback = response.content[0].text if response.content else ""

            return ReviewResult(
                section_name=section.section_name,
                success=True,
                feedback=feedback,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        except Exception as e:
            # Log and re-raise for gather() to handle
            logger.warning(f"API call failed for section '{section.section_name}': {e}")
            raise

    def get_cost_calculator(self) -> CostCalculator:
        """
        Get cost calculator initialized with same client.

        Returns:
            CostCalculator instance
        """
        return CostCalculator(client=self.client)

    async def close(self):
        """Close the async client if initialized."""
        if self._client is not None:
            await self._client.close()
            self._client = None
