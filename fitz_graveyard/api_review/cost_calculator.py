# fitz_graveyard/api_review/cost_calculator.py
"""
Cost calculator for API review using Anthropic's count_tokens API.

Pricing: Claude Sonnet 4.5
- Input: $3.00 per million tokens
- Output: $15.00 per million tokens
"""

import logging
from typing import TYPE_CHECKING

from .schemas import CostBreakdown, CostEstimate, ReviewRequest, ReviewResult, build_review_prompt

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)


class CostCalculator:
    """Calculates API review costs using official Anthropic token counting API."""

    # Anthropic Sonnet 4.5 pricing (2026)
    INPUT_PRICE = 3.00 / 1_000_000  # $3 per million tokens
    OUTPUT_PRICE = 15.00 / 1_000_000  # $15 per million tokens
    OUTPUT_MULTIPLIER = 1.5  # Estimate output at 1.5x input for safety

    def __init__(self, client: "anthropic.AsyncAnthropic"):
        """
        Initialize cost calculator.

        Args:
            client: Anthropic async client for token counting
        """
        self._client = client
        self._converter = None  # Lazy-loaded CurrencyConverter

    async def estimate_review_cost(
        self, sections: list[ReviewRequest], model: str
    ) -> CostEstimate:
        """
        Estimate cost of reviewing sections using count_tokens API.

        Args:
            sections: Sections to review
            model: Model to use for review

        Returns:
            Cost estimate with token counts and USD/EUR costs

        Note:
            Falls back to 0 tokens if count_tokens API fails (rate limit, network error).
            Output tokens estimated at 1.5x input for safety margin.
        """
        total_input_tokens = 0

        # Count tokens for each section using official API
        for section in sections:
            try:
                prompt = build_review_prompt(section)

                # Build messages in Anthropic format
                messages = [{"role": "user", "content": prompt}]

                # Use official count_tokens API
                token_count = await self._client.messages.count_tokens(
                    model=model,
                    messages=messages,
                    system="You are an expert technical reviewer for software project plans.",
                )
                total_input_tokens += token_count.input_tokens

            except Exception as e:
                # Handle rate limits and API errors gracefully
                import anthropic

                if isinstance(e, (anthropic.RateLimitError, anthropic.APIError)):
                    logger.warning(
                        f"Token counting failed for section '{section.section_name}': {e}. "
                        "Using 0 tokens for estimate."
                    )
                else:
                    logger.error(f"Unexpected error counting tokens: {e}", exc_info=True)

        # Estimate output tokens at 1.5x input
        estimated_output_tokens = int(total_input_tokens * self.OUTPUT_MULTIPLIER)

        # Calculate USD cost
        cost_usd = (
            total_input_tokens * self.INPUT_PRICE
            + estimated_output_tokens * self.OUTPUT_PRICE
        )

        # Convert to EUR
        cost_eur = self._convert_usd_to_eur(cost_usd)

        return CostEstimate(
            sections_count=len(sections),
            input_tokens=total_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            cost_usd=cost_usd,
            cost_eur=cost_eur,
            model=model,
        )

    def calculate_actual_cost(
        self, results: list[ReviewResult], model: str
    ) -> CostBreakdown:
        """
        Calculate actual cost from review results.

        Args:
            results: Review results with actual token usage
            model: Model used for review

        Returns:
            Cost breakdown with actual vs estimated costs
        """
        actual_input_tokens = sum(r.input_tokens for r in results)
        actual_output_tokens = sum(r.output_tokens for r in results)
        sections_reviewed = sum(1 for r in results if r.success)
        sections_failed = sum(1 for r in results if not r.success)

        # Calculate USD cost
        actual_cost_usd = (
            actual_input_tokens * self.INPUT_PRICE
            + actual_output_tokens * self.OUTPUT_PRICE
        )

        # Convert to EUR
        actual_cost_eur = self._convert_usd_to_eur(actual_cost_usd)

        return CostBreakdown(
            estimate=None,  # Set by caller if they have the estimate
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
            actual_cost_usd=actual_cost_usd,
            actual_cost_eur=actual_cost_eur,
            sections_reviewed=sections_reviewed,
            sections_failed=sections_failed,
        )

    def _convert_usd_to_eur(self, usd_amount: float) -> float:
        """
        Convert USD to EUR using ECB rates.

        Falls back to hardcoded 0.92 rate if conversion fails.
        """
        try:
            # Lazy-load CurrencyConverter
            if self._converter is None:
                from currency_converter import CurrencyConverter

                self._converter = CurrencyConverter()

            return self._converter.convert(usd_amount, "USD", "EUR")

        except Exception as e:
            # Fallback to hardcoded rate if conversion fails (no network, etc.)
            logger.warning(
                f"EUR conversion failed: {e}. Using fallback rate 0.92. "
                "Check network connection for accurate rates."
            )
            return usd_amount * 0.92
