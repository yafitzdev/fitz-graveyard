# fitz_graveyard/api_review/schemas.py
"""
Pydantic schemas for API review cost estimation and results.

All models use extra="ignore" for forward compatibility.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReviewRequest(BaseModel):
    """Request for API review of a plan section."""

    model_config = ConfigDict(extra="ignore")

    section_name: str = Field(description="Name of the section being reviewed")
    section_type: str = Field(description="Type of section (e.g., 'objective', 'tasks')")
    content: str = Field(description="Section content to review")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score from LLM self-assessment"
    )
    flag_reason: str = Field(description="Reason this section was flagged for review")


class CostEstimate(BaseModel):
    """Estimated cost for API review before execution."""

    model_config = ConfigDict(extra="ignore")

    sections_count: int = Field(ge=0, description="Number of sections to review")
    input_tokens: int = Field(ge=0, description="Estimated input tokens")
    estimated_output_tokens: int = Field(
        ge=0, description="Estimated output tokens (1.5x input)"
    )
    cost_usd: float = Field(ge=0.0, description="Estimated cost in USD")
    cost_eur: float = Field(ge=0.0, description="Estimated cost in EUR")
    model: str = Field(description="Model to use for review")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When estimate was calculated",
    )

    def to_user_display(self) -> str:
        """Format cost estimate for user display."""
        return (
            f"API Review Cost Estimate\n"
            f"------------------------\n"
            f"Sections to review: {self.sections_count}\n"
            f"Model: {self.model}\n"
            f"Input tokens: {self.input_tokens:,}\n"
            f"Estimated output tokens: {self.estimated_output_tokens:,} (1.5x input)\n"
            f"Total tokens: {self.input_tokens + self.estimated_output_tokens:,}\n"
            f"Cost: ${self.cost_usd:.4f} USD / €{self.cost_eur:.4f} EUR\n"
            f"\n"
            f"Note: Output token estimate uses 1.5x multiplier for safety margin."
        )


class ReviewResult(BaseModel):
    """Result of reviewing a single section."""

    model_config = ConfigDict(extra="ignore")

    section_name: str = Field(description="Name of the reviewed section")
    success: bool = Field(description="Whether review completed successfully")
    feedback: str | None = Field(
        default=None, description="Expert review feedback from API"
    )
    input_tokens: int = Field(default=0, description="Actual input tokens used")
    output_tokens: int = Field(default=0, description="Actual output tokens used")
    error: str | None = Field(default=None, description="Error message if failed")


class CostBreakdown(BaseModel):
    """Actual cost breakdown after API review execution."""

    model_config = ConfigDict(extra="ignore")

    estimate: CostEstimate | None = Field(
        default=None, description="Original pre-execution estimate"
    )
    actual_input_tokens: int = Field(default=0, description="Total input tokens used")
    actual_output_tokens: int = Field(default=0, description="Total output tokens used")
    actual_cost_usd: float = Field(default=0.0, description="Actual cost in USD")
    actual_cost_eur: float = Field(default=0.0, description="Actual cost in EUR")
    sections_reviewed: int = Field(default=0, description="Number of sections reviewed")
    sections_failed: int = Field(default=0, description="Number of sections that failed")


def build_review_prompt(section: ReviewRequest) -> str:
    """
    Build review prompt for a plan section.

    Shared between CostCalculator (for token counting) and AnthropicReviewClient (for actual review).
    """
    return f"""You are an expert technical reviewer for software project plans. Review this plan section for quality and completeness.

**Section Information:**
- Name: {section.section_name}
- Type: {section.section_type}
- Confidence Score: {section.confidence_score:.2f}
- Flag Reason: {section.flag_reason}

**Section Content:**
{section.content}

**Review Instructions:**
Analyze the section for:
1. Missing critical details or assumptions
2. Ambiguities that could lead to implementation errors
3. Technical inaccuracies or unrealistic expectations
4. Opportunities for improvement or clarification

Provide actionable feedback that helps improve the plan. Be concise but thorough.
Focus on substantive issues, not minor style points."""
