# tests/unit/test_api_review_schemas.py
"""Tests for api_review schemas."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from fitz_graveyard.api_review.schemas import (
    CostBreakdown,
    CostEstimate,
    ReviewRequest,
    ReviewResult,
    build_review_prompt,
)


class TestReviewRequest:
    """Tests for ReviewRequest schema."""

    def test_valid_request(self):
        """Test creating a valid review request."""
        request = ReviewRequest(
            section_name="objective",
            section_type="planning",
            content="Build a feature",
            confidence_score=0.65,
            flag_reason="Low confidence",
        )
        assert request.section_name == "objective"
        assert request.confidence_score == 0.65

    def test_invalid_confidence_score(self):
        """Test confidence score validation."""
        with pytest.raises(ValidationError):
            ReviewRequest(
                section_name="test",
                section_type="test",
                content="test",
                confidence_score=1.5,  # Invalid: > 1.0
                flag_reason="test",
            )

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored (forward compatibility)."""
        request = ReviewRequest(
            section_name="test",
            section_type="test",
            content="test",
            confidence_score=0.5,
            flag_reason="test",
            extra_field="ignored",  # Should be ignored
        )
        assert request.section_name == "test"


class TestCostEstimate:
    """Tests for CostEstimate schema."""

    def test_valid_estimate(self):
        """Test creating a valid cost estimate."""
        estimate = CostEstimate(
            sections_count=3,
            input_tokens=1500,
            estimated_output_tokens=2250,
            cost_usd=0.056,
            cost_eur=0.052,
            model="claude-sonnet-4-5-20250929",
        )
        assert estimate.sections_count == 3
        assert estimate.input_tokens == 1500
        assert estimate.cost_usd == 0.056

    def test_to_user_display(self):
        """Test formatted cost display."""
        estimate = CostEstimate(
            sections_count=2,
            input_tokens=1000,
            estimated_output_tokens=1500,
            cost_usd=0.025,
            cost_eur=0.023,
            model="test-model",
        )
        display = estimate.to_user_display()

        assert "Sections to review: 2" in display
        assert "Model: test-model" in display
        assert "Input tokens: 1,000" in display
        assert "Estimated output tokens: 1,500 (1.5x input)" in display
        assert "Total tokens: 2,500" in display
        assert "$0.0250 USD" in display
        assert "0.0230 EUR" in display
        assert "1.5x multiplier" in display

    def test_negative_tokens_rejected(self):
        """Test that negative token counts are rejected."""
        with pytest.raises(ValidationError):
            CostEstimate(
                sections_count=1,
                input_tokens=-100,  # Invalid: negative
                estimated_output_tokens=150,
                cost_usd=0.01,
                cost_eur=0.009,
                model="test",
            )

    def test_default_timestamp(self):
        """Test that timestamp defaults to current UTC time."""
        estimate = CostEstimate(
            sections_count=1,
            input_tokens=100,
            estimated_output_tokens=150,
            cost_usd=0.001,
            cost_eur=0.0009,
            model="test",
        )
        # Should be close to now (within 1 second)
        now = datetime.now(timezone.utc)
        assert (now - estimate.timestamp).total_seconds() < 1.0


class TestReviewResult:
    """Tests for ReviewResult schema."""

    def test_successful_result(self):
        """Test a successful review result."""
        result = ReviewResult(
            section_name="objective",
            success=True,
            feedback="Looks good with minor suggestions",
            input_tokens=500,
            output_tokens=150,
        )
        assert result.success is True
        assert result.feedback is not None
        assert result.error is None

    def test_failed_result(self):
        """Test a failed review result."""
        result = ReviewResult(
            section_name="tasks",
            success=False,
            error="Rate limit exceeded",
        )
        assert result.success is False
        assert result.error is not None
        assert result.feedback is None

    def test_default_token_counts(self):
        """Test that token counts default to 0."""
        result = ReviewResult(section_name="test", success=False)
        assert result.input_tokens == 0
        assert result.output_tokens == 0


class TestCostBreakdown:
    """Tests for CostBreakdown schema."""

    def test_complete_breakdown(self):
        """Test a complete cost breakdown."""
        estimate = CostEstimate(
            sections_count=2,
            input_tokens=1000,
            estimated_output_tokens=1500,
            cost_usd=0.025,
            cost_eur=0.023,
            model="test",
        )

        breakdown = CostBreakdown(
            estimate=estimate,
            actual_input_tokens=950,
            actual_output_tokens=1200,
            actual_cost_usd=0.020,
            actual_cost_eur=0.018,
            sections_reviewed=2,
            sections_failed=0,
        )

        assert breakdown.estimate is not None
        assert breakdown.actual_input_tokens == 950
        assert breakdown.sections_reviewed == 2

    def test_default_values(self):
        """Test that all fields have sensible defaults."""
        breakdown = CostBreakdown()
        assert breakdown.estimate is None
        assert breakdown.actual_input_tokens == 0
        assert breakdown.actual_output_tokens == 0
        assert breakdown.actual_cost_usd == 0.0
        assert breakdown.actual_cost_eur == 0.0
        assert breakdown.sections_reviewed == 0
        assert breakdown.sections_failed == 0


class TestBuildReviewPrompt:
    """Tests for build_review_prompt helper."""

    def test_prompt_includes_all_fields(self):
        """Test that prompt includes all section metadata."""
        request = ReviewRequest(
            section_name="architecture",
            section_type="design",
            content="Use microservices pattern",
            confidence_score=0.72,
            flag_reason="Lacks specifics",
        )

        prompt = build_review_prompt(request)

        assert "architecture" in prompt
        assert "design" in prompt
        assert "Use microservices pattern" in prompt
        assert "0.72" in prompt
        assert "Lacks specifics" in prompt

    def test_prompt_has_review_instructions(self):
        """Test that prompt includes review instructions."""
        request = ReviewRequest(
            section_name="test",
            section_type="test",
            content="test content",
            confidence_score=0.5,
            flag_reason="test",
        )

        prompt = build_review_prompt(request)

        # Check for key review criteria
        assert "missing" in prompt.lower()
        assert "ambiguities" in prompt.lower() or "ambiguous" in prompt.lower()
        assert "technical" in prompt.lower()
        assert "improvement" in prompt.lower()
