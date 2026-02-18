# tests/unit/test_api_review_client.py
"""Tests for AnthropicReviewClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fitz_planner_mcp.api_review.client import AnthropicReviewClient
from fitz_planner_mcp.api_review.schemas import ReviewRequest, ReviewResult


@pytest.fixture
def mock_anthropic():
    """Mock Anthropic AsyncAnthropic client."""
    # Patch where anthropic is imported (inside the client property)
    with patch("anthropic.AsyncAnthropic") as mock_anthropic_class:
        # Create mock client
        mock_client = AsyncMock()
        mock_anthropic_class.return_value = mock_client

        # Create mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Good plan with minor improvements needed")]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 150

        mock_client.messages.create.return_value = mock_response
        mock_client.close = AsyncMock()

        yield mock_client


@pytest.fixture
def review_client():
    """Create a review client."""
    return AnthropicReviewClient(api_key="test-key", model="test-model")


@pytest.fixture
def sample_section():
    """Create a sample review request."""
    return ReviewRequest(
        section_name="objective",
        section_type="planning",
        content="Build a feature",
        confidence_score=0.65,
        flag_reason="Low confidence",
    )


class TestAnthropicReviewClient:
    """Tests for AnthropicReviewClient."""

    def test_initialization(self):
        """Test client initialization."""
        client = AnthropicReviewClient(api_key="test-key")
        assert client._api_key == "test-key"
        assert client._model == "claude-sonnet-4-5-20250929"  # Default model
        assert client._client is None  # Lazy-loaded

    def test_custom_model(self):
        """Test initialization with custom model."""
        client = AnthropicReviewClient(api_key="test-key", model="custom-model")
        assert client._model == "custom-model"

    def test_lazy_client_loading(self, review_client, mock_anthropic):
        """Test that client is lazy-loaded on first access."""
        assert review_client._client is None
        _ = review_client.client  # Access property
        assert review_client._client is not None

    @pytest.mark.asyncio
    async def test_review_single_section_success(
        self, review_client, mock_anthropic, sample_section
    ):
        """Test successful review of a single section."""
        result = await review_client._review_single_section(sample_section)

        assert isinstance(result, ReviewResult)
        assert result.section_name == "objective"
        assert result.success is True
        assert result.feedback == "Good plan with minor improvements needed"
        assert result.input_tokens == 500
        assert result.output_tokens == 150
        assert result.error is None

        # Verify API was called
        mock_anthropic.messages.create.assert_called_once()
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["max_tokens"] == 2048
        assert "expert technical reviewer" in call_kwargs["system"]
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_review_single_section_api_error(self, review_client, mock_anthropic):
        """Test handling of API errors during review."""
        # Make API call raise an error
        mock_anthropic.messages.create.side_effect = Exception("API error")

        section = ReviewRequest(
            section_name="test",
            section_type="test",
            content="test",
            confidence_score=0.5,
            flag_reason="test",
        )

        # Should raise the exception (for gather to handle)
        with pytest.raises(Exception, match="API error"):
            await review_client._review_single_section(section)

    @pytest.mark.asyncio
    async def test_review_sections_all_success(
        self, review_client, mock_anthropic, sample_section
    ):
        """Test concurrent review of multiple sections."""
        sections = [
            sample_section,
            ReviewRequest(
                section_name="tasks",
                section_type="implementation",
                content="Task list",
                confidence_score=0.68,
                flag_reason="Vague tasks",
            ),
        ]

        results = await review_client.review_sections(sections)

        assert len(results) == 2
        assert all(isinstance(r, ReviewResult) for r in results)
        assert all(r.success for r in results)
        assert results[0].section_name == "objective"
        assert results[1].section_name == "tasks"

        # API should be called twice (once per section)
        assert mock_anthropic.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_review_sections_partial_failure(self, review_client, mock_anthropic):
        """Test that failures in one section don't affect others."""
        sections = [
            ReviewRequest(
                section_name="section1",
                section_type="test",
                content="test",
                confidence_score=0.5,
                flag_reason="test",
            ),
            ReviewRequest(
                section_name="section2",
                section_type="test",
                content="test",
                confidence_score=0.5,
                flag_reason="test",
            ),
        ]

        # Make second call fail
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Second section failed")
            # Return mock response for first call
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Review feedback")]
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            return mock_response

        mock_anthropic.messages.create.side_effect = side_effect

        results = await review_client.review_sections(sections)

        assert len(results) == 2
        assert results[0].success is True
        assert results[0].section_name == "section1"
        assert results[1].success is False
        assert results[1].section_name == "section2"
        assert results[1].error == "Second section failed"

    def test_get_cost_calculator(self, review_client, mock_anthropic):
        """Test getting cost calculator with same client."""
        calculator = review_client.get_cost_calculator()

        # Should use the same client
        assert calculator._client is review_client.client

    @pytest.mark.asyncio
    async def test_close(self, review_client, mock_anthropic):
        """Test closing the client."""
        # Access client to initialize it
        _ = review_client.client

        await review_client.close()

        # Client should be closed and set to None
        mock_anthropic.close.assert_called_once()
        assert review_client._client is None

    @pytest.mark.asyncio
    async def test_close_before_initialization(self, review_client):
        """Test closing client that was never initialized."""
        # Should not raise error
        await review_client.close()
        assert review_client._client is None


class TestConfigIntegration:
    """Tests for config schema integration."""

    def test_anthropic_config_defaults(self):
        """Test AnthropicConfig default values."""
        from fitz_planner_mcp.config.schema import AnthropicConfig

        config = AnthropicConfig()
        assert config.api_key is None
        assert config.model == "claude-sonnet-4-5-20250929"
        assert config.max_review_tokens == 2048

    def test_anthropic_config_in_root(self):
        """Test AnthropicConfig nested in FitzPlannerConfig."""
        from fitz_planner_mcp.config.schema import FitzPlannerConfig

        config = FitzPlannerConfig()
        assert hasattr(config, "anthropic")
        assert config.anthropic.api_key is None
        assert config.anthropic.model == "claude-sonnet-4-5-20250929"

    def test_anthropic_config_custom_values(self):
        """Test setting custom values in AnthropicConfig."""
        from fitz_planner_mcp.config.schema import AnthropicConfig

        config = AnthropicConfig(
            api_key="sk-test-key", model="custom-model", max_review_tokens=4096
        )
        assert config.api_key == "sk-test-key"
        assert config.model == "custom-model"
        assert config.max_review_tokens == 4096

    def test_anthropic_config_extra_ignored(self):
        """Test that extra fields are ignored (forward compatibility)."""
        from fitz_planner_mcp.config.schema import AnthropicConfig

        config = AnthropicConfig(api_key="test", extra_field="ignored")
        assert config.api_key == "test"
        # Extra field should not raise error
