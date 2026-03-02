# tests/unit/test_grounding_instruction.py
"""Tests for grounding rule injection in _extract_field_group."""

from unittest.mock import AsyncMock

import pytest

from fitz_graveyard.planning.pipeline.stages import ContextStage


class TestGroundingInstruction:
    """Verify GROUNDING RULE is injected when extra_context is provided."""

    @pytest.mark.asyncio
    async def test_grounding_rule_included_with_context(self):
        """GROUNDING RULE appears in prompt when extra_context is non-empty."""
        stage = ContextStage()
        mock_client = AsyncMock()
        mock_client.generate.return_value = '{"key": "value"}'

        await stage._extract_field_group(
            mock_client,
            "Some reasoning text",
            ["key"],
            '{"key": "str"}',
            "test_group",
            extra_context="## some_file.py\nclass Foo: ...",
        )

        call_args = mock_client.generate.call_args
        messages = call_args.kwargs.get("messages", call_args.args[0] if call_args.args else None)
        user_content = messages[-1]["content"]
        assert "GROUNDING RULE" in user_content
        assert "MUST appear in the codebase context above" in user_content

    @pytest.mark.asyncio
    async def test_grounding_rule_absent_without_context(self):
        """GROUNDING RULE does NOT appear when extra_context is empty."""
        stage = ContextStage()
        mock_client = AsyncMock()
        mock_client.generate.return_value = '{"key": "value"}'

        await stage._extract_field_group(
            mock_client,
            "Some reasoning text",
            ["key"],
            '{"key": "str"}',
            "test_group",
            extra_context="",
        )

        call_args = mock_client.generate.call_args
        messages = call_args.kwargs.get("messages", call_args.args[0] if call_args.args else None)
        user_content = messages[-1]["content"]
        assert "GROUNDING RULE" not in user_content
