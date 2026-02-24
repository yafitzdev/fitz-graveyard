# tests/unit/test_agent_gatherer.py
"""Unit tests for AgentContextGatherer."""

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from fitz_graveyard.config.schema import AgentConfig
from fitz_graveyard.llm.types import AgentMessage, AgentToolCall
from fitz_graveyard.planning.agent.gatherer import AgentContextGatherer


def _make_config(**kwargs):
    defaults = dict(enabled=True, max_iterations=5, max_file_bytes=50_000)
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _make_response(content="", tool_calls=None):
    """Build a normalized AgentMessage (new interface)."""
    agent_tool_calls = None
    if tool_calls is not None:
        agent_tool_calls = [
            AgentToolCall(id="", name=tc["name"], arguments=tc["args"])
            for tc in tool_calls
        ]
    assistant_dict = {"role": "assistant", "content": content}
    return AgentMessage(
        content=content,
        tool_calls=agent_tool_calls,
        assistant_dict=assistant_dict,
    )


def _make_tool_call(name, **args):
    """Build an AgentToolCall dict for _make_response."""
    return {"name": name, "args": args}


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.model = "test-model"
    client.generate_with_tools = AsyncMock()
    client.tool_result_message = MagicMock(
        side_effect=lambda tool_call_id, content: {"role": "tool", "content": content}
    )
    return client


class TestGatherDisabled:
    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self, tmp_path, mock_client):
        gatherer = AgentContextGatherer(
            config=_make_config(enabled=False), source_dir=str(tmp_path)
        )
        result = await gatherer.gather(mock_client, "test task")
        assert result == ""
        mock_client.generate_with_tools.assert_not_called()


class TestGatherSingleIteration:
    @pytest.mark.asyncio
    async def test_returns_content_when_no_tool_calls(self, tmp_path, mock_client):
        mock_client.generate_with_tools.return_value = _make_response(
            content="## Project Overview\nA test project.", tool_calls=None
        )
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer.gather(mock_client, "add feature X")
        assert "## Project Overview" in result
        assert mock_client.generate_with_tools.call_count == 1

    @pytest.mark.asyncio
    async def test_uses_agent_model_when_configured(self, tmp_path, mock_client):
        mock_client.generate_with_tools.return_value = _make_response(
            content="done", tool_calls=None
        )
        gatherer = AgentContextGatherer(
            config=_make_config(agent_model="special-model"),
            source_dir=str(tmp_path),
        )
        await gatherer.gather(mock_client, "task")
        _, kwargs = mock_client.generate_with_tools.call_args
        assert kwargs["model"] == "special-model"

    @pytest.mark.asyncio
    async def test_falls_back_to_client_model(self, tmp_path, mock_client):
        mock_client.generate_with_tools.return_value = _make_response(
            content="done", tool_calls=None
        )
        gatherer = AgentContextGatherer(
            config=_make_config(agent_model=None), source_dir=str(tmp_path)
        )
        await gatherer.gather(mock_client, "task")
        _, kwargs = mock_client.generate_with_tools.call_args
        assert kwargs["model"] == "test-model"


class TestGatherWithToolCalls:
    @pytest.mark.asyncio
    async def test_executes_tools_then_returns_final_content(self, tmp_path, mock_client):
        # First call: has tool_calls (list_directory)
        # Second call: no tool_calls (final answer)
        tc = _make_tool_call("list_directory", path=".")

        # Create a real file for the tool to find
        (tmp_path / "main.py").write_text("# main")

        mock_client.generate_with_tools.side_effect = [
            _make_response(content="", tool_calls=[tc]),
            _make_response(content="## Context\nFound main.py.", tool_calls=None),
        ]


        gatherer = AgentContextGatherer(
            config=_make_config(max_iterations=5), source_dir=str(tmp_path)
        )
        result = await gatherer.gather(mock_client, "plan something")

        assert "## Context" in result
        assert mock_client.generate_with_tools.call_count == 2

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_string_continues(self, tmp_path, mock_client):
        tc = _make_tool_call("nonexistent_tool", **{"arg": "val"})
        mock_client.generate_with_tools.side_effect = [
            _make_response(content="", tool_calls=[tc]),
            _make_response(content="Final context.", tool_calls=None),
        ]
        gatherer = AgentContextGatherer(
            config=_make_config(max_iterations=5), source_dir=str(tmp_path)
        )
        result = await gatherer.gather(mock_client, "task")
        # Should not raise; tool error appended to messages, loop continues
        assert result == "Final context."
        assert mock_client.generate_with_tools.call_count == 2


class TestGatherMaxIterations:
    @pytest.mark.asyncio
    async def test_stops_at_max_iterations(self, tmp_path, mock_client):
        tc = _make_tool_call("list_directory", **{"path": "."})
        # Always returns tool_calls — never "done"
        mock_client.generate_with_tools.return_value = _make_response(
            content="partial", tool_calls=[tc]
        )
        gatherer = AgentContextGatherer(
            config=_make_config(max_iterations=3), source_dir=str(tmp_path)
        )
        result = await gatherer.gather(mock_client, "task")
        assert mock_client.generate_with_tools.call_count == 3
        # Returns something (not raises)
        assert isinstance(result, str)


class TestGatherLLMFailure:
    @pytest.mark.asyncio
    async def test_llm_exception_returns_empty(self, tmp_path, mock_client):
        mock_client.generate_with_tools.side_effect = RuntimeError("connection refused")
        gatherer = AgentContextGatherer(
            config=_make_config(max_iterations=3), source_dir=str(tmp_path)
        )
        result = await gatherer.gather(mock_client, "task")
        assert result == ""


class TestGatherProgressCallback:
    @pytest.mark.asyncio
    async def test_callback_called_on_start(self, tmp_path, mock_client):
        mock_client.generate_with_tools.return_value = _make_response(
            content="done", tool_calls=None
        )
        callback_calls = []

        def callback(progress, phase):
            callback_calls.append((progress, phase))

        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        await gatherer.gather(mock_client, "task", progress_callback=callback)
        assert len(callback_calls) >= 1
        assert callback_calls[0][0] == 0.02
        assert callback_calls[0][1] == "agent_exploring"

    @pytest.mark.asyncio
    async def test_async_callback_is_awaited(self, tmp_path, mock_client):
        mock_client.generate_with_tools.return_value = _make_response(
            content="done", tool_calls=None
        )
        callback_calls = []

        async def async_callback(progress, phase):
            callback_calls.append((progress, phase))

        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        await gatherer.gather(mock_client, "task", progress_callback=async_callback)
        assert len(callback_calls) >= 1


class TestGatherPathTraversal:
    @pytest.mark.asyncio
    async def test_traversal_attempt_returns_error_continues(self, tmp_path, mock_client):
        # Agent tries to read outside source_dir
        tc = _make_tool_call("read_file", **{"path": "../../etc/passwd"})
        mock_client.generate_with_tools.side_effect = [
            _make_response(content="", tool_calls=[tc]),
            _make_response(content="Final context.", tool_calls=None),
        ]
        gatherer = AgentContextGatherer(
            config=_make_config(max_iterations=5), source_dir=str(tmp_path)
        )
        result = await gatherer.gather(mock_client, "task")
        # Tool returned an error string, but loop continued and got final content
        assert result == "Final context."
