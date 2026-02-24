# tests/unit/test_lm_studio_client.py
"""Unit tests for LMStudioClient."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fitz_graveyard.llm.lm_studio import LMStudioClient, _callable_to_openai_tool
from fitz_graveyard.llm.types import AgentMessage, AgentToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(**kwargs):
    """Create LMStudioClient with openai patched out."""
    with patch("fitz_graveyard.llm.lm_studio.AsyncOpenAI"):
        client = LMStudioClient(**kwargs)
    return client


def _make_completion(content=None, tool_calls=None, finish_reason="stop"):
    """Build a mock openai ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_openai_tool_call(call_id, name, arguments_dict):
    """Build a mock OpenAI ToolCall object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments_dict)
    return tc


# ---------------------------------------------------------------------------
# _callable_to_openai_tool
# ---------------------------------------------------------------------------

class TestCallableToOpenaiTool:
    def test_basic_function(self):
        def my_func(path: str, count: int) -> str:
            """List files in a directory."""
            ...

        schema = _callable_to_openai_tool(my_func)
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "my_func"
        assert fn["description"] == "List files in a directory."
        assert fn["parameters"]["properties"]["path"]["type"] == "string"
        assert fn["parameters"]["properties"]["count"]["type"] == "integer"
        assert "path" in fn["parameters"]["required"]
        assert "count" in fn["parameters"]["required"]

    def test_type_mapping(self):
        def fn(a: str, b: int, c: bool, d: float): ...

        schema = _callable_to_openai_tool(fn)
        props = schema["function"]["parameters"]["properties"]
        assert props["a"]["type"] == "string"
        assert props["b"]["type"] == "integer"
        assert props["c"]["type"] == "boolean"
        assert props["d"]["type"] == "number"

    def test_optional_param_not_required(self):
        def fn(required: str, optional: str = "default"): ...

        schema = _callable_to_openai_tool(fn)
        required = schema["function"]["parameters"]["required"]
        assert "required" in required
        assert "optional" not in required

    def test_no_docstring_uses_function_name(self):
        def fn_no_doc(x: str): ...

        schema = _callable_to_openai_tool(fn_no_doc)
        assert schema["function"]["description"] == "fn_no_doc"

    def test_unknown_annotation_defaults_to_string(self):
        def fn(x): ...  # no annotation

        schema = _callable_to_openai_tool(fn)
        assert schema["function"]["parameters"]["properties"]["x"]["type"] == "string"


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_returns_true_on_200(self):
        client = _make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("fitz_graveyard.llm.lm_studio.httpx.AsyncClient") as mock_http:
            mock_http.return_value.__aenter__ = AsyncMock(return_value=mock_http.return_value)
            mock_http.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            result = await client.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_connection_error(self):
        client = _make_client()

        with patch("fitz_graveyard.llm.lm_studio.httpx.AsyncClient") as mock_http:
            mock_http.return_value.__aenter__ = AsyncMock(return_value=mock_http.return_value)
            mock_http.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_http.return_value.get = AsyncMock(
                side_effect=Exception("connection refused")
            )
            result = await client.health_check()

        assert result is False


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

async def _async_iter(items):
    for item in items:
        yield item


class TestGenerate:
    @pytest.mark.asyncio
    async def test_accumulates_streamed_content(self):
        client = _make_client(model="test-model")

        chunks = []
        for text in ["Hello", ", ", "world"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = text
            chunks.append(chunk)

        client._client.chat.completions.create = AsyncMock(
            return_value=_async_iter(chunks)
        )

        result = await client.generate([{"role": "user", "content": "hi"}])
        assert result == "Hello, world"

    @pytest.mark.asyncio
    async def test_skips_empty_delta_content(self):
        client = _make_client()

        chunks = []
        for text in ["data", None, "more"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = text
            chunks.append(chunk)

        client._client.chat.completions.create = AsyncMock(
            return_value=_async_iter(chunks)
        )

        result = await client.generate([{"role": "user", "content": "hi"}])
        assert result == "datamore"


# ---------------------------------------------------------------------------
# generate_with_tools
# ---------------------------------------------------------------------------

class TestGenerateWithTools:
    @pytest.mark.asyncio
    async def test_returns_agent_message_no_tool_calls(self):
        client = _make_client(model="test-model")

        completion = _make_completion(content="Here is my answer.", tool_calls=None)
        client._client.chat.completions.create = AsyncMock(return_value=completion)

        def my_tool(path: str) -> str:
            """List files."""
            return ""

        msg = await client.generate_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[my_tool],
        )

        assert isinstance(msg, AgentMessage)
        assert msg.content == "Here is my answer."
        assert msg.tool_calls is None

    @pytest.mark.asyncio
    async def test_returns_agent_message_with_tool_calls(self):
        client = _make_client(model="test-model")

        tc = _make_openai_tool_call("call-1", "list_directory", {"path": "."})
        completion = _make_completion(content=None, tool_calls=[tc], finish_reason="tool_calls")
        client._client.chat.completions.create = AsyncMock(return_value=completion)

        def list_directory(path: str) -> str:
            """List directory contents."""
            return ""

        msg = await client.generate_with_tools(
            messages=[{"role": "user", "content": "explore"}],
            tools=[list_directory],
        )

        assert isinstance(msg, AgentMessage)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

        tc_result = msg.tool_calls[0]
        assert isinstance(tc_result, AgentToolCall)
        assert tc_result.id == "call-1"
        assert tc_result.name == "list_directory"
        assert tc_result.arguments == {"path": "."}

    @pytest.mark.asyncio
    async def test_assistant_dict_has_tool_calls_for_openai_format(self):
        client = _make_client(model="test-model")

        tc = _make_openai_tool_call("call-42", "read_file", {"path": "main.py"})
        completion = _make_completion(content=None, tool_calls=[tc], finish_reason="tool_calls")
        client._client.chat.completions.create = AsyncMock(return_value=completion)

        def read_file(path: str) -> str:
            """Read a file."""
            return ""

        msg = await client.generate_with_tools(
            messages=[{"role": "user", "content": "read"}],
            tools=[read_file],
        )

        assert "tool_calls" in msg.assistant_dict
        tc_dict = msg.assistant_dict["tool_calls"][0]
        assert tc_dict["id"] == "call-42"
        assert tc_dict["function"]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_converts_callables_to_openai_schema(self):
        client = _make_client(model="test-model")

        completion = _make_completion(content="done", tool_calls=None)
        client._client.chat.completions.create = AsyncMock(return_value=completion)

        def search_text(query: str, max_results: int = 10) -> str:
            """Search for text in files."""
            return ""

        await client.generate_with_tools(
            messages=[{"role": "user", "content": "search"}],
            tools=[search_text],
        )

        _, kwargs = client._client.chat.completions.create.call_args
        tools_arg = kwargs["tools"]
        assert len(tools_arg) == 1
        assert tools_arg[0]["function"]["name"] == "search_text"
        assert tools_arg[0]["function"]["description"] == "Search for text in files."


# ---------------------------------------------------------------------------
# tool_result_message
# ---------------------------------------------------------------------------

class TestToolResultMessage:
    def test_includes_tool_call_id(self):
        client = _make_client()
        msg = client.tool_result_message("call-abc", "some result")
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call-abc"
        assert msg["content"] == "some result"

    def test_empty_id_still_works(self):
        client = _make_client()
        msg = client.tool_result_message("", "result")
        assert msg["tool_call_id"] == ""
        assert msg["content"] == "result"
