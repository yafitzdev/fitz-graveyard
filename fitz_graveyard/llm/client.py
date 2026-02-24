# fitz_graveyard/llm/client.py
"""Ollama client with health checks, streaming generation, and fallback support."""

import asyncio
import logging
from typing import TYPE_CHECKING

import httpx
import ollama
from ollama import AsyncClient, ResponseError

from .retry import ollama_retry
from .types import AgentMessage, AgentToolCall

if TYPE_CHECKING:
    from .memory import MemoryMonitor

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Async Ollama client with health checks, streaming, and OOM fallback.

    Handles:
    - Health checks (server + model availability)
    - Streaming generation with content accumulation
    - Automatic fallback to smaller model on OOM errors
    - Memory monitoring integration
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        fallback_model: str | None = None,
        timeout: int = 300,
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API base URL (e.g., "http://localhost:11434")
            model: Primary model name (e.g., "qwen2.5-coder-next:80b-instruct")
            fallback_model: Fallback model on OOM (e.g., "qwen2.5-coder-next:32b-instruct")
            timeout: Request timeout in seconds (generous for model loading)
        """
        self.base_url = base_url
        self.model = model
        self.fallback_model = fallback_model
        self.client = AsyncClient(host=base_url, timeout=httpx.Timeout(timeout))

    async def health_check(self) -> bool:
        """
        Check Ollama server health and model availability.

        Returns:
            True if server is reachable and model is available (or can be pulled).
            False if server is down or unreachable.
        """
        try:
            # Check server availability
            models_response = await self.client.list()
            available_models = [m["name"] for m in models_response.get("models", [])]

            # Check if our model is available (exact match or with :latest tag)
            model_base = self.model.split(":")[0]
            model_available = any(
                model_base in m or self.model == m for m in available_models
            )

            if not model_available:
                logger.warning(
                    f"Model {self.model} not found in available models. "
                    f"It can be pulled on demand during generation."
                )

            # Return True even if model not found - Ollama can pull on demand
            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    @ollama_retry
    async def generate(self, messages: list[dict], model: str | None = None) -> str:
        """
        Generate a response from Ollama with streaming.

        Args:
            messages: Chat messages in format [{"role": "user", "content": "..."}]
            model: Model to use (defaults to self.model)

        Returns:
            Full accumulated response text.

        Raises:
            ResponseError: On API errors (retry decorator handles transient errors)
        """
        model = model or self.model
        logger.info(f"Generating with model={model}, messages={len(messages)}")

        # Stream response and accumulate
        accumulated = []
        async for chunk in await self.client.chat(
            model=model, messages=messages, stream=True
        ):
            if content := chunk.get("message", {}).get("content"):
                accumulated.append(content)

        result = "".join(accumulated)
        logger.info(f"Generated {len(result)} chars")
        return result

    async def generate_with_fallback(self, messages: list[dict]) -> tuple[str, str]:
        """
        Generate with automatic fallback to smaller model on OOM.

        Args:
            messages: Chat messages

        Returns:
            (response_text, model_used): Response and which model generated it

        Raises:
            ResponseError: On non-OOM errors or if fallback also fails
        """
        try:
            result = await self.generate(messages, model=self.model)
            return result, self.model

        except ResponseError as e:
            # Check if this is an OOM error
            if e.status_code == 500 and "requires more system memory" in str(e).lower():
                if self.fallback_model is None:
                    logger.error("OOM error but no fallback model configured")
                    raise

                logger.warning(
                    f"OOM error on {self.model}, falling back to {self.fallback_model}"
                )
                result = await self.generate(messages, model=self.fallback_model)
                return result, self.fallback_model

            # Non-OOM error - re-raise
            raise

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list,
        model: str | None = None,
    ) -> AgentMessage:
        """
        Single non-streaming chat call with tool definitions.

        Does NOT stream — tool call responses are atomic in Ollama.
        Returns a normalized AgentMessage for provider-agnostic consumption.

        Args:
            messages: Chat messages list
            tools:    Tool callables (ollama auto-generates schemas from docstrings)
            model:    Model override (defaults to self.model)

        Returns:
            AgentMessage with .tool_calls or .content
        """
        model = model or self.model
        logger.info(
            f"generate_with_tools: model={model}, messages={len(messages)}, tools={len(tools)}"
        )
        response = await self.client.chat(
            model=model,
            messages=messages,
            tools=tools,
            stream=False,
        )
        logger.info(
            f"generate_with_tools: done_reason={response.done_reason}, "
            f"tool_calls={bool(response.message.tool_calls)}"
        )

        msg = response.message
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                AgentToolCall(
                    id="",
                    name=tc.function.name,
                    arguments=tc.function.arguments or {},
                )
                for tc in msg.tool_calls
            ]

        return AgentMessage(
            content=msg.content,
            tool_calls=tool_calls,
            assistant_dict=msg,  # Ollama Message objects work directly in messages list
        )

    def tool_result_message(self, tool_call_id: str, content: str) -> dict:
        """
        Build a tool result message dict for appending to the messages list.

        Ollama ignores tool_call_id — only content matters.

        Args:
            tool_call_id: Tool call id (ignored by Ollama)
            content:      Tool result text

        Returns:
            Message dict with role="tool"
        """
        return {"role": "tool", "content": content}

    async def generate_with_monitoring(
        self, messages: list[dict], monitor: "MemoryMonitor"
    ) -> tuple[str, str]:
        """
        Generate with memory monitoring running in parallel.

        If monitor detects threshold violation before generation completes,
        cancels generation and raises MemoryError.

        Args:
            messages: Chat messages
            monitor: MemoryMonitor instance

        Returns:
            (response_text, model_used): Response and which model generated it

        Raises:
            MemoryError: If RAM threshold exceeded during generation
            ResponseError: On API errors
        """
        # Start monitor and generation tasks
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        generation_task = asyncio.create_task(self.generate_with_fallback(messages))

        # Wait for first to complete
        done, pending = await asyncio.wait(
            {monitor_task, generation_task}, return_when=asyncio.FIRST_COMPLETED
        )

        # Check which completed first
        if monitor_task in done:
            # Monitor finished first (threshold exceeded)
            generation_task.cancel()
            try:
                await generation_task
            except asyncio.CancelledError:
                pass

            threshold_exceeded = monitor_task.result()
            if threshold_exceeded:
                raise MemoryError(
                    f"Memory threshold exceeded ({monitor.threshold_percent}%) during generation"
                )

            # Monitor stopped cleanly (shouldn't happen, but handle it)
            result = await generation_task
            return result

        else:
            # Generation finished first
            monitor.stop()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            result = generation_task.result()
            return result
