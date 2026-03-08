# fitz_graveyard/llm/lm_studio.py
"""LM Studio client using OpenAI-compatible API."""

import asyncio
import inspect
import json
import logging
import shutil
import subprocess
import time
from typing import TYPE_CHECKING

import httpx

from .retry import lm_studio_retry
from .types import AgentMessage, AgentToolCall

if TYPE_CHECKING:
    from .memory import MemoryMonitor

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Maps Python type annotations → JSON Schema types
_TYPE_MAP = {
    str: "string",
    int: "integer",
    bool: "boolean",
    float: "number",
}


def _callable_to_openai_tool(fn) -> dict:
    """
    Convert a Python callable to an OpenAI tool schema dict.

    Uses inspect.signature() for parameter info, type annotations for types,
    and the first line of the docstring as the description.

    Args:
        fn: Callable with type annotations and docstring

    Returns:
        OpenAI tool schema dict
    """
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""
    description = doc.splitlines()[0] if doc else fn.__name__

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        annotation = param.annotation
        json_type = _TYPE_MAP.get(annotation, "string")
        properties[name] = {"type": json_type}

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class LMStudioClient:
    """
    Async LM Studio client using the OpenAI-compatible API.

    LM Studio exposes an OpenAI-compatible endpoint at http://localhost:1234/v1.
    Requires: pip install fitz-graveyard[lm-studio]
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "local-model",
        fallback_model: str | None = None,
        timeout: int = 300,
        context_length: int = 32768,
    ):
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package required for LM Studio support. "
                "Install with: pip install fitz-graveyard[lm-studio]"
            )

        self.base_url = base_url
        self.model = model
        self.fallback_model = fallback_model
        self._timeout = timeout
        self._context_length = context_length
        self._client = AsyncOpenAI(base_url=base_url, api_key="lm-studio", timeout=timeout)
        self._call_metrics: list[dict] = []

    @property
    def context_size(self) -> int:
        """Configured context window size in tokens."""
        return self._context_length

    @property
    def fast_model(self) -> str:
        """Model name for fast/screening tasks (same as primary for LM Studio)."""
        return self.model

    @property
    def mid_model(self) -> str:
        """Model name for mid-tier tasks (same as primary for LM Studio)."""
        return self.model

    @property
    def smart_model(self) -> str:
        """Model name for smart/reasoning tasks (same as primary for LM Studio)."""
        return self.model

    async def ensure_model(
        self, model_name: str, context_size: int | None = None,
    ) -> None:
        """Ensure the requested model is loaded in LM Studio."""
        if await self.is_model_loaded():
            return
        await self._load_model_via_cli()

    async def health_check(self) -> bool:
        """Check LM Studio is reachable and the configured model is loaded.

        If LM Studio is running but no model is loaded, auto-loads the
        configured model via the ``lms`` CLI.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as http:
                response = await http.get(f"{self.base_url}/models")
            if response.status_code != 200:
                return False
        except Exception as e:
            logger.error(f"LM Studio health check failed: {e}")
            return False

        if await self.is_model_loaded():
            return True

        logger.info(f"No model loaded in LM Studio, auto-loading {self.model}")
        return await self._load_model_via_cli()

    async def is_model_loaded(self) -> bool:
        """Check if any model is currently loaded (not just available)."""
        lms = shutil.which("lms")
        if not lms:
            return True  # Can't check, assume loaded

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [lms, "ps"],
                capture_output=True, text=True, timeout=10,
                encoding="utf-8", errors="replace",
            )
            # lms ps writes to stderr, not stdout
            output = result.stdout + result.stderr
            if "No models" in output:
                return False
            return True
        except Exception:
            return True  # Can't check, assume loaded

    async def _load_model_via_cli(self) -> bool:
        """Load the configured model via ``lms load``."""
        lms = shutil.which("lms")
        if not lms:
            logger.warning(
                "lms CLI not found — cannot auto-load model. "
                "Load it manually in LM Studio."
            )
            return False

        logger.info(
            f"Running: lms load {self.model} -y -c {self._context_length} --parallel 1"
        )
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    lms, "load", self.model, "-y",
                    "-c", str(self._context_length),
                    "--parallel", "1",
                ],
                capture_output=True, text=True, timeout=300,
                encoding="utf-8", errors="replace",
            )
            if result.returncode == 0:
                logger.info(f"Model {self.model} loaded successfully")
                return True
            logger.error(
                f"lms load failed (code {result.returncode}): "
                f"{result.stderr[:300]}"
            )
            return False
        except subprocess.TimeoutExpired:
            logger.error("lms load timed out after 300s")
            return False
        except Exception as e:
            logger.error(f"lms load failed: {e}")
            return False

    async def unload_model(self) -> bool:
        """Unload the current model via ``lms unload`` to free VRAM."""
        lms = shutil.which("lms")
        if not lms:
            logger.warning("lms CLI not found in PATH — cannot unload model")
            return False

        logger.info(f"Running: {lms} unload --all")
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [lms, "unload", "--all"],
                capture_output=True, text=True, timeout=30,
                encoding="utf-8", errors="replace",
            )
            output = (result.stdout + result.stderr).strip()
            if result.returncode == 0:
                logger.info(f"Model unloaded successfully: {output}")
                return True
            logger.warning(
                f"lms unload failed (code {result.returncode}): {output}"
            )
            return False
        except Exception as e:
            logger.warning(f"lms unload failed: {e}")
            return False

    async def reload_model(self) -> bool:
        """Reload the model after unloading."""
        return await self._load_model_via_cli()

    @lm_studio_retry
    async def generate(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Generate a streaming response from LM Studio.

        Args:
            messages: Chat messages in format [{"role": "user", "content": "..."}]
            model:    Model override (defaults to self.model)
            temperature: Sampling temperature (0.0 = deterministic). None = server default.

        Returns:
            Full accumulated response text.
        """
        model = model or self.model
        logger.info(f"LMStudio.generate: model={model}, messages={len(messages)}")

        t0 = time.monotonic()
        accumulated = []
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                accumulated.append(delta.content)

        result = "".join(accumulated)
        elapsed = time.monotonic() - t0
        est_tokens = len(result) / 4
        tok_s = est_tokens / elapsed if elapsed > 0 else 0
        self._call_metrics.append({"elapsed_s": elapsed, "output_chars": len(result), "model": model})
        logger.info(
            f"LMStudio.generate: {len(result)} chars in {elapsed:.1f}s "
            f"(~{tok_s:.1f} tok/s)"
        )
        return result

    async def generate_with_fallback(self, messages: list[dict]) -> tuple[str, str]:
        """
        Generate without OOM handling (LM Studio manages its own memory).

        Args:
            messages: Chat messages

        Returns:
            (response_text, model_used)
        """
        result = await self.generate(messages)
        return result, self.model

    def drain_call_metrics(self) -> list[dict]:
        """Return and clear accumulated call metrics from generate() calls."""
        metrics = self._call_metrics
        self._call_metrics = []
        return metrics

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list,
        model: str | None = None,
    ) -> AgentMessage:
        """
        Single chat call with tool definitions, returns normalized AgentMessage.

        Args:
            messages: Chat messages list
            tools:    Tool callables (converted to OpenAI schema automatically)
            model:    Model override (defaults to self.model)

        Returns:
            AgentMessage with .tool_calls or .content
        """
        model = model or self.model
        openai_tools = [_callable_to_openai_tool(fn) for fn in tools]
        logger.info(
            f"LMStudio.generate_with_tools: model={model}, "
            f"messages={len(messages)}, tools={len(openai_tools)}"
        )

        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            stream=False,
        )

        choice = response.choices[0]
        msg = choice.message
        logger.info(
            f"LMStudio.generate_with_tools: finish_reason={choice.finish_reason}, "
            f"tool_calls={bool(msg.tool_calls)}"
        )

        tool_calls = None
        assistant_tool_calls = None
        if msg.tool_calls:
            tool_calls = []
            assistant_tool_calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    AgentToolCall(
                        id=tc.id or "",
                        name=tc.function.name,
                        arguments=args,
                    )
                )
                # Preserve original OpenAI format for assistant_dict
                assistant_tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        assistant_dict: dict = {"role": "assistant", "content": msg.content}
        if assistant_tool_calls:
            assistant_dict["tool_calls"] = assistant_tool_calls

        return AgentMessage(
            content=msg.content,
            tool_calls=tool_calls,
            assistant_dict=assistant_dict,
        )

    async def generate_with_monitoring(
        self, messages: list[dict], monitor: "MemoryMonitor"
    ) -> tuple[str, str]:
        """
        Generate with memory monitoring running in parallel.

        Args:
            messages: Chat messages
            monitor:  MemoryMonitor instance

        Returns:
            (response_text, model_used)
        """
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        generation_task = asyncio.create_task(self.generate_with_fallback(messages))

        done, pending = await asyncio.wait(
            {monitor_task, generation_task}, return_when=asyncio.FIRST_COMPLETED
        )

        if monitor_task in done:
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

            result = await generation_task
            return result
        else:
            monitor.stop()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            return generation_task.result()

    def tool_result_message(self, tool_call_id: str, content: str) -> dict:
        """
        Build a tool result message dict for the OpenAI messages format.

        Args:
            tool_call_id: Tool call id from the assistant message
            content:      Tool result text

        Returns:
            Message dict with role="tool" and tool_call_id
        """
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}
