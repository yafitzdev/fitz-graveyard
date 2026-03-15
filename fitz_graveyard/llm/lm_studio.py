# fitz_graveyard/llm/lm_studio.py
"""LM Studio client using OpenAI-compatible API."""

import asyncio
import inspect
import json
import logging
import re
import shutil
import subprocess
import time
from typing import TYPE_CHECKING

import httpx

from .retry import lm_studio_retry
from .types import AgentMessage, AgentToolCall

if TYPE_CHECKING:
    from .gpu_monitor import GPUTemperatureGuard
    from .memory import MemoryMonitor

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Strip <think>...</think> blocks that some models emit even when thinking
# is disabled.  Applied once after accumulation so all downstream parsers
# receive clean text.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks from model output."""
    text = _THINK_RE.sub("", text)
    # Handle unclosed <think> (generation ended mid-thought)
    if "<think>" in text:
        text = text.split("</think>")[-1].lstrip() if "</think>" in text else text.split("<think>")[0].rstrip()
    return text


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
        gpu_guard: "GPUTemperatureGuard | None" = None,
        fast_model: str | None = None,
        smart_model: str | None = None,
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
        self._gpu_guard = gpu_guard
        self._fast_model = fast_model
        self._smart_model = smart_model
        self._client = AsyncOpenAI(base_url=base_url, api_key="lm-studio", timeout=timeout)
        self._call_metrics: list[dict] = []

    @property
    def context_size(self) -> int:
        """Configured context window size in tokens."""
        return self._context_length

    @property
    def fast_model(self) -> str:
        """Model name for fast/screening tasks."""
        return self._fast_model or self.model

    @property
    def mid_model(self) -> str:
        """Model name for mid-tier tasks."""
        return self.model

    @property
    def smart_model(self) -> str:
        """Model name for reasoning tasks."""
        return self._smart_model or self.model

    async def ensure_model(
        self, model_name: str, context_size: int | None = None,
    ) -> None:
        """Ensure the requested model is loaded in LM Studio."""
        if await self.is_model_loaded():
            return
        await self._load_model_via_cli()

    # Minimum context window in tokens.  Derived from the pipeline's largest
    # prompt: arch+design reasoning can send up to _REASONING_KRAG_BUDGET_CHARS
    # (200K chars ≈ 50K tokens) of codebase context plus prompt template
    # overhead (~2K tokens) plus max_tokens output (16K).  Stages that exceed
    # the model's context will crash mid-pipeline with cryptic CUDA errors.
    _MIN_CONTEXT_TOKENS = 32_768

    async def health_check(self) -> bool:
        """Check LM Studio is reachable and load the configured model.

        The config is the single source of truth for model name and context
        length.  Any previously loaded model is unloaded and the configured
        model is loaded with the configured context_length via ``lms load``.

        Raises RuntimeError if the configured context window is below the
        minimum required by the planning pipeline.
        """
        # Context window preflight — fail fast instead of crashing mid-pipeline
        if self._context_length < self._MIN_CONTEXT_TOKENS:
            raise RuntimeError(
                f"Context window too small: {self._context_length} tokens "
                f"(minimum {self._MIN_CONTEXT_TOKENS}). "
                f"Increase context_length in config."
            )

        # Check LM Studio is reachable
        try:
            async with httpx.AsyncClient(timeout=5.0) as http:
                response = await http.get(f"{self.base_url}/models")
            if response.status_code != 200:
                return False
        except Exception as e:
            logger.error(f"LM Studio health check failed: {e}")
            return False

        # Load the first model needed.  If smart_model differs from model,
        # the agent runs first on smart_model, so load that.  Otherwise
        # load the planning model.  The orchestrator handles switching
        # between agent and planning stages.
        loaded = await self.get_loaded_model()
        first_model = (
            self.smart_model
            if self._smart_model and self._smart_model != self.model
            else self.model
        )
        if loaded == first_model:
            return True
        if loaded:
            # Wrong model loaded — switch
            logger.info(f"Health check: switching {loaded} -> {first_model}")
            return await self.switch_model(first_model)
        logger.info(f"No model loaded, auto-loading {first_model} (ctx={self._context_length})")
        return await self._load_model_via_cli(first_model)

    async def get_loaded_model(self) -> str | None:
        """Return the identifier of the currently loaded model, or None."""
        lms = shutil.which("lms")
        if not lms:
            return None

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [lms, "ps"],
                capture_output=True, text=True, timeout=10,
                encoding="utf-8", errors="replace",
            )
            output = result.stdout + result.stderr
            if "No models" in output:
                return None
            # Parse table: first non-header line, first column is identifier
            for line in output.splitlines():
                line = line.strip()
                if not line or line.startswith("IDENTIFIER") or line.startswith("-"):
                    continue
                return line.split()[0]
            return None
        except Exception:
            return None

    async def is_model_loaded(self) -> bool:
        """Check if any model is currently loaded (not just available)."""
        return await self.get_loaded_model() is not None

    async def _load_model_via_cli(self, model_name: str | None = None) -> bool:
        """Load a model via ``lms load``.

        Args:
            model_name: Model to load. Defaults to self.model.
        """
        model_name = model_name or self.model
        lms = shutil.which("lms")
        if not lms:
            logger.warning(
                "lms CLI not found — cannot auto-load model. "
                "Load it manually in LM Studio."
            )
            return False

        logger.info(
            f"Running: lms load {model_name} -y -c {self._context_length} --parallel 1"
        )
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    lms, "load", model_name, "-y",
                    "-c", str(self._context_length),
                    "--parallel", "1",
                ],
                capture_output=True, text=True, timeout=300,
                encoding="utf-8", errors="replace",
            )
            if result.returncode == 0:
                logger.info(f"Model {model_name} loaded successfully")
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

    async def switch_model(self, model_name: str) -> bool:
        """Unload current model and load the specified one.

        Skips the switch if the target model is already loaded (avoids
        CUDA context destruction on WDDM consumer GPUs).

        Args:
            model_name: Model to switch to.

        Returns:
            True if the target model is loaded (was already or newly loaded).
        """
        loaded = await self.get_loaded_model()
        if loaded and loaded == model_name:
            logger.info(f"Model {model_name} already loaded, skipping switch")
            return True
        logger.info(f"Switching model: {loaded} -> {model_name}")
        await self.unload_model()
        return await self._load_model_via_cli(model_name)

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
        max_tokens: int = 16384,
    ) -> str:
        """
        Generate a streaming response from LM Studio.

        Args:
            messages: Chat messages in format [{"role": "user", "content": "..."}]
            model:    Model override (defaults to self.model)
            temperature: Sampling temperature (0.0 = deterministic). None = server default.
            max_tokens:  Hard cap on output tokens. Prevents infinite generation.

        Returns:
            Full accumulated response text.
        """
        if self._gpu_guard:
            await self._gpu_guard.preflight()

        model = model or self.model
        logger.info(f"LMStudio.generate: model={model}, messages={len(messages)}")

        t0 = time.monotonic()
        accumulated = []
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
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
            if self._gpu_guard:
                await self._gpu_guard.maybe_throttle()

        result = _strip_thinking("".join(accumulated))
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
        if self._gpu_guard:
            await self._gpu_guard.preflight()

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
