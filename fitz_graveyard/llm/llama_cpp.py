# fitz_graveyard/llm/llama_cpp.py
"""
llama.cpp client with subprocess management and model tier support.

Manages a llama-server process as a subprocess. Restarts the server
when switching between fast/smart model tiers (since context_size and
gpu_layers are server-level settings).
"""

import asyncio
import inspect
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from .retry import llama_cpp_retry
from .types import AgentMessage, AgentToolCall

if TYPE_CHECKING:
    from fitz_graveyard.config.schema import LlamaCppModelConfig

    from .memory import MemoryMonitor

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Maps Python type annotations → JSON Schema types (same as lm_studio.py)
_TYPE_MAP = {
    str: "string",
    int: "integer",
    bool: "boolean",
    float: "number",
}


def _callable_to_openai_tool(fn) -> dict:
    """Convert a Python callable to an OpenAI tool schema dict."""
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


class LlamaCppClient:
    """
    Async llama.cpp client that manages llama-server as a subprocess.

    Restarts the server when switching between fast/smart tiers, since
    context_size and gpu_layers are server-level (not per-request).

    Requires: pip install fitz-graveyard[lm-studio]  (for openai SDK)
    """

    def __init__(
        self,
        server_path: str,
        models_dir: str,
        fast_model: "LlamaCppModelConfig",
        mid_model: "LlamaCppModelConfig | None" = None,
        smart_model: "LlamaCppModelConfig | None" = None,
        port: int = 8012,
        timeout: int = 300,
        startup_timeout: int = 120,
    ):
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package required for llama.cpp support. "
                "Install with: pip install fitz-graveyard[lm-studio]"
            )

        self._server_path = server_path
        self._models_dir = models_dir
        self._fast_model = fast_model
        self._mid_model = mid_model or fast_model
        self._smart_model = smart_model or fast_model
        self._port = port
        self._timeout = timeout
        self._startup_timeout = startup_timeout

        # Public attributes for interface parity with OllamaClient/LMStudioClient
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.model = fast_model.path
        self.fallback_model = smart_model.path if smart_model else None

        self._process: subprocess.Popen | None = None
        self._client: AsyncOpenAI | None = None
        self._active_tier: str | None = None
        self._call_metrics: list[dict] = []

    # ------------------------------------------------------------------
    # Model tier properties
    # ------------------------------------------------------------------

    @property
    def fast_model(self) -> str:
        """Model name for fast/screening tasks."""
        return self._fast_model.path

    @property
    def mid_model(self) -> str:
        """Model name for mid-tier/summarization tasks."""
        return self._mid_model.path

    @property
    def smart_model(self) -> str:
        """Model name for smart/reasoning tasks."""
        return self._smart_model.path

    @property
    def active_model(self) -> str:
        """Path of the model currently loaded in llama-server."""
        if self._active_tier == "smart":
            return self._smart_model.path
        if self._active_tier == "mid":
            return self._mid_model.path
        return self._fast_model.path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, tier: str = "fast") -> None:
        """Start llama-server subprocess for the given tier.

        Args:
            tier: "fast", "mid", or "smart" — determines which model config to use.
        """
        tier_map = {"fast": self._fast_model, "mid": self._mid_model, "smart": self._smart_model}
        model_cfg = tier_map.get(tier, self._fast_model)
        model_path = str(Path(self._models_dir) / model_cfg.path)

        cmd = [
            self._server_path,
            "--host", "127.0.0.1",
            "--port", str(self._port),
            "-m", model_path,
            "-c", str(model_cfg.context_size),
            "-ngl", str(model_cfg.gpu_layers),
        ]
        if model_cfg.flash_attention:
            cmd.extend(["--flash-attn", "on"])
        if model_cfg.cache_type_k:
            cmd.extend(["--cache-type-k", model_cfg.cache_type_k])
        if model_cfg.cache_type_v:
            cmd.extend(["--cache-type-v", model_cfg.cache_type_v])

        logger.info(
            f"Starting llama-server ({tier}): {' '.join(cmd)}"
        )

        # Use DEVNULL for stdin, pipe stderr for error reporting
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        await self._wait_for_ready()

        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="llama-cpp",
            timeout=self._timeout,
        )
        self._active_tier = tier
        self.model = model_cfg.path

        logger.info(
            f"llama-server ready ({tier}): {model_cfg.path}"
        )

    async def stop(self) -> None:
        """Stop the llama-server subprocess."""
        if self._process is None:
            return

        logger.info("Stopping llama-server...")
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server did not stop, killing")
            self._process.kill()
            self._process.wait(timeout=5)

        self._process = None
        self._active_tier = None
        self._client = None
        logger.info("llama-server stopped")

    async def ensure_model(self, model_name: str) -> None:
        """Switch to the tier for the given model name.

        Public API for callers that need to pre-load a specific tier
        (e.g. orchestrator switching from smart back to mid after agent
        gathering).
        """
        await self._ensure_tier(model_name)

    async def _ensure_tier(self, model_name: str | None) -> None:
        """Restart server if the requested model needs a different tier.

        Args:
            model_name: Model name from generate() call.
                        Resolved to tier by matching against model paths.
        """
        if model_name is None:
            # No override — use whatever is currently loaded
            if self._active_tier is None:
                await self.start("fast")
            return

        # Resolve tier from model name. Check smart first, then mid,
        # then fall back to fast. Skip mid if it has the same path as fast.
        if model_name == self._smart_model.path:
            needed = "smart"
        elif (model_name == self._mid_model.path
              and self._mid_model.path != self._fast_model.path):
            needed = "mid"
        else:
            needed = "fast"

        if self._active_tier == needed:
            return

        logger.info(
            f"Switching tier: {self._active_tier} → {needed} "
            f"(model={model_name})"
        )
        await self.stop()
        await self.start(needed)

    async def _wait_for_ready(self) -> None:
        """Poll /health until server responds 200 or timeout."""
        url = f"http://127.0.0.1:{self._port}/health"
        deadline = time.monotonic() + self._startup_timeout

        async with httpx.AsyncClient(timeout=2.0) as http:
            while time.monotonic() < deadline:
                # Check if process died
                if self._process and self._process.poll() is not None:
                    stderr = ""
                    if self._process.stderr:
                        stderr = self._process.stderr.read().decode(
                            errors="replace"
                        )
                    raise RuntimeError(
                        f"llama-server exited with code "
                        f"{self._process.returncode}: {stderr[:500]}"
                    )
                try:
                    resp = await http.get(url)
                    if resp.status_code == 200:
                        return
                except httpx.ConnectError:
                    pass
                await asyncio.sleep(1.0)

        raise TimeoutError(
            f"llama-server did not become ready within "
            f"{self._startup_timeout}s"
        )

    async def _ensure_alive(self) -> None:
        """Restart the server if it has crashed, raise if not started."""
        if self._process is None:
            raise RuntimeError("llama-server not started")
        if self._process.poll() is not None:
            stderr = ""
            if self._process.stderr:
                stderr = self._process.stderr.read().decode(errors="replace")
            code = self._process.returncode
            logger.warning(
                f"llama-server crashed (code {code}), restarting: "
                f"{stderr[:500]}"
            )
            self._process = None
            self._client = None
            tier = self._active_tier or "fast"
            self._active_tier = None
            await self.start(tier)

    # ------------------------------------------------------------------
    # Interface methods (match OllamaClient / LMStudioClient)
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Check if llama-server is running and healthy."""
        try:
            await self._ensure_alive()
            async with httpx.AsyncClient(timeout=5.0) as http:
                resp = await http.get(
                    f"http://127.0.0.1:{self._port}/health"
                )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"llama-cpp health check failed: {e}")
            return False

    @llama_cpp_retry
    async def generate(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a streaming response. Switches tier if model differs.

        Args:
            messages:    Chat messages in OpenAI format.
            model:       Model name (triggers tier switch if needed).
            temperature: Sampling temperature. None = server default.

        Returns:
            Full accumulated response text.
        """
        await self._ensure_tier(model)
        await self._ensure_alive()

        effective_model = model or self.model
        logger.info(
            f"LlamaCpp.generate: model={effective_model}, "
            f"messages={len(messages)}"
        )

        t0 = time.monotonic()
        accumulated = []
        kwargs: dict = {
            "model": effective_model,
            "messages": messages,
            "stream": True,
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
        self._call_metrics.append({
            "elapsed_s": elapsed,
            "output_chars": len(result),
            "model": effective_model,
        })
        logger.info(
            f"LlamaCpp.generate: {len(result)} chars in {elapsed:.1f}s"
        )
        return result

    async def generate_with_fallback(
        self, messages: list[dict],
    ) -> tuple[str, str]:
        """Generate using the smart model (no OOM fallback needed).

        Returns:
            (response_text, model_used)
        """
        result = await self.generate(messages, model=self.smart_model)
        return result, self.model

    def drain_call_metrics(self) -> list[dict]:
        """Return and clear accumulated call metrics."""
        metrics = self._call_metrics
        self._call_metrics = []
        return metrics

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list,
        model: str | None = None,
    ) -> AgentMessage:
        """Single chat call with tool definitions.

        Args:
            messages: Chat messages list.
            tools:    Tool callables.
            model:    Model override.

        Returns:
            AgentMessage with .tool_calls or .content.
        """
        await self._ensure_tier(model)
        await self._ensure_alive()

        effective_model = model or self.model
        openai_tools = [_callable_to_openai_tool(fn) for fn in tools]
        logger.info(
            f"LlamaCpp.generate_with_tools: model={effective_model}, "
            f"messages={len(messages)}, tools={len(openai_tools)}"
        )

        response = await self._client.chat.completions.create(
            model=effective_model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            stream=False,
        )

        choice = response.choices[0]
        msg = choice.message

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
                assistant_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        assistant_dict: dict = {"role": "assistant", "content": msg.content}
        if assistant_tool_calls:
            assistant_dict["tool_calls"] = assistant_tool_calls

        return AgentMessage(
            content=msg.content,
            tool_calls=tool_calls,
            assistant_dict=assistant_dict,
        )

    async def generate_with_monitoring(
        self, messages: list[dict], monitor: "MemoryMonitor",
    ) -> tuple[str, str]:
        """Generate with memory monitoring running in parallel.

        Args:
            messages: Chat messages.
            monitor:  MemoryMonitor instance.

        Returns:
            (response_text, model_used)
        """
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        generation_task = asyncio.create_task(
            self.generate_with_fallback(messages)
        )

        done, pending = await asyncio.wait(
            {monitor_task, generation_task},
            return_when=asyncio.FIRST_COMPLETED,
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
                    f"Memory threshold exceeded "
                    f"({monitor.threshold_percent}%) during generation"
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
        """Build a tool result message dict for the OpenAI messages format."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
