# fitz_graveyard/planning/agent/gatherer.py
"""
AgentContextGatherer — runs a tool-calling agent loop to explore the codebase
and produce a structured markdown context document.

Called once before all planning stages. Stores result in
prior_outputs['_gathered_context'] for all stages to consume.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fitz_graveyard.planning.agent.tools import _make_tools
from fitz_graveyard.planning.prompts import load_prompt

if TYPE_CHECKING:
    from fitz_graveyard.config.schema import AgentConfig

logger = logging.getLogger(__name__)


class AgentContextGatherer:
    """
    Runs a Qwen tool-calling loop to gather codebase context.

    The agent:
    1. Receives a system prompt with the job description
    2. Iteratively calls list_directory / read_file / search_text / find_files
    3. Terminates when it produces a response with no tool_calls
    4. Returns the final text as the context document
    """

    def __init__(self, config: "AgentConfig", source_dir: str) -> None:
        """
        Initialize AgentContextGatherer.

        Args:
            config:     AgentConfig (max_iterations, agent_model, max_file_bytes)
            source_dir: Absolute path to the project root for tool calls
        """
        self._config = config
        self._source_dir = source_dir

    async def gather(
        self,
        client: Any,
        job_description: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str:
        """
        Run the tool-calling agent loop and return context markdown.

        Iterates up to config.max_iterations times. On each iteration:
        - If the response has tool_calls: execute tools, append results, loop
        - If the response has no tool_calls: return response text as context

        If max_iterations is reached without a final response, returns whatever
        partial text the last response contained (graceful degradation).

        Args:
            client:            OllamaClient for LLM calls
            job_description:   The user's planning request (injected into prompt)
            progress_callback: Optional (progress: float, phase: str) callback

        Returns:
            Markdown string with codebase context, or "" on failure/disabled
        """
        if not self._config.enabled:
            logger.info("Agent context gathering disabled by config")
            return ""

        tools = _make_tools(
            source_dir=self._source_dir,
            max_file_bytes=self._config.max_file_bytes,
        )
        tool_map = {fn.__name__: fn for fn in tools}
        model = self._config.agent_model or client.model

        system_prompt = load_prompt("agent_context").format(
            job_description=job_description
        )
        messages: list = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Please explore the codebase and produce the context document. "
                    "Start with list_directory('.')."
                ),
            },
        ]

        if progress_callback:
            result_or_coro = progress_callback(0.02, "agent_exploring")
            if hasattr(result_or_coro, "__await__"):
                await result_or_coro

        last_content = ""

        for iteration in range(self._config.max_iterations):
            logger.info(
                f"AgentContextGatherer: iteration {iteration + 1}/"
                f"{self._config.max_iterations}"
            )

            try:
                response = await client.generate_with_tools(
                    messages=messages,
                    tools=tools,
                    model=model,
                )
            except Exception as e:
                logger.warning(
                    f"AgentContextGatherer: LLM call failed at iteration "
                    f"{iteration + 1}: {e}"
                )
                break

            msg = response

            if not msg.tool_calls:
                last_content = msg.content or ""
                logger.info(
                    f"AgentContextGatherer: agent finished after {iteration + 1} "
                    f"iterations, {len(last_content)} chars of context"
                )
                break

            # Append assistant message using provider-specific dict
            messages.append(msg.assistant_dict)

            # Execute each tool and append results
            for tool_call in msg.tool_calls:
                fn_name = tool_call.name
                fn_args = tool_call.arguments or {}
                tool_fn = tool_map.get(fn_name)

                if tool_fn is None:
                    result_text = f"Error: unknown tool '{fn_name}'"
                    logger.warning(f"AgentContextGatherer: unknown tool '{fn_name}'")
                else:
                    try:
                        result_text = tool_fn(**fn_args)
                        logger.debug(
                            f"AgentContextGatherer: {fn_name}({fn_args}) → "
                            f"{len(result_text)} chars"
                        )
                    except Exception as e:
                        result_text = f"Error executing {fn_name}: {e}"
                        logger.warning(
                            f"AgentContextGatherer: tool '{fn_name}' raised: {e}"
                        )

                messages.append(client.tool_result_message(tool_call.id, result_text))

            if progress_callback:
                iter_progress = 0.02 + (
                    0.07 * (iteration + 1) / self._config.max_iterations
                )
                result_or_coro = progress_callback(
                    iter_progress, f"agent_iteration_{iteration + 1}"
                )
                if hasattr(result_or_coro, "__await__"):
                    await result_or_coro

        else:
            logger.warning(
                f"AgentContextGatherer: max_iterations={self._config.max_iterations} "
                "reached without final response. Using partial content."
            )
            # Try to get content from the last assistant message
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content"):
                    last_content = m["content"]
                    break

        if not last_content:
            logger.warning("AgentContextGatherer: produced empty context document")

        return last_content
