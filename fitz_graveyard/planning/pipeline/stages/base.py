# fitz_graveyard/planning/pipeline/stages/base.py
"""
Abstract base class for pipeline stages.

Each stage defines its prompt template, output parsing logic, and
execution logic. Stages are executed sequentially by the PlanningPipeline.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a senior software architect and technical strategist with 15 years of experience "
    "shipping production systems at scale. You think rigorously, challenge vague requirements, "
    "call out hidden assumptions, and give specific actionable recommendations — not generic advice. "
    "When something is unclear or underspecified, you state your assumptions explicitly."
)


def extract_json(raw_output: str) -> dict[str, Any]:
    """
    Extract JSON from LLM output, handling common formatting variations.

    Tries multiple extraction strategies:
    1. Direct JSON parse (if output is pure JSON)
    2. Code fence extraction (```json ... ```)
    3. Bare code block extraction ({...} or [...])

    Args:
        raw_output: Raw text from LLM

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If no valid JSON found
    """
    # Strategy 1: Direct parse
    try:
        return json.loads(raw_output.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: Code fence (```json ... ```)
    fence_match = re.search(
        r"```(?:json)?\s*\n(.*?)\n```", raw_output, re.DOTALL | re.IGNORECASE
    )
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Bare code block ({...} or [...])
    # Find first { or [ and last matching } or ]
    json_match = re.search(r"(\{.*\}|\[.*\])", raw_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract valid JSON from output. First 200 chars: {raw_output[:200]}"
    )


@dataclass
class StageResult:
    """
    Result of executing a pipeline stage.

    Attributes:
        stage_name: Name of the stage that produced this result
        success: Whether the stage completed successfully
        output: Parsed output data (structure varies by stage)
        raw_output: Raw LLM response text
        error: Error message if success=False
    """

    stage_name: str
    success: bool
    output: dict[str, Any]
    raw_output: str
    error: str | None = None


class PipelineStage(ABC):
    """
    Abstract base class for planning pipeline stages.

    Each stage defines:
    1. Prompt template construction (from job + prior outputs)
    2. Output parsing (raw LLM response → structured data)
    3. Execution (orchestrated by PlanningPipeline)

    Stages are executed sequentially, with each stage receiving
    the outputs of all prior stages.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Stage name (used in logging and checkpointing).

        Returns:
            Unique stage identifier (e.g., "vision", "architectural_analysis")
        """
        pass

    @property
    @abstractmethod
    def progress_range(self) -> tuple[float, float]:
        """
        Progress range this stage covers (0.0 to 1.0).

        Used to map stage progress to overall job progress.
        Example: Stage 2 of 5 might return (0.2, 0.4).

        Returns:
            Tuple of (start_progress, end_progress)
        """
        pass

    @abstractmethod
    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any]
    ) -> list[dict]:
        """
        Build the LLM prompt for this stage.

        Args:
            job_description: User's planning request
            prior_outputs: Dictionary mapping stage_name -> output dict
                          (empty for first stage)

        Returns:
            List of message dicts with "role" and "content" keys
        """
        pass

    @abstractmethod
    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """
        Parse the raw LLM output into structured data.

        Args:
            raw_output: Raw text response from LLM

        Returns:
            Structured output dictionary (schema varies by stage)

        Raises:
            ValueError: If output cannot be parsed
        """
        pass

    def _make_messages(self, user_content: str) -> list[dict]:
        """Build messages list with system prompt prepended."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    async def _two_pass(
        self,
        client: Any,
        reasoning_messages: list[dict],
        schema_json: str,
    ) -> tuple[str, str]:
        """
        Two-pass execution: free-form reasoning then JSON formatting.

        Returns (reasoning_text, json_output_text).
        """
        reasoning = await client.generate(messages=reasoning_messages)

        format_messages = reasoning_messages + [
            {"role": "assistant", "content": reasoning},
            {
                "role": "user",
                "content": (
                    "Based on your analysis above, extract the structured output.\n"
                    "Return ONLY valid JSON matching this exact schema:\n\n"
                    + schema_json
                ),
            },
        ]
        json_output = await client.generate(messages=format_messages)
        return reasoning, json_output

    def _get_gathered_context(self, prior_outputs: dict[str, Any]) -> str:
        """
        Get pre-gathered codebase context from AgentContextGatherer output.

        Retrieves the context string stored in prior_outputs['_gathered_context']
        by the pipeline orchestrator before stage execution begins.
        Returns empty string if not available (graceful fallback).

        Args:
            prior_outputs: Dictionary containing '_gathered_context' string

        Returns:
            Context markdown string, or empty string if unavailable
        """
        return prior_outputs.get("_gathered_context", "")

    async def execute(
        self,
        client: Any,  # OllamaClient (avoiding circular import)
        job_description: str,
        prior_outputs: dict[str, Any],
    ) -> StageResult:
        """
        Execute this stage (prompt construction + LLM call + parsing).

        Default implementation:
        1. Build prompt from job + prior outputs
        2. Call LLM via client.generate()
        3. Parse output
        4. Return StageResult

        Can be overridden for custom execution logic.

        Args:
            client: OllamaClient instance for LLM calls
            job_description: User's planning request
            prior_outputs: Outputs from all prior stages

        Returns:
            StageResult with parsed output or error
        """
        try:
            # Build prompt
            messages = self.build_prompt(job_description, prior_outputs)
            logger.info(f"Stage '{self.name}': Built prompt with {len(messages)} messages")

            # Call LLM
            logger.info(f"Stage '{self.name}': Calling LLM")
            raw_output = await client.generate(messages=messages)

            logger.info(
                f"Stage '{self.name}': Received {len(raw_output)} chars from LLM"
            )

            # Parse output
            parsed = self.parse_output(raw_output)
            logger.info(f"Stage '{self.name}': Parsed output successfully")

            return StageResult(
                stage_name=self.name,
                success=True,
                output=parsed,
                raw_output=raw_output,
            )

        except Exception as e:
            logger.error(f"Stage '{self.name}' failed: {e}", exc_info=True)
            return StageResult(
                stage_name=self.name,
                success=False,
                output={},
                raw_output="",
                error=str(e),
            )
