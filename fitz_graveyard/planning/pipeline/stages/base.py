# fitz_graveyard/planning/pipeline/stages/base.py
"""
Abstract base class for pipeline stages.

Each stage defines its prompt template, output parsing logic, and
execution logic. Stages are executed sequentially by the PlanningPipeline.
"""

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a senior software architect and technical strategist with 15 years of experience "
    "shipping production systems at scale. You think rigorously, challenge vague requirements, "
    "call out hidden assumptions, and give specific actionable recommendations — not generic advice. "
    "When something is unclear or underspecified, you state your assumptions explicitly."
)


def _count_unclosed_delimiters(text: str) -> tuple[int, int, bool]:
    """Count unclosed { and [ accounting for JSON string escaping.

    Returns (unclosed_braces, unclosed_brackets, ended_in_string).
    """
    braces = 0
    brackets = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            braces += 1
        elif ch == '}':
            braces -= 1
        elif ch == '[':
            brackets += 1
        elif ch == ']':
            brackets -= 1
    return max(braces, 0), max(brackets, 0), in_string


def _repair_truncated_json(
    candidate: str, braces: int, brackets: int, in_string: bool
) -> dict[str, Any] | None:
    """Try to repair truncated JSON by trimming back and closing delimiters.

    When the LLM hits its token limit mid-output, we may have:
    - An unclosed string (in_string=True)
    - Unclosed arrays/objects (brackets/braces > 0)
    - A partial key-value pair or trailing comma

    Strategy: if in a string, close it, then trim back to the last
    structurally valid point (after a complete value), then close delimiters.
    """
    text = candidate

    # If truncated inside a string, close it
    if in_string:
        text = text + '"'

    # Try progressively trimming from the end to find a parseable state.
    # Strip back to after the last complete JSON value by removing trailing
    # partial tokens (commas, colons, partial keys, whitespace).
    for _ in range(200):  # max trim attempts
        suffix = "]" * brackets + "}" * braces
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            pass

        # Trim: remove last non-whitespace token and any trailing whitespace/commas
        text = text.rstrip()
        if not text:
            return None
        # Remove trailing comma or colon (incomplete key-value)
        if text[-1] in (',', ':'):
            text = text[:-1]
            continue
        # Remove a trailing string that looks like a dangling key: "key"
        # (happens when truncation is after "key": with no value yet)
        if text.endswith('"'):
            # Find the opening quote of this string
            quote_start = text.rfind('"', 0, len(text) - 1)
            if quote_start >= 0:
                before = text[:quote_start].rstrip()
                # If preceded by comma, colon, or opening bracket — trim the string
                if before and before[-1] in (',', ':', '[', '{'):
                    text = before
                    # Also strip trailing comma left behind
                    text = text.rstrip().rstrip(',')
                    continue
        # Remove last character as fallback
        text = text[:-1]
        # Recount after trimming
        braces, brackets, in_string = _count_unclosed_delimiters(text)
        if in_string:
            text = text + '"'
            braces, brackets, in_string = _count_unclosed_delimiters(text)

    return None


def _sanitize_json_strings(text: str) -> str:
    """Replace literal control characters inside JSON string values with escape sequences.

    LLMs often produce literal newlines/tabs inside JSON strings, which is
    invalid JSON. This fixes that while leaving structural whitespace untouched.
    """
    result = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            result.append(ch)
            escape = False
        elif ch == '\\' and in_string:
            result.append(ch)
            escape = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif in_string and ch == '\n':
            result.append('\\n')
        elif in_string and ch == '\r':
            result.append('\\r')
        elif in_string and ch == '\t':
            result.append('\\t')
        else:
            result.append(ch)
    return ''.join(result)


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
    sanitized = _sanitize_json_strings(raw_output)

    # Strategy 1: Direct parse
    try:
        return json.loads(sanitized.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: Code fence (```json ... ```)
    fence_match = re.search(
        r"```(?:json)?\s*\n(.*?)\n```", sanitized, re.DOTALL | re.IGNORECASE
    )
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Bare code block ({...} or [...])
    # Find first { or [ and last matching } or ]
    json_match = re.search(r"(\{.*\}|\[.*\])", sanitized, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 4: Attempt to repair truncated JSON by closing unclosed brackets/braces
    # (handles models that hit context/token limits mid-output)
    first_brace = sanitized.find("{")
    first_bracket = sanitized.find("[")
    if first_brace != -1 or first_bracket != -1:
        start = first_brace if (first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket)) else first_bracket
        candidate = sanitized[start:]
        braces, brackets, in_string = _count_unclosed_delimiters(candidate)
        # Always attempt repair if strategies 1-3 failed — even if delimiters
        # appear balanced, the JSON may have trailing garbage or subtle issues
        repaired = _repair_truncated_json(candidate, braces, brackets, in_string)
        if repaired is not None:
            return repaired

    preview = raw_output[:500].replace('\n', '\\n')
    raise ValueError(
        f"Could not extract valid JSON from output ({len(raw_output)} chars). "
        f"Preview: {preview}"
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

    def __init__(self) -> None:
        self._substep_cb: Callable[[str], Coroutine] | None = None

    def set_substep_callback(
        self, cb: Callable[[str], Coroutine] | None
    ) -> None:
        """Set callback for reporting sub-step progress (e.g. 'architecture:reasoning')."""
        self._substep_cb = cb

    async def _report_substep(self, substep: str) -> None:
        """Report a sub-step if callback is set."""
        if self._substep_cb:
            await self._substep_cb(f"{self.name}:{substep}")

    @property
    def generation_strategy(self) -> str:
        """Generation strategy: 'single_pass' (default) or 'two_pass'."""
        return "single_pass"

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

        The JSON extraction pass uses a fresh minimal context (just the reasoning
        output) rather than repeating the full prompt. This prevents context
        overflow that causes truncated/invalid JSON responses.

        Returns (reasoning_text, json_output_text).
        """
        await self._report_substep("reasoning")
        t0 = time.monotonic()
        reasoning = await client.generate(messages=reasoning_messages)
        t1 = time.monotonic()
        logger.info(f"Stage '{self.name}': two-pass reasoning took {t1 - t0:.1f}s")

        await self._report_substep("formatting")
        # Minimal context for extraction: only the reasoning + schema.
        # Do NOT re-include reasoning_messages — the full codebase context + reasoning
        # combined can exceed the model's practical output length, causing truncation.
        extract_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Below is an architectural analysis. Extract it into structured JSON.\n"
                    "Return ONLY valid JSON matching this exact schema — no prose, no markdown fences:\n\n"
                    f"{schema_json}\n\n"
                    "--- ANALYSIS TO EXTRACT ---\n"
                    f"{reasoning}"
                ),
            },
        ]
        t2 = time.monotonic()
        json_output = await client.generate(messages=extract_messages)
        t3 = time.monotonic()
        logger.info(f"Stage '{self.name}': two-pass formatting took {t3 - t2:.1f}s")
        return reasoning, json_output

    async def _single_pass_with_fallback(
        self,
        client: Any,
        messages: list[dict],
        schema_json: str,
    ) -> str:
        """Try single-pass JSON generation, fall back to two-pass if extraction fails.

        Appends schema instruction to the last user message and asks the model
        to respond with JSON directly. If the output isn't valid JSON, falls
        back to the standard two-pass (reasoning then extraction) approach.

        Returns the raw JSON output string (from either path).
        """
        await self._report_substep("generating")

        json_instruction = (
            "\n\nRespond with ONLY valid JSON matching this schema — "
            "no prose, no markdown fences:\n" + schema_json
        )
        single_pass_messages = [*messages]
        single_pass_messages[-1] = {
            **single_pass_messages[-1],
            "content": single_pass_messages[-1]["content"] + json_instruction,
        }

        t0 = time.monotonic()
        raw = await client.generate(messages=single_pass_messages)
        t1 = time.monotonic()
        logger.info(f"Stage '{self.name}': single-pass took {t1 - t0:.1f}s")

        try:
            extract_json(raw)  # validate — parse_output will re-extract
            return raw
        except ValueError:
            logger.info(
                f"Stage '{self.name}': single-pass JSON failed, falling back to two-pass"
            )
            _, json_output = await self._two_pass(client, messages, schema_json)
            return json_output

    async def _generate_structured(
        self,
        client: Any,
        messages: list[dict],
        schema_json: str,
    ) -> str:
        """Route to single-pass or two-pass based on generation_strategy."""
        if self.generation_strategy == "two_pass":
            _, json_output = await self._two_pass(client, messages, schema_json)
            return json_output
        return await self._single_pass_with_fallback(client, messages, schema_json)

    async def _extract_field_group(
        self,
        client: Any,
        reasoning_text: str,
        field_names: list[str],
        mini_schema: str,
        group_label: str,
        extra_context: str = "",
    ) -> dict[str, Any]:
        """Extract a small group of fields from reasoning text as JSON.

        Builds a minimal prompt asking the LLM to extract specific fields
        from the reasoning into a tiny JSON matching mini_schema. On any
        failure (generation error, invalid JSON), logs a warning and returns
        {} so the caller can fall back to Pydantic defaults.

        Args:
            client: LLM client
            reasoning_text: Free-form reasoning output from the first pass
            field_names: List of field names being extracted (for logging)
            mini_schema: JSON schema string showing expected output shape
            group_label: Human-readable label (e.g. "approaches") for logging/substeps
            extra_context: Optional codebase context to include (e.g. gathered context)

        Returns:
            Parsed dict of extracted fields, or {} on failure
        """
        await self._report_substep(f"extracting:{group_label}")

        context_block = ""
        if extra_context:
            context_block = (
                "--- CODEBASE CONTEXT (use this for accurate file paths, field names, and existing behavior) ---\n"
                f"{extra_context}\n\n"
                "GROUNDING RULE: Every file path, module name, or API field you write in your JSON "
                "MUST appear in the codebase context above. If you cannot find it above, do not invent it.\n\n"
            )

        extract_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Extract the following fields from the analysis below: {', '.join(field_names)}\n"
                    "Return ONLY valid JSON matching this exact schema — no prose, no markdown fences:\n\n"
                    f"{mini_schema}\n\n"
                    f"{context_block}"
                    "--- ANALYSIS TO EXTRACT FROM ---\n"
                    f"{reasoning_text}"
                ),
            },
        ]
        try:
            t0 = time.monotonic()
            raw = await client.generate(messages=extract_messages)
            t1 = time.monotonic()
            logger.info(
                f"Stage '{self.name}': extracted '{group_label}' ({t1 - t0:.1f}s, {len(raw)} chars)"
            )
            return extract_json(raw)
        except Exception as e:
            logger.warning(
                f"Stage '{self.name}': field group '{group_label}' extraction failed: {e}"
            )
            return {}

    async def _self_critique(
        self,
        client: Any,
        reasoning_text: str,
        job_description: str,
        krag_context: str = "",
    ) -> str:
        """Run a self-critique pass on the reasoning text.

        Asks the LLM to review its own reasoning for:
        - Scope inflation (features not requested)
        - Hallucinated files/APIs (not in codebase context)
        - Missed existing code (reinventing what exists)
        - Vague hand-waving instead of specific decisions

        Returns refined reasoning text. On failure, returns original reasoning unchanged.
        """
        await self._report_substep("critiquing")

        context_block = ""
        if krag_context:
            context_block = (
                "--- ACTUAL CODEBASE (ground truth — use to verify file references) ---\n"
                f"{krag_context}\n\n"
            )

        critique_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "You are reviewing your own architectural analysis for quality issues.\n\n"
                    f"ORIGINAL REQUEST: {job_description}\n\n"
                    f"{context_block}"
                    "--- YOUR ANALYSIS TO REVIEW ---\n"
                    f"{reasoning_text}\n\n"
                    "--- REVIEW CHECKLIST ---\n"
                    "Check for these specific problems:\n"
                    "1. SCOPE INFLATION: Features or components not asked for in the original request\n"
                    "2. HALLUCINATED FILES: References to files/APIs/modules that don't exist in the codebase\n"
                    "3. MISSED EXISTING CODE: Proposing to build something that already exists in the codebase\n"
                    "4. VAGUE HAND-WAVING: Generic advice instead of specific, actionable decisions\n"
                    "5. INCONSISTENCIES: Contradictions between different parts of the analysis\n\n"
                    "For each problem found, state what's wrong and how to fix it.\n"
                    "Then rewrite the analysis with all corrections applied.\n"
                    "If no problems found, reproduce the analysis as-is.\n\n"
                    "Output ONLY the corrected analysis — no preamble, no 'Here is the corrected version'."
                ),
            },
        ]

        try:
            t0 = time.monotonic()
            refined = await client.generate(messages=critique_messages)
            t1 = time.monotonic()
            logger.info(
                f"Stage '{self.name}': self-critique took {t1 - t0:.1f}s "
                f"({len(reasoning_text)} → {len(refined)} chars)"
            )
            # Only use refined if it's substantial (not a short error/refusal)
            if len(refined) > len(reasoning_text) * 0.3:
                return refined
            logger.warning(
                f"Stage '{self.name}': critique output too short ({len(refined)} chars), keeping original"
            )
            return reasoning_text
        except Exception as e:
            logger.warning(f"Stage '{self.name}': self-critique failed: {e}")
            return reasoning_text

    async def _devil_advocate(
        self,
        client: Any,
        reasoning_text: str,
        job_description: str,
        krag_context: str = "",
    ) -> str:
        """Run a devil's advocate pass that cross-references reasoning against source code.

        Unlike self-critique (which checks for generic quality issues), this
        specifically looks for factual contradictions between the reasoning's
        claims and the actual source code. Catches errors like assuming a method
        returns a dict when it actually returns str.

        Returns corrected reasoning text. On failure, returns original unchanged.
        """
        if not krag_context:
            return reasoning_text

        await self._report_substep("challenging")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "You are a skeptical code reviewer. Your job is to find where the "
                    "architectural reasoning makes claims that CONTRADICT the actual source code.\n\n"
                    f"ORIGINAL REQUEST: {job_description}\n\n"
                    "--- ACTUAL SOURCE CODE (ground truth) ---\n"
                    f"{krag_context}\n\n"
                    "--- ARCHITECTURAL REASONING TO CHALLENGE ---\n"
                    f"{reasoning_text}\n\n"
                    "--- VERIFICATION CHECKLIST ---\n"
                    "For each claim in the reasoning, verify against the source code:\n"
                    "1. METHOD SIGNATURES: Does the reasoning correctly state return types and parameters?\n"
                    "2. DATA FLOW: Does data actually flow the way the reasoning claims?\n"
                    "3. INTERFACE CONTRACTS: Do classes/protocols actually expose what the reasoning assumes?\n"
                    "4. EXISTING PATTERNS: Does the reasoning correctly describe how the codebase works?\n"
                    "5. ASSUMPTIONS: What implicit assumptions does the reasoning make that the code contradicts?\n\n"
                    "List every factual error you find, citing the specific source code that disproves it.\n"
                    "Then rewrite the reasoning with all factual errors corrected.\n\n"
                    "Output ONLY the corrected reasoning — no preamble."
                ),
            },
        ]

        try:
            t0 = time.monotonic()
            refined = await client.generate(messages=messages)
            t1 = time.monotonic()
            logger.info(
                f"Stage '{self.name}': devil's advocate took {t1 - t0:.1f}s "
                f"({len(reasoning_text)} → {len(refined)} chars)"
            )
            if len(refined) > len(reasoning_text) * 0.3:
                return refined
            logger.warning(
                f"Stage '{self.name}': devil's advocate output too short "
                f"({len(refined)} chars), keeping original"
            )
            return reasoning_text
        except Exception as e:
            logger.warning(f"Stage '{self.name}': devil's advocate failed: {e}")
            return reasoning_text

    # ------------------------------------------------------------------
    # Decomposed reasoning: focused investigations
    # ------------------------------------------------------------------

    _INVESTIGATION_SYSTEM = (
        "You are analyzing source code to answer a specific question about a codebase. "
        "Be precise and concrete. Cite specific files, methods, and return types. "
        "Do not speculate — only state what the code actually shows. "
        "If you cannot determine the answer from the provided code, say so."
    )

    _GENERIC_QUESTIONS = [
        (
            "What are the key interfaces and their exact contracts? "
            "For each important class or protocol, list every public method with its "
            "parameter types and return type. What data goes in and what comes out at each layer?"
        ),
        (
            "Trace the data flow end-to-end for the main operation in this codebase. "
            "At each step, what data is available? Where does data get transformed, "
            "reduced, or discarded? What information exists at one layer but is NOT "
            "passed to the next layer?"
        ),
        (
            "What extension points exist in this codebase (hooks, middleware, plugins, "
            "base classes, event systems)? For each extension point, what data can it "
            "access? What can it NOT see? What are its limitations?"
        ),
        (
            "What patterns and conventions does this codebase follow? How is new "
            "functionality typically added? What existing patterns should new code match? "
            "Are there naming conventions, directory structure rules, or architectural "
            "patterns that are consistently applied?"
        ),
    ]

    _INVESTIGATION_CONCURRENCY = 4

    async def _investigate(
        self,
        client: Any,
        job_description: str,
        prior_outputs: dict[str, Any],
    ) -> str:
        """Run focused investigation questions in parallel before reasoning.

        Each question gets its own LLM call with the full source context,
        producing pre-digested facts that the reasoning pass can build on.

        Returns formatted findings string, or empty string on failure.
        """
        krag_context = self._get_raw_summaries(prior_outputs)
        if not krag_context:
            return ""

        await self._report_substep("investigating")

        # Generic questions (always asked)
        questions = list(self._GENERIC_QUESTIONS)

        # Customized questions (from AST data)
        agent_files = prior_outputs.get("_agent_context", {}).get("agent_files", {})
        if agent_files:
            from fitz_graveyard.planning.agent.indexer import generate_investigation_questions
            # Extract signatures from the raw context (it's prepended)
            sig_marker = "--- INTERFACE SIGNATURES"
            sig_end = krag_context.find("\n\n###")
            signatures = krag_context[krag_context.find(sig_marker):sig_end] if sig_marker in krag_context else ""

            # Reconstruct forward_map from serialized form
            fwd_raw = agent_files.get("forward_map", {})
            forward_map = {k: set(v) for k, v in fwd_raw.items()}
            reverse_count = agent_files.get("reverse_count", {})

            custom = generate_investigation_questions(signatures, forward_map, reverse_count)
            if custom:
                logger.info(
                    f"Stage '{self.name}': generated {len(custom)} "
                    f"customized investigation questions"
                )
                questions.extend(custom)

        # Run all investigations in parallel
        sem = asyncio.Semaphore(self._INVESTIGATION_CONCURRENCY)

        async def _bounded(q: str) -> tuple[str, str]:
            async with sem:
                answer = await self._ask_one(client, q, job_description, krag_context)
                return q, answer

        t0 = time.monotonic()
        results = await asyncio.gather(
            *[_bounded(q) for q in questions],
            return_exceptions=True,
        )
        t1 = time.monotonic()

        # Format findings
        findings: list[str] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Stage '{self.name}': investigation failed: {r}")
                continue
            q, a = r
            if a:
                findings.append(f"Q: {q}\nA: {a}")

        if not findings:
            return ""

        logger.info(
            f"Stage '{self.name}': {len(findings)}/{len(questions)} investigations "
            f"completed in {t1 - t0:.1f}s"
        )
        return (
            "--- INVESTIGATION FINDINGS (verified against source code) ---\n\n"
            + "\n\n".join(findings)
            + "\n\n---"
        )

    async def _ask_one(
        self,
        client: Any,
        question: str,
        job_description: str,
        krag_context: str,
    ) -> str:
        """Ask one focused investigation question about the codebase."""
        messages = [
            {"role": "system", "content": self._INVESTIGATION_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"TASK: {job_description}\n\n"
                    f"--- SOURCE CODE ---\n{krag_context}\n\n"
                    f"QUESTION: {question}\n\n"
                    "Answer concisely but precisely. Cite specific files and methods."
                ),
            },
        ]
        try:
            return await client.generate(messages=messages, temperature=0)
        except Exception as e:
            logger.warning(f"Stage '{self.name}': _ask_one failed: {e}")
            return ""

    _MAX_GATHERED_CONTEXT_CHARS = 32000

    def _get_gathered_context(self, prior_outputs: dict[str, Any]) -> str:
        """
        Get pre-gathered codebase context from AgentContextGatherer output.

        Retrieves the context string stored in prior_outputs['_gathered_context']
        by the pipeline orchestrator before stage execution begins.
        Returns empty string if not available (graceful fallback).
        Caps at _MAX_GATHERED_CONTEXT_CHARS to avoid inflating stage prompts.

        Args:
            prior_outputs: Dictionary containing '_gathered_context' string

        Returns:
            Context markdown string, or empty string if unavailable
        """
        ctx = prior_outputs.get("_gathered_context", "")
        if len(ctx) > self._MAX_GATHERED_CONTEXT_CHARS:
            logger.info(
                f"Trimming gathered context: {len(ctx)} -> {self._MAX_GATHERED_CONTEXT_CHARS} chars"
            )
            ctx = ctx[:self._MAX_GATHERED_CONTEXT_CHARS] + "\n\n[... context trimmed for brevity]"
        return ctx

    _MAX_RAW_SUMMARIES_CHARS = 40000

    def _get_raw_summaries(self, prior_outputs: dict[str, Any]) -> str:
        """
        Get raw source code from AgentContextGatherer output.

        Contains actual source code of selected files (fenced code blocks).
        Use for reasoning passes where seeing real code matters.
        Falls back to _gathered_context if not available.

        Caps at _MAX_RAW_SUMMARIES_CHARS (higher limit than synthesized context
        since reasoning passes benefit from more detail).
        """
        ctx = prior_outputs.get("_raw_summaries", "")
        if not ctx:
            # Fallback to synthesized if raw not available (old checkpoints)
            return self._get_gathered_context(prior_outputs)
        if len(ctx) > self._MAX_RAW_SUMMARIES_CHARS:
            logger.info(
                f"Trimming raw summaries: {len(ctx)} -> {self._MAX_RAW_SUMMARIES_CHARS} chars"
            )
            ctx = ctx[:self._MAX_RAW_SUMMARIES_CHARS] + "\n\n[... summaries trimmed for brevity]"
        return ctx

    @staticmethod
    def _get_source_dir(prior_outputs: dict[str, Any]) -> str | None:
        """Get the codebase source directory for reading actual source files."""
        return prior_outputs.get("_source_dir")

    def _get_implementation_check(self, prior_outputs: dict[str, Any]) -> str:
        """Format the implementation check result for injection into prompts.

        Returns a short directive string if the task is already implemented,
        or empty string if not (or if no check was performed).
        """
        check = prior_outputs.get("_implementation_check", {})
        if not check or not isinstance(check, dict):
            return ""
        if not check.get("already_implemented"):
            return ""
        evidence = check.get("evidence", "")
        gaps = check.get("gaps", [])
        text = f"IMPORTANT — EXISTING IMPLEMENTATION DETECTED: {evidence}"
        if gaps:
            text += f"\nGaps to fill: {', '.join(gaps)}"
        else:
            text += "\nNo gaps identified — task may already be complete."
        text += "\nDo NOT propose building something that already exists. If the task is done, say so."
        return text

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
