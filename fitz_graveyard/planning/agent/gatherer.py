# fitz_graveyard/planning/agent/gatherer.py
"""
AgentContextGatherer — brute-force parallel file screening pipeline.

Pipeline:
  Pass 1:  Map         (pure Python — pathlib walk, build file list)
  Pass 2:  Screen      (N parallel LLM calls — "is this file relevant?")
  Pass 3:  Summarize   (N parallel LLM calls — one per relevant file)
  Pass 4:  Synthesize  (1 LLM call — combine summaries into context doc)

Every file in the codebase is shown to the LLM individually for
relevance screening. No heuristics, no graph traversal, no truncation.
Parallelized with asyncio.Semaphore for MoE models that handle
concurrent requests efficiently.
"""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from fitz_graveyard.planning.agent.indexer import INDEXABLE_EXTENSIONS
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.validation.sanitize import sanitize_agent_path

if TYPE_CHECKING:
    from fitz_graveyard.config.schema import AgentConfig

logger = logging.getLogger(__name__)

_MAX_TREE_FILES = 2000
_CONCURRENCY = 8

# Directories that are never source code in any project.
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}


class AgentContextGatherer:
    """
    Brute-force parallel pipeline to gather codebase context.

    Every file is individually screened by the LLM for relevance.
    No keyword matching, no import graphs, no structural index truncation.
    """

    def __init__(self, config: "AgentConfig", source_dir: str) -> None:
        self._config = config
        self._source_dir = source_dir

    async def gather(
        self,
        client: Any,
        job_description: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, str]:
        """
        Run the brute-force pipeline and return context dict.

        Returns:
            Dict with "synthesized", "raw_summaries", and "agent_files".
            Both strings "" on failure.
        """
        empty = {"synthesized": "", "raw_summaries": ""}

        if not self._config.enabled:
            logger.info("Agent context gathering disabled by config")
            return empty

        model = self._config.agent_model or client.model

        try:
            # Pass 1: Map (pure Python)
            await self._report(progress_callback, 0.06, "agent:mapping")
            _file_tree, file_paths = self._build_file_tree()
            if not file_paths:
                logger.info("AgentContextGatherer: empty or invalid source dir")
                return empty

            logger.info(
                f"AgentContextGatherer: mapped {len(file_paths)} files"
            )

            # Pass 2: Screen all files (parallel LLM calls)
            await self._report(progress_callback, 0.062, "agent:screening")
            relevant = await self._screen_all(
                client, model, file_paths, job_description,
                progress_callback,
            )

            if not relevant:
                logger.info("AgentContextGatherer: no relevant files found")
                return empty

            logger.info(
                f"AgentContextGatherer: screening found {len(relevant)} "
                f"relevant files out of {len(file_paths)}: "
                + ", ".join(relevant)
            )

            # Cap at max_summary_files for summarization
            selected = relevant
            if len(selected) > self._config.max_summary_files:
                logger.info(
                    f"AgentContextGatherer: capping {len(selected)} relevant "
                    f"to {self._config.max_summary_files} for summarization"
                )
                selected = selected[:self._config.max_summary_files]

            # Pass 3: Summarize (parallel LLM calls)
            await self._report(progress_callback, 0.074, "agent:summarizing")
            summaries = await self._summarize_all(
                client, model, selected, job_description,
                progress_callback,
            )

            if not summaries:
                logger.warning("AgentContextGatherer: all summaries failed")
                return empty

            raw_summaries = "\n\n".join(summaries)

            logger.info(
                f"AgentContextGatherer: summarized {len(summaries)} files"
            )

            # Pass 4: Synthesize (1 LLM call)
            await self._report(progress_callback, 0.088, "agent:synthesizing")
            synthesized = await self._synthesize(
                client, model, summaries, job_description
            )

            logger.info(
                f"AgentContextGatherer: synthesized={len(synthesized)} chars, "
                f"raw_summaries={len(raw_summaries)} chars"
            )
            return {
                "synthesized": synthesized,
                "raw_summaries": raw_summaries,
                "agent_files": {
                    "total_screened": len(file_paths),
                    "relevant": relevant,
                    "selected": selected,
                },
            }

        except Exception:
            logger.exception("AgentContextGatherer: pipeline failed")
            return empty

    # ------------------------------------------------------------------
    # Pass 1: Map
    # ------------------------------------------------------------------

    def _build_file_tree(self) -> tuple[str, list[str]]:
        """Build a file tree string and path list from the source directory.

        Returns:
            Tuple of (tree_string, list_of_relative_posix_paths).
            ("", []) if directory is empty/invalid.
        """
        root = Path(self._source_dir).resolve()
        if not root.is_dir():
            return "", []

        entries = []
        paths = []
        try:
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in INDEXABLE_EXTENSIONS:
                    continue

                rel = path.relative_to(root)
                parts = rel.parts
                if any(self._should_skip_dir(p) for p in parts[:-1]):
                    continue

                rel_posix = str(PurePosixPath(*rel.parts))
                size = path.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                else:
                    size_str = f"{size / 1024:.1f}KB"

                entries.append(f"{rel_posix} ({size_str})")
                paths.append(rel_posix)

                if len(entries) >= _MAX_TREE_FILES:
                    entries.append(
                        f"... (truncated at {_MAX_TREE_FILES} files)"
                    )
                    break
        except OSError as e:
            logger.warning(f"AgentContextGatherer: tree walk error: {e}")
            return "", []

        return "\n".join(entries), paths

    @staticmethod
    def _should_skip_dir(name: str) -> bool:
        """Check if a directory name should be skipped."""
        return name in _SKIP_DIRS

    # ------------------------------------------------------------------
    # Pass 2: Screen
    # ------------------------------------------------------------------

    async def _screen_file(
        self,
        client: Any,
        model: str,
        rel_path: str,
        job_description: str,
    ) -> bool:
        """Ask LLM if a single file is relevant to the task.

        Reads the file, sends content + task to LLM, parses YES/NO.
        Returns False on any error (file unreadable, LLM failure, etc).
        """
        try:
            resolved = sanitize_agent_path(rel_path, self._source_dir)
        except ValueError:
            return False

        if not resolved.is_file():
            return False

        try:
            raw = resolved.read_bytes()[:self._config.max_file_bytes]
            content = raw.decode("utf-8", errors="replace")
        except OSError:
            return False

        if not content.strip():
            return False

        prompt = load_prompt("agent_screen").format(
            job_description=job_description,
            file_path=rel_path,
            file_content=content,
        )

        try:
            response = await client.generate(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0,
            )
            return self._parse_yes_no(response)
        except Exception as e:
            logger.warning(
                f"AgentContextGatherer: screen failed for {rel_path}: {e}"
            )
            return False

    @staticmethod
    def _parse_yes_no(response: str) -> bool:
        """Parse a YES/NO response from the LLM."""
        text = response.strip().upper()
        # Check first word
        first_word = text.split()[0] if text.split() else ""
        return first_word == "YES"

    async def _screen_all(
        self,
        client: Any,
        model: str,
        file_paths: list[str],
        job_description: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[str]:
        """Screen all files in parallel. Returns paths the LLM marked relevant."""
        sem = asyncio.Semaphore(_CONCURRENCY)
        completed = 0
        total = len(file_paths)

        async def _do_one(path: str) -> tuple[str, bool]:
            nonlocal completed
            async with sem:
                result = await self._screen_file(
                    client, model, path, job_description,
                )
                completed += 1
                if progress_callback and total > 0:
                    pct = 0.062 + 0.012 * (completed / total)
                    await self._report(
                        progress_callback, pct,
                        f"agent:screening:{completed}/{total}",
                    )
                return path, result

        tasks = [_do_one(p) for p in file_paths]
        results = await asyncio.gather(*tasks)
        return [path for path, relevant in results if relevant]

    # ------------------------------------------------------------------
    # Pass 3: Summarize
    # ------------------------------------------------------------------

    async def _summarize_file(
        self,
        client: Any,
        model: str,
        rel_path: str,
        job_description: str,
    ) -> str | None:
        """Read and summarize a single file. 1 LLM call."""
        try:
            resolved = sanitize_agent_path(rel_path, self._source_dir)
        except ValueError as e:
            logger.warning(f"AgentContextGatherer: path rejected: {e}")
            return None

        if not resolved.is_file():
            return None

        try:
            raw = resolved.read_bytes()[: self._config.max_file_bytes]
            content = raw.decode("utf-8", errors="replace")
        except OSError as e:
            logger.warning(
                f"AgentContextGatherer: can't read {rel_path}: {e}"
            )
            return None

        if not content.strip():
            return None

        prompt = load_prompt("agent_summarize").format(
            file_path=rel_path,
            file_content=content,
            job_description=job_description,
        )

        try:
            return await client.generate(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0,
            )
        except Exception as e:
            logger.warning(
                f"AgentContextGatherer: summarize failed for {rel_path}: {e}"
            )
            return None

    async def _summarize_all(
        self,
        client: Any,
        model: str,
        file_paths: list[str],
        job_description: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[str]:
        """Summarize all files in parallel. Returns formatted summaries."""
        sem = asyncio.Semaphore(_CONCURRENCY)
        completed = 0
        total = len(file_paths)

        async def _do_one(path: str) -> tuple[str, str | None]:
            nonlocal completed
            async with sem:
                result = await self._summarize_file(
                    client, model, path, job_description,
                )
                completed += 1
                if progress_callback and total > 0:
                    pct = 0.074 + 0.014 * (completed / total)
                    await self._report(
                        progress_callback, pct,
                        f"agent:summarizing:{path}",
                    )
                return path, result

        tasks = [_do_one(p) for p in file_paths]
        results = await asyncio.gather(*tasks)
        return [
            f"### {path}\n{summary}"
            for path, summary in results
            if summary
        ]

    # ------------------------------------------------------------------
    # Pass 4: Synthesize
    # ------------------------------------------------------------------

    async def _synthesize(
        self,
        client: Any,
        model: str,
        summaries: list[str],
        job_description: str,
    ) -> str:
        """Combine all file summaries into a structured context document."""
        combined = "\n\n".join(summaries)
        prompt = load_prompt("agent_synthesize").format(
            summaries=combined,
            job_description=job_description,
        )

        try:
            return await client.generate(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0,
            )
        except Exception as e:
            logger.warning(f"AgentContextGatherer: synthesize failed: {e}")
            return f"## File Summaries\n\n{combined}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_file_list(response: str) -> list[str] | None:
        """Extract a list of file paths from LLM response."""
        text = response.strip()
        fence_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL,
        )
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None

        if isinstance(parsed, list):
            return [str(p) for p in parsed if isinstance(p, str)]
        if isinstance(parsed, dict) and "files" in parsed:
            files = parsed["files"]
            if isinstance(files, list):
                return [str(p) for p in files if isinstance(p, str)]

        return None

    @staticmethod
    async def _report(
        callback: Callable[[float, str], None] | None,
        progress: float,
        phase: str,
    ) -> None:
        """Report progress, handling both sync and async callbacks."""
        if not callback:
            return
        result = callback(progress, phase)
        if hasattr(result, "__await__"):
            await result
