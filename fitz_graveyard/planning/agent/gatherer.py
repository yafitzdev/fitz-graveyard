# fitz_graveyard/planning/agent/gatherer.py
"""
AgentContextGatherer — multi-pass pipeline to explore the codebase
and produce a structured markdown context document.

Pipeline:
  Pass 1: Map        (pure Python — pathlib walk, build file list)
  Pass 2: Index      (pure Python — structural extraction per file)
  Pass 3: Navigate   (1 LLM call — select files using structural index)
  Pass 4: Summarize  (N LLM calls — one per selected file)
  Pass 5: Synthesize (1 LLM call — combine summaries into context doc)

Every LLM call uses generate() (plain text), not generate_with_tools().
"""

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from fitz_graveyard.planning.agent.indexer import INDEXABLE_EXTENSIONS, build_structural_index
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.validation.sanitize import sanitize_agent_path

if TYPE_CHECKING:
    from fitz_graveyard.config.schema import AgentConfig

logger = logging.getLogger(__name__)

_MAX_TREE_FILES = 2000

# Directories that are never source code in any project.
# Intentionally minimal — the real filter is INDEXABLE_EXTENSIONS.
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}


class AgentContextGatherer:
    """
    Multi-pass pipeline to gather codebase context.

    Pass 1: Map       — pure Python pathlib walk, builds file list
    Pass 2: Index     — pure Python structural extraction (AST, regex, parsers)
    Pass 3: Navigate  — 1 LLM call selects files using structural index
    Pass 4: Summarize — N LLM calls, one per file
    Pass 5: Synthesize — 1 LLM call combines summaries
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
        Run the multi-pass pipeline and return context dict.

        Args:
            client:            LLM client (OllamaClient/LMStudioClient)
            job_description:   The user's planning request
            progress_callback: Optional (progress: float, phase: str) callback

        Returns:
            Dict with "synthesized" (concise context doc) and "raw_summaries"
            (detailed per-file summaries). Both "" on failure/disabled.
        """
        empty = {"synthesized": "", "raw_summaries": ""}

        if not self._config.enabled:
            logger.info("Agent context gathering disabled by config")
            return empty

        model = self._config.agent_model or client.model

        try:
            # Pass 1: Map (pure Python, instant)
            await self._report(progress_callback, 0.06, "agent:mapping")
            file_tree, file_paths = self._build_file_tree()
            if not file_paths:
                logger.info("AgentContextGatherer: empty or invalid source dir")
                return empty

            # Pass 2: Index (pure Python, no LLM)
            await self._report(progress_callback, 0.063, "agent:indexing")
            structural_index = self._build_index(file_paths)
            logger.info(
                f"AgentContextGatherer: indexed {len(file_paths)} files "
                f"({len(structural_index)} chars)"
            )

            # Pass 3: Navigate (1 LLM call with structural index)
            await self._report(progress_callback, 0.065, "agent:navigating")
            selected = await self._navigate_files(
                client, model, structural_index, job_description
            )
            if not selected:
                logger.info("AgentContextGatherer: no files selected")
                return empty

            # Pass 4: Summarize (N LLM calls)
            summaries = []
            for i, rel_path in enumerate(selected):
                progress = 0.065 + 0.02 * ((i + 1) / len(selected))
                await self._report(
                    progress_callback, progress, f"agent:summarizing:{rel_path}"
                )
                summary = await self._summarize_file(
                    client, model, rel_path, job_description
                )
                if summary:
                    summaries.append(f"### {rel_path}\n{summary}")

            if not summaries:
                logger.warning("AgentContextGatherer: all summaries failed")
                return empty

            raw_summaries = "\n\n".join(summaries)

            # Pass 5: Synthesize (1 LLM call)
            await self._report(progress_callback, 0.085, "agent:synthesizing")
            synthesized = await self._synthesize(
                client, model, summaries, job_description
            )

            logger.info(
                f"AgentContextGatherer: synthesized={len(synthesized)} chars, "
                f"raw_summaries={len(raw_summaries)} chars"
            )
            return {"synthesized": synthesized, "raw_summaries": raw_summaries}

        except Exception:
            logger.exception("AgentContextGatherer: pipeline failed")
            return empty

    def _build_file_tree(self) -> tuple[str, list[str]]:
        """Build a file tree string and path list from the source directory.

        Only includes files whose extension is in INDEXABLE_EXTENSIONS —
        this is the primary filter that keeps junk out regardless of
        directory names.  A minimal _SKIP_DIRS set handles the few
        directories that are universally non-source (.git, __pycache__,
        .venv, node_modules).

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

                # Only include files the indexer can extract structure from
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
                    entries.append(f"... (truncated at {_MAX_TREE_FILES} files)")
                    break
        except OSError as e:
            logger.warning(f"AgentContextGatherer: tree walk error: {e}")
            return "", []

        return "\n".join(entries), paths

    @staticmethod
    def _should_skip_dir(name: str) -> bool:
        """Check if a directory name should be skipped."""
        return name in _SKIP_DIRS

    def _build_index(self, file_paths: list[str]) -> str:
        """Build structural index of all files using pure Python extraction."""
        return build_structural_index(
            self._source_dir, file_paths, self._config.max_file_bytes,
        )

    async def _navigate_files(
        self,
        client: Any,
        model: str,
        structural_index: str,
        job_description: str,
    ) -> list[str]:
        """Ask LLM to select relevant files using the structural index.

        1 LLM call via generate(). Parses JSON array from response.
        Falls back to _heuristic_select() on parse failure.
        """
        max_files = self._config.max_summary_files

        prompt = load_prompt("agent_navigate").format(
            structural_index=structural_index,
            job_description=job_description,
            max_files=max_files,
        )

        try:
            response = await client.generate(
                messages=[{"role": "user", "content": prompt}],
                model=model,
            )
            selected = self._parse_file_list(response)
        except Exception as e:
            logger.warning(f"AgentContextGatherer: navigate LLM failed: {e}")
            selected = None

        if not selected:
            logger.info("AgentContextGatherer: falling back to heuristic select")
            selected = self._heuristic_select(structural_index, job_description)

        # Validate paths exist, stay in source_dir, and cap at max_files
        root = Path(self._source_dir).resolve()
        valid = []
        for rel_path in selected:
            if len(valid) >= max_files:
                break
            full = root / rel_path
            if full.is_file():
                try:
                    full.resolve().relative_to(root)
                    valid.append(rel_path)
                except ValueError:
                    continue

        logger.info(f"AgentContextGatherer: selected {len(valid)} files via structural index")
        return valid

    @staticmethod
    def _parse_file_list(response: str) -> list[str] | None:
        """Extract a list of file paths from LLM response.

        Handles:
        - Plain JSON array: ["a.py", "b.py"]
        - JSON object: {"files": ["a.py", "b.py"]}
        - Markdown fenced JSON: ```json\\n[...]\\n```
        """
        text = response.strip()
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
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
    def _heuristic_select(index_or_tree: str, job_description: str) -> list[str]:
        """Fallback file selection when LLM fails.

        Extracts file paths from the structural index (## path lines),
        prioritizes project roots (README, pyproject.toml), then sorts
        by path depth (shorter = more central).
        """
        # Extract paths from ## headers in structural index
        paths = re.findall(r'^## (.+)$', index_or_tree, re.MULTILINE)

        if not paths:
            # Fallback: try parsing as file tree ("path (size)" format)
            for line in index_or_tree.strip().splitlines():
                if line.startswith("..."):
                    continue
                match = re.match(r"^(.+?)\s+\(", line)
                if match:
                    paths.append(match.group(1))

        priority = []
        rest = []

        for p in paths:
            name = PurePosixPath(p).name.lower()
            if name in ("readme.md", "readme.rst", "readme.txt", "readme",
                        "pyproject.toml", "setup.py", "setup.cfg",
                        "package.json", "cargo.toml"):
                priority.append(p)
            elif name == "__init__.py" and p.count("/") <= 1:
                priority.append(p)
            else:
                rest.append(p)

        rest.sort(key=lambda p: (p.count("/"), len(p)))
        return priority + rest

    async def _summarize_file(
        self,
        client: Any,
        model: str,
        rel_path: str,
        job_description: str,
    ) -> str | None:
        """Read and summarize a single file.

        1 LLM call. Returns None on any failure.
        """
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
            logger.warning(f"AgentContextGatherer: can't read {rel_path}: {e}")
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
            )
        except Exception as e:
            logger.warning(f"AgentContextGatherer: summarize failed for {rel_path}: {e}")
            return None

    async def _synthesize(
        self,
        client: Any,
        model: str,
        summaries: list[str],
        job_description: str,
    ) -> str:
        """Combine all file summaries into a structured context document.

        1 LLM call. Falls back to concatenation if LLM fails.
        """
        combined = "\n\n".join(summaries)
        prompt = load_prompt("agent_synthesize").format(
            summaries=combined,
            job_description=job_description,
        )

        try:
            return await client.generate(
                messages=[{"role": "user", "content": prompt}],
                model=model,
            )
        except Exception as e:
            logger.warning(f"AgentContextGatherer: synthesize failed: {e}")
            return f"## File Summaries\n\n{combined}"

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
