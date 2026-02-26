# fitz_graveyard/planning/agent/gatherer.py
"""
AgentContextGatherer — multi-pass pipeline to explore the codebase
and produce a structured markdown context document.

Replaces the old tool-calling loop with a deterministic pipeline:
  Pass 1: Map        (pure Python — pathlib walk, build file tree)
  Pass 2a: Select    (1 LLM call — pick relevant files from tree)
  Pass 2b: Summarize (N LLM calls — one per selected file)
  Pass 3: Synthesize (1 LLM call — combine summaries into context doc)

Every LLM call uses generate() (plain text), not generate_with_tools().
"""

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.validation.sanitize import sanitize_agent_path

if TYPE_CHECKING:
    from fitz_graveyard.config.schema import AgentConfig

logger = logging.getLogger(__name__)

_MAX_TREE_FILES = 500

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".eggs", "*.egg-info",
}

_BINARY_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".db", ".sqlite",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".svg",
    ".woff", ".woff2", ".ttf", ".eot",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".bin", ".dat", ".o", ".a", ".lib",
}


class AgentContextGatherer:
    """
    Multi-pass pipeline to gather codebase context.

    Pass 1: Map — pure Python pathlib walk, builds file tree string
    Pass 2a: Select — 1 LLM call picks relevant files
    Pass 2b: Summarize — N LLM calls, one per file
    Pass 3: Synthesize — 1 LLM call combines summaries
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
            file_tree = self._build_file_tree()
            if not file_tree:
                logger.info("AgentContextGatherer: empty or invalid source dir")
                return empty

            # Pass 2a: Select (1 LLM call)
            await self._report(progress_callback, 0.065, "agent:selecting")
            selected = await self._select_files(
                client, model, file_tree, job_description
            )
            if not selected:
                logger.info("AgentContextGatherer: no files selected")
                return empty

            # Pass 2b: Summarize (N LLM calls)
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

            # Pass 3: Synthesize (1 LLM call)
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

    def _build_file_tree(self) -> str:
        """Build a file tree string from the source directory.

        Pure Python: walks directory with pathlib, skips ignored dirs and
        binary files, caps at _MAX_TREE_FILES entries.

        Returns:
            Newline-separated "relative/path (size)" entries, or "" if
            the directory is empty/invalid.
        """
        root = Path(self._source_dir).resolve()
        if not root.is_dir():
            return ""

        entries = []
        try:
            for path in sorted(root.rglob("*")):
                # Skip directories themselves (we list files only)
                if not path.is_file():
                    continue

                # Check if any parent is in skip list
                rel = path.relative_to(root)
                parts = rel.parts
                if any(self._should_skip_dir(p) for p in parts[:-1]):
                    continue

                # Skip binary files
                if path.suffix.lower() in _BINARY_SUFFIXES:
                    continue

                # Use posix-style paths for consistency
                rel_posix = PurePosixPath(*rel.parts)
                size = path.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                else:
                    size_str = f"{size / 1024:.1f}KB"

                entries.append(f"{rel_posix} ({size_str})")

                if len(entries) >= _MAX_TREE_FILES:
                    entries.append(f"... (truncated at {_MAX_TREE_FILES} files)")
                    break
        except OSError as e:
            logger.warning(f"AgentContextGatherer: tree walk error: {e}")
            return ""

        return "\n".join(entries)

    @staticmethod
    def _should_skip_dir(name: str) -> bool:
        """Check if a directory name should be skipped."""
        if name in _SKIP_DIRS:
            return True
        # Handle *.egg-info pattern
        if name.endswith(".egg-info"):
            return True
        return False

    async def _select_files(
        self,
        client: Any,
        model: str,
        file_tree: str,
        job_description: str,
    ) -> list[str]:
        """Ask LLM to select relevant files from the tree.

        1 LLM call via generate(). Parses JSON array from response.
        Falls back to _heuristic_select() on parse failure.
        """
        max_files = self._config.max_summary_files
        prompt = load_prompt("agent_select").format(
            file_tree=file_tree,
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
            logger.warning(f"AgentContextGatherer: select LLM failed: {e}")
            selected = None

        if not selected:
            logger.info("AgentContextGatherer: falling back to heuristic select")
            selected = self._heuristic_select(file_tree, job_description)

        # Validate paths exist and cap at max_files
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

        logger.info(f"AgentContextGatherer: selected {len(valid)} files")
        return valid

    @staticmethod
    def _parse_file_list(response: str) -> list[str] | None:
        """Extract a list of file paths from LLM response.

        Handles:
        - Plain JSON array: ["a.py", "b.py"]
        - JSON object: {"files": ["a.py", "b.py"]}
        - Markdown fenced JSON: ```json\n[...]\n```
        """
        # Strip markdown fences
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
    def _heuristic_select(file_tree: str, job_description: str) -> list[str]:
        """Fallback file selection when LLM fails.

        Priority: README/pyproject.toml first, then shallow __init__.py,
        then remaining sorted by path length (shorter = more central).
        """
        lines = file_tree.strip().splitlines()
        # Extract just the path (before the size in parens)
        paths = []
        for line in lines:
            if line.startswith("..."):
                continue
            # "path/to/file.py (1.2KB)" → "path/to/file.py"
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

        # Sort rest by path depth (shorter paths = more central)
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
