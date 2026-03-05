# fitz_graveyard/planning/agent/gatherer.py
"""
AgentContextGatherer — brute-force parallel file screening pipeline.

Pipeline:
  Pass 1:  Map            (pure Python — pathlib walk, build file list)
  Pass 2a: Broad screen   (N parallel LLM calls — fast model, "is this relevant?")
  Pass 2b: Refine screen  (M parallel LLM calls — mid model on broad candidates)
  Pass 3:  Import expand  (pure Python — trace forward imports from relevant files)
  Pass 4:  Read raw source (pure Python — stuff actual source into context)

Two-stage screening cascade: fast model (4B) screens all files for broad
relevance (high recall, many false positives), then mid model (30B MoE)
re-screens the candidates for precise selection (high precision).
Raw source is stuffed directly into the planning context — no summarization
or synthesis LLM calls. Files sorted by import connectivity; truncated at
a char budget so they fit in the reasoning model's context window.
Parallelized with asyncio.Semaphore.
"""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from fitz_graveyard.planning.agent.indexer import (
    INDEXABLE_EXTENSIONS,
    build_import_graph,
    extract_interface_signatures,
)
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

    Every file is individually screened by the LLM for relevance,
    then the raw source of selected files is stuffed into context.
    No summarization, no synthesis — the planning stages see real code.
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

        fast = self._config.agent_model or client.fast_model
        mid = self._config.agent_model or client.mid_model
        smart = self._config.agent_model or client.smart_model

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

            # Pass 2a: Broad screen (parallel LLM calls — fast model)
            await self._report(progress_callback, 0.062, "agent:screening")
            broad = await self._screen_all(
                client, fast, file_paths, job_description,
                progress_callback,
            )

            if not broad:
                logger.info("AgentContextGatherer: no relevant files found")
                return empty

            logger.info(
                f"AgentContextGatherer: broad screen found {len(broad)} "
                f"candidates out of {len(file_paths)}"
            )

            # Pass 2b: Precise screen (parallel LLM calls — mid model)
            await self._report(progress_callback, 0.068, "agent:refining")
            relevant = await self._screen_all(
                client, mid, broad, job_description,
            )

            if not relevant:
                logger.info(
                    "AgentContextGatherer: mid screen rejected all, "
                    "falling back to broad results"
                )
                relevant = broad

            logger.info(
                f"AgentContextGatherer: refined to {len(relevant)} "
                f"relevant files from {len(broad)} candidates: "
                + ", ".join(relevant)
            )

            # Pass 3: Import expansion (pure Python, depth 1)
            expanded = self._import_expand(relevant, file_paths)
            import_added = len(expanded) - len(relevant)
            if import_added > 0:
                logger.info(
                    f"AgentContextGatherer: import expansion added "
                    f"{import_added} files"
                )

            # Prioritize code files over docs, then cap
            selected = self._prioritize_for_summary(expanded)
            if len(selected) > self._config.max_summary_files:
                logger.info(
                    f"AgentContextGatherer: capping {len(selected)} relevant "
                    f"to {self._config.max_summary_files} for summarization"
                )
                selected = selected[:self._config.max_summary_files]

            # Pass 4: Read raw source (pure Python, no LLM)
            await self._report(progress_callback, 0.074, "agent:reading")
            raw_source, included, fwd_map, rev_count = self._read_raw_source(
                selected, file_paths,
            )

            if not raw_source:
                logger.warning("AgentContextGatherer: no readable source files")
                return empty

            logger.info(
                f"AgentContextGatherer: stuffed {len(included)}/{len(selected)} "
                f"files ({len(raw_source)} chars) into context"
            )
            # Serialize forward_map for JSON compatibility
            serializable_fwd = {k: sorted(v) for k, v in fwd_map.items()}
            return {
                "synthesized": raw_source,
                "raw_summaries": raw_source,
                "agent_files": {
                    "total_screened": len(file_paths),
                    "broad": broad,
                    "relevant": relevant,
                    "import_expanded": import_added,
                    "selected": selected,
                    "included": included,
                    "forward_map": serializable_fwd,
                    "reverse_count": rev_count,
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
    # Pass 3: Import expansion
    # ------------------------------------------------------------------

    def _import_expand(
        self, relevant: list[str], file_paths: list[str],
    ) -> list[str]:
        """Add direct imports of relevant files. Pure Python, depth 1.

        Traces forward imports from each relevant file and adds any
        new files not already in the relevant list. This catches
        architecturally central files (e.g. base.py, protocols) that
        are imported by every relevant file but contain no task keywords.

        Args:
            relevant: Paths the LLM marked as relevant.
            file_paths: All file paths in the codebase.

        Returns:
            Merged list: relevant + newly discovered imports.
        """
        forward_map, _module_lookup = build_import_graph(
            self._source_dir, file_paths, self._config.max_file_bytes,
        )
        relevant_set = set(relevant)
        expanded: list[str] = list(relevant)
        for rel_path in relevant:
            for dep in forward_map.get(rel_path, set()):
                if dep not in relevant_set:
                    relevant_set.add(dep)
                    expanded.append(dep)
        return expanded

    @staticmethod
    def _prioritize_for_summary(paths: list[str]) -> list[str]:
        """Sort paths so code files come before docs/tests/examples.

        Priority tiers (lower = summarized first):
          0: source code (.py under the main package)
          1: config/build files (.yaml, .toml, .cfg, etc.)
          2: tests
          3: everything else (docs, examples, tools, .md, etc.)

        Within each tier, original order is preserved.
        """
        _DOC_DIRS = {"docs", "examples", "tools", ".fitz-graveyard", ".github"}
        _TEST_DIRS = {"tests", "test"}

        def _tier(p: str) -> int:
            first_dir = p.split("/")[0] if "/" in p else ""
            if first_dir in _TEST_DIRS:
                return 2
            if first_dir in _DOC_DIRS or p.endswith(".md"):
                return 3
            if p.endswith(".py"):
                return 0
            return 1

        return sorted(paths, key=_tier)

    # ------------------------------------------------------------------
    # Pass 4: Read raw source
    # ------------------------------------------------------------------

    def _read_raw_source(
        self,
        selected: list[str],
        all_paths: list[str],
    ) -> tuple[str, list[str], dict[str, set[str]], dict[str, int]]:
        """Read actual source code of selected files into a context string.

        Files are sorted by import connectivity (most-connected first)
        so that architecturally central files survive budget truncation.

        Returns:
            (raw_source_string, included_paths, forward_map, reverse_count)
        """
        # Compute import connectivity for prioritization
        forward_map, _ = build_import_graph(
            self._source_dir, all_paths, self._config.max_file_bytes,
        )
        reverse_count: dict[str, int] = {}
        for deps in forward_map.values():
            for dep in deps:
                reverse_count[dep] = reverse_count.get(dep, 0) + 1

        def _connectivity(path: str) -> int:
            return len(forward_map.get(path, set())) + reverse_count.get(path, 0)

        # Sort: most-connected first (survives truncation)
        sorted_files = sorted(selected, key=_connectivity, reverse=True)

        budget = self._config.max_context_chars
        blocks: list[str] = []
        included: list[str] = []
        used = 0

        for rel_path in sorted_files:
            try:
                resolved = sanitize_agent_path(rel_path, self._source_dir)
            except ValueError:
                continue
            if not resolved.is_file():
                continue
            try:
                raw = resolved.read_bytes()[:self._config.max_file_bytes]
                content = raw.decode("utf-8", errors="replace")
            except OSError:
                continue
            if not content.strip():
                continue

            block = f"### {rel_path}\n```\n{content}\n```"
            if used + len(block) > budget:
                logger.info(
                    f"AgentContextGatherer: budget exhausted at "
                    f"{len(included)} files ({used} chars), "
                    f"dropping {len(sorted_files) - len(included)} remaining"
                )
                break

            blocks.append(block)
            included.append(rel_path)
            used += len(block)

        # Prepend interface signatures cheat sheet
        signatures = extract_interface_signatures(
            self._source_dir, included, self._config.max_file_bytes,
        )
        if signatures:
            header = (
                "--- INTERFACE SIGNATURES (auto-extracted, ground truth) ---\n"
                + signatures
            )
            blocks.insert(0, header)

        return "\n\n".join(blocks), included, forward_map, reverse_count

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
