# fitz_graveyard/planning/agent/gatherer.py
"""
AgentContextGatherer — multi-pass pipeline to explore the codebase
and produce a structured markdown context document.

Pipeline:
  Pass 1:  Map        (pure Python — pathlib walk, build file list)
  Pass 2:  Index      (pure Python — structural extraction per file)
  Pass 3:  Match      (pure Python — keyword match task against index content)
  Pass 4:  Expand     (pure Python — BFS import graph both directions)
  Pass 5a: Filter     (1 LLM call — LLM judges relevance of graph candidates)
  Pass 5b: Scan       (1 LLM call — LLM finds graph-unreachable files)
  Pass 6:  Summarize  (N LLM calls — one per selected file)
  Pass 7:  Synthesize (1 LLM call — combine summaries into context doc)

Two parallel selection strategies fused:
  - Graph path (passes 3-5a): deterministic keyword+import selection
  - LLM scan (pass 5b): catches architecturally relevant files with
    no import connection to seeds (data classes, protocols, utilities)
"""

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from fitz_graveyard.planning.agent.indexer import (
    INDEXABLE_EXTENSIONS,
    build_import_graph,
    build_structural_index,
)
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.validation.sanitize import sanitize_agent_path

if TYPE_CHECKING:
    from fitz_graveyard.config.schema import AgentConfig

logger = logging.getLogger(__name__)

_MAX_TREE_FILES = 2000

# Directories that are never source code in any project.
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}

# Extensions that are actual source code (not docs/configs).
_SOURCE_EXTS = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".rb",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt", ".scala",
    ".pyx", ".pxd",
})

# Directories unlikely to contain task-relevant source code.
_NON_SOURCE_DIRS = frozenset({
    "docs", "doc", ".github", ".fitz-graveyard", ".planning",
})

# Words too generic to be useful for keyword matching.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "not", "no", "but", "if", "so", "as", "it",
    "i", "me", "my", "we", "our", "you", "your", "this", "that",
    "do", "does", "did", "has", "have", "had", "can", "could",
    "will", "would", "should", "may", "might", "must",
    "add", "create", "build", "make", "implement", "write",
    "want", "need", "like", "how", "what", "when", "where",
    "new", "use", "using", "used", "all", "each", "every",
    "also", "just", "into", "about", "than", "then", "them",
    "its", "other", "some", "such", "only", "over", "after",
})


class AgentContextGatherer:
    """
    Multi-pass pipeline to gather codebase context.

    Passes 1-4 are pure Python (deterministic). The LLM is only used
    for relevance filtering (pass 5), summarization (pass 6), and
    synthesis (pass 7).
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

        Returns:
            Dict with "synthesized" and "raw_summaries". Both "" on failure.
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

            # Pass 2: Index (pure Python)
            await self._report(progress_callback, 0.062, "agent:indexing")
            structural_index = self._build_index(file_paths)
            logger.info(
                f"AgentContextGatherer: indexed {len(file_paths)} files "
                f"({len(structural_index)} chars)"
            )

            # Pass 3: Keyword match (pure Python — deterministic seed selection)
            await self._report(progress_callback, 0.064, "agent:matching")
            seed_cap = min(10, self._config.max_summary_files)
            seeds = self._keyword_match(
                structural_index, job_description, seed_cap,
            )
            if not seeds:
                logger.info("AgentContextGatherer: no keyword matches")
                return empty

            logger.info(
                f"AgentContextGatherer: keyword matched {len(seeds)} seeds: "
                + ", ".join(seeds)
            )

            # Pass 4: Graph expansion (pure Python)
            await self._report(
                progress_callback, 0.066, "agent:expanding_graph"
            )
            candidates = self._expand_graph(seeds, file_paths)
            logger.info(
                f"AgentContextGatherer: graph expansion found "
                f"{len(candidates)} candidates from {len(seeds)} seeds"
            )

            # Pass 5a: Filter graph candidates (1 LLM call)
            if candidates:
                await self._report(
                    progress_callback, 0.068, "agent:filtering"
                )
                filtered = await self._filter_candidates(
                    client, model, seeds, candidates,
                    job_description,
                )
            else:
                filtered = []

            # Pass 5b: LLM scan for graph-unreachable files (1 LLM call)
            already_found = set(seeds) | set(filtered)
            await self._report(
                progress_callback, 0.072, "agent:scanning"
            )
            scanned = await self._scan_index(
                client, model, structural_index,
                job_description, already_found,
            )

            selected = seeds + filtered + scanned

            logger.info(
                f"AgentContextGatherer: selected {len(selected)} files "
                f"({len(seeds)} seeds + {len(filtered)} filtered "
                f"+ {len(scanned)} scanned): " + ", ".join(selected)
            )

            # Cap at max_summary_files for summarization (the expensive part)
            if len(selected) > self._config.max_summary_files:
                logger.info(
                    f"AgentContextGatherer: capping {len(selected)} selected "
                    f"to {self._config.max_summary_files} for summarization"
                )
                selected = selected[:self._config.max_summary_files]

            # Pass 6: Summarize (N LLM calls)
            summaries: list[str] = []
            for i, rel_path in enumerate(selected):
                progress = 0.069 + 0.019 * ((i + 1) / len(selected))
                await self._report(
                    progress_callback, progress,
                    f"agent:summarizing:{rel_path}",
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

            logger.info(
                f"AgentContextGatherer: final selection "
                f"({len(selected)} files, {len(summaries)} summaries): "
                + ", ".join(selected)
            )

            # Pass 7: Synthesize (1 LLM call)
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
                    "seeds": seeds,
                    "graph_candidates": [c[0] for c in candidates],
                    "filtered": filtered,
                    "scanned": scanned,
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
    # Pass 2: Index
    # ------------------------------------------------------------------

    def _build_index(self, file_paths: list[str]) -> str:
        """Build structural index of all files using pure Python extraction."""
        return build_structural_index(
            self._source_dir, file_paths, self._config.max_file_bytes,
        )

    # ------------------------------------------------------------------
    # Pass 3: Keyword match (deterministic seed selection)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract meaningful keywords from text for matching."""
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text.lower())
        return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}

    @staticmethod
    def _parse_structural_index(index_text: str) -> dict[str, str]:
        """Parse structural index into per-file lookup.

        Returns dict mapping file path -> structural info text.
        """
        lookup: dict[str, str] = {}
        current_path: str | None = None
        current_lines: list[str] = []

        for line in index_text.splitlines():
            if line.startswith("## "):
                if current_path is not None:
                    lookup[current_path] = "\n".join(current_lines)
                current_path = line[3:].strip()
                current_lines = []
            elif current_path is not None:
                if line.strip():
                    current_lines.append(line)

        if current_path is not None:
            lookup[current_path] = "\n".join(current_lines)

        return lookup

    def _keyword_match(
        self,
        structural_index: str,
        job_description: str,
        max_seeds: int,
    ) -> list[str]:
        """Score every file in the structural index against task keywords.

        Matches keywords against file path components AND structural content
        (class names, function names, imports, exports). Returns top-scoring
        files, excluding test files unless the task mentions testing.

        Pure Python, deterministic, instant.
        """
        keywords = self._extract_keywords(job_description)
        if not keywords:
            return []

        task_mentions_tests = any(
            w in keywords for w in ("test", "tests", "testing")
        )

        # Parse index into per-file entries
        file_lookup = self._parse_structural_index(structural_index)

        scored: list[tuple[str, int]] = []
        for path, info in file_lookup.items():
            # Seeds must be source code, not docs/configs
            ext = PurePosixPath(path).suffix.lower()
            if ext not in _SOURCE_EXTS:
                continue

            # Skip non-source directories
            first_dir = path.split("/")[0] if "/" in path else ""
            if first_dir in _NON_SOURCE_DIRS:
                continue

            # Skip test files unless task is about testing
            if not task_mentions_tests and (
                path.startswith("tests/")
                or path.startswith("test/")
                or "/test_" in path
                or "/tests/" in path
            ):
                continue

            # Score against path + structural content
            searchable = (path + " " + info).lower()
            score = sum(1 for kw in keywords if kw in searchable)
            if score > 0:
                scored.append((path, score))

        # Sort by score desc, then path depth asc (prefer central files)
        scored.sort(key=lambda x: (-x[1], x[0].count("/"), x[0]))

        # Validate paths exist
        root = Path(self._source_dir).resolve()
        valid: list[str] = []
        for path, _score in scored:
            if len(valid) >= max_seeds:
                break
            full = root / path
            if full.is_file():
                try:
                    full.resolve().relative_to(root)
                    valid.append(path)
                except ValueError:
                    continue

        return valid

    # ------------------------------------------------------------------
    # Pass 4: Graph expansion (BFS both directions)
    # ------------------------------------------------------------------

    def _expand_graph(
        self,
        seeds: list[str],
        file_paths: list[str],
        max_depth: int = 2,
    ) -> list[tuple[str, list[str]]]:
        """Expand seeds via import graph in both directions.

        Every file within max_depth hops of a seed is a candidate —
        no artificial cap. The filter pass decides relevance.

        Returns:
            List of (candidate_path, [connection_descriptions, ...])
            sorted by depth then connection count. Excludes seeds.
        """
        forward, _module_lookup = build_import_graph(
            self._source_dir, file_paths, self._config.max_file_bytes,
        )

        # Build reverse map
        reverse: dict[str, set[str]] = {}
        for caller, targets in forward.items():
            for target in targets:
                reverse.setdefault(target, set()).add(caller)

        seed_set = set(seeds)
        # (depth, connections, seed_connection_count)
        found: dict[str, tuple[int, list[str], int]] = {}

        frontier: list[tuple[str, int]] = [(s, 0) for s in seeds]
        visited: set[str] = set(seeds)

        while frontier:
            next_frontier: list[tuple[str, int]] = []
            for src, depth in frontier:
                if depth >= max_depth:
                    continue
                next_depth = depth + 1
                src_is_seed = src in seed_set

                # Forward: files that src imports
                for dep in forward.get(src, set()):
                    if dep in seed_set:
                        continue
                    conn = f"imported by {src}"
                    if dep in found:
                        existing_depth, conns, sc = found[dep]
                        if conn not in conns:
                            conns.append(conn)
                        found[dep] = (
                            min(existing_depth, next_depth),
                            conns,
                            sc + (1 if src_is_seed else 0),
                        )
                    else:
                        found[dep] = (
                            next_depth, [conn],
                            1 if src_is_seed else 0,
                        )
                    if dep not in visited:
                        visited.add(dep)
                        next_frontier.append((dep, next_depth))

                # Reverse: files that import src
                for caller in reverse.get(src, set()):
                    if caller in seed_set:
                        continue
                    conn = f"imports {src}"
                    if caller in found:
                        existing_depth, conns, sc = found[caller]
                        if conn not in conns:
                            conns.append(conn)
                        found[caller] = (
                            min(existing_depth, next_depth),
                            conns,
                            sc + (1 if src_is_seed else 0),
                        )
                    else:
                        found[caller] = (
                            next_depth, [conn],
                            1 if src_is_seed else 0,
                        )
                    if caller not in visited:
                        visited.add(caller)
                        next_frontier.append((caller, next_depth))

            frontier = next_frontier

        if not found:
            return []

        # Sort by: depth asc, seed connections desc, total connections desc
        result = [
            (path, conns)
            for path, (depth, conns, seed_conns) in sorted(
                found.items(),
                key=lambda item: (
                    item[1][0],           # depth asc
                    -item[1][2],          # seed connections desc
                    -len(item[1][1]),      # total connections desc
                    item[0],              # path alphabetical
                ),
            )
            # Only source code files, skip tests and non-source dirs
            if PurePosixPath(path).suffix.lower() in _SOURCE_EXTS
            and not path.startswith("tests/")
            and not path.startswith("test/")
            and path.split("/")[0] not in _NON_SOURCE_DIRS
        ]

        return result

    # ------------------------------------------------------------------
    # Pass 5: Filter candidates (1 LLM call)
    # ------------------------------------------------------------------

    async def _filter_candidates(
        self,
        client: Any,
        model: str,
        seeds: list[str],
        candidates: list[tuple[str, list[str]]],
        job_description: str,
    ) -> list[str]:
        """Filter graph candidates using LLM relevance judgment.

        Falls back to top candidates by connection count on failure.
        No artificial budget — the LLM picks all files it deems relevant.
        """
        entries: list[str] = []
        for path, connections in candidates:
            conn_str = "; ".join(connections)
            entries.append(f"- {path}  [{conn_str}]")

        prompt = load_prompt("agent_filter_candidates").format(
            job_description=job_description,
            seed_list="\n".join(f"- {s}" for s in seeds),
            candidate_entries="\n\n".join(entries),
        )

        try:
            response = await client.generate(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0,
            )
            logger.info(
                f"AgentContextGatherer: filter raw response "
                f"({len(response)} chars): {response[:500]}"
            )
            selected = self._parse_file_list(response)
        except Exception as e:
            logger.warning(
                f"AgentContextGatherer: filter LLM failed: {e}"
            )
            selected = None

        if not selected:
            logger.info(
                "AgentContextGatherer: filter fallback — "
                "using top candidates by connection count"
            )
            selected = [path for path, _ in candidates]

        root = Path(self._source_dir).resolve()
        seed_set = set(seeds)
        valid: list[str] = []
        for rel_path in selected:
            if rel_path in seed_set:
                continue
            full = root / rel_path
            if full.is_file():
                try:
                    full.resolve().relative_to(root)
                    valid.append(rel_path)
                except ValueError:
                    continue

        logger.info(
            f"AgentContextGatherer: filtered to {len(valid)} files: "
            + ", ".join(valid)
        )
        return valid

    # ------------------------------------------------------------------
    # Pass 5b: LLM scan for graph-unreachable files
    # ------------------------------------------------------------------

    async def _scan_index(
        self,
        client: Any,
        model: str,
        structural_index: str,
        job_description: str,
        already_found: set[str],
    ) -> list[str]:
        """Scan structural index for files the import graph can't reach.

        The graph approach finds import-connected files. This pass catches
        architecturally relevant files with no import link to seeds —
        e.g. data classes, utility patterns, protocols.

        No artificial budget — the LLM picks all files it deems relevant.
        Falls back to empty list on failure (the graph results are enough).
        """

        # Strip files already found so the LLM focuses on the gap
        lines: list[str] = []
        current_path: str | None = None
        current_lines: list[str] = []
        for line in structural_index.splitlines():
            if line.startswith("## "):
                if current_path and current_path not in already_found:
                    lines.append(f"## {current_path}")
                    lines.extend(current_lines)
                current_path = line[3:].strip()
                current_lines = []
            elif current_path is not None and line.strip():
                current_lines.append(line)
        if current_path and current_path not in already_found:
            lines.append(f"## {current_path}")
            lines.extend(current_lines)

        remaining_index = "\n".join(lines)
        if not remaining_index.strip():
            return []

        prompt = load_prompt("agent_scan_index").format(
            job_description=job_description,
            already_found="\n".join(f"- {f}" for f in sorted(already_found)),
            structural_index=remaining_index,
        )

        try:
            response = await client.generate(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0,
            )
            logger.info(
                f"AgentContextGatherer: scan raw response "
                f"({len(response)} chars): {response[:500]}"
            )
            selected = self._parse_file_list(response)
        except Exception as e:
            logger.warning(
                f"AgentContextGatherer: scan LLM failed: {e}"
            )
            return []

        if not selected:
            logger.info("AgentContextGatherer: scan returned no files")
            return []

        root = Path(self._source_dir).resolve()
        valid: list[str] = []
        for rel_path in selected:
            if rel_path in already_found:
                continue
            # Only source code
            ext = PurePosixPath(rel_path).suffix.lower()
            if ext not in _SOURCE_EXTS:
                continue
            full = root / rel_path
            if full.is_file():
                try:
                    full.resolve().relative_to(root)
                    valid.append(rel_path)
                except ValueError:
                    continue

        logger.info(
            f"AgentContextGatherer: scan found {len(valid)} additional files: "
            + ", ".join(valid)
        )
        return valid

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

    # ------------------------------------------------------------------
    # Pass 6: Summarize
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

    # ------------------------------------------------------------------
    # Pass 7: Synthesize
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
