# fitz_graveyard/planning/agent/gatherer.py
"""
AgentContextGatherer — BM25 + per-file LLM confirm screening pipeline.

Pipeline:
  Pass 1:  Map            (pure Python — pathlib walk, build file list)
  Pass 2:  BM25 screen    (pure Python — score all files, dynamic top-K)
  Pass 3:  LLM confirm    (per-file mid-model YES/NO on BM25 shortlist)
  Pass 4:  Import expand  (pure Python — BFS depth 2, both directions)
  Pass 5:  Read raw source (pure Python — stuff actual source into context)

BM25 prefilters a dynamic top-K shortlist (scales with codebase size, default
~200 for 25 max_summary_files). Per-file LLM screening confirms candidates.
Import expansion then chases imports recursively (depth 2, forward + reverse)
to catch architecturally central files that lack task keywords. Raw source is
stuffed directly into the planning context — no summarization or synthesis.
Files sorted by import connectivity; truncated at a char budget so they fit
in the reasoning model's context window.
"""

import asyncio
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from fitz_graveyard.planning.agent.indexer import (
    INDEXABLE_EXTENSIONS,
    build_import_graph,
    extract_interface_signatures,
    extract_library_signatures,
)
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.validation.sanitize import sanitize_agent_path

if TYPE_CHECKING:
    from fitz_graveyard.config.schema import AgentConfig

logger = logging.getLogger(__name__)

_MAX_TREE_FILES = 2000

# Directories that are never source code in any project.
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}

# BM25 parameters (Okapi BM25 standard defaults)
_BM25_K1 = 1.5
_BM25_B = 0.75
_PATH_BONUS_WEIGHT = 2.0

# English stopwords (compact set for tokenization)
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "are",
    "this", "that", "not", "can", "all", "if", "do", "my", "no", "so",
    "we", "he", "up", "one", "its", "has", "had", "may", "our", "out",
    "you", "his", "her", "she", "how", "new", "now", "old", "see",
    "way", "who", "did", "get", "let", "say", "too", "use",
    "import", "from", "def", "class", "self", "return", "none", "true",
    "false", "pass", "else", "elif", "try", "except", "finally",
    "raise", "yield", "lambda", "assert", "global", "nonlocal",
    "while", "for", "break", "continue", "del", "with", "async", "await",
})

# File extensions excluded from BM25 scoring (too text-dense, drown out code).
# These files are still indexed and can be discovered via import expansion.
_BM25_SKIP_EXTS = frozenset({".md"})

# Concurrency limit for per-file LLM screening calls
_SCREEN_CONCURRENCY = 10


class AgentContextGatherer:
    """
    BM25 + LLM confirm pipeline to gather codebase context.

    BM25 scores all files for keyword relevance (pure Python, milliseconds),
    then per-file LLM calls confirm the shortlist. Raw source of selected
    files is stuffed into context. No summarization, no synthesis.
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
        Run the BM25 + LLM confirm pipeline and return context dict.

        Returns:
            Dict with "synthesized", "raw_summaries", and "agent_files".
            Both strings "" on failure.
        """
        empty = {"synthesized": "", "raw_summaries": ""}

        if not self._config.enabled:
            logger.info("Agent context gathering disabled by config")
            return empty

        mid = self._config.agent_model or client.mid_model

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

            # Pass 2: BM25 screen (pure Python)
            await self._report(progress_callback, 0.062, "agent:screening")
            # Dynamic top-K: scale with codebase size
            # Small codebases (<100 files): take ~50% of files
            # Medium (100-500): ~30%
            # Large (500+): cap at max_summary_files * 8 (default 200)
            dynamic_k = max(
                self._config.max_summary_files * 8,
                len(file_paths) // 3,
            )
            top_k = min(dynamic_k, len(file_paths))
            bm25_candidates, bm25_scores = self._bm25_screen(
                file_paths, job_description, top_k,
            )

            if not bm25_candidates:
                logger.info("AgentContextGatherer: BM25 found no candidates")
                return empty

            logger.info(
                f"AgentContextGatherer: BM25 selected {len(bm25_candidates)} "
                f"candidates from {len(file_paths)} files "
                f"(top score: {bm25_scores[0]:.2f}, "
                f"cutoff: {bm25_scores[-1]:.2f})"
            )

            # Pass 3: Per-file LLM confirm (mid model, parallel)
            await self._report(progress_callback, 0.068, "agent:confirming")
            relevant = await self._screen_all(
                client, mid, bm25_candidates, job_description,
            )

            if not relevant:
                logger.info(
                    "AgentContextGatherer: LLM confirm rejected all, "
                    "falling back to BM25 top results"
                )
                relevant = bm25_candidates[:self._config.max_summary_files]

            logger.info(
                f"AgentContextGatherer: confirmed {len(relevant)} "
                f"relevant files: " + ", ".join(relevant)
            )

            # Pass 4: Import expansion (pure Python, BFS depth 2, both directions)
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

            # Pass 5: Read raw source (pure Python, no LLM)
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
                    "bm25_candidates": bm25_candidates,
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
    # Pass 2: BM25 screen
    # ------------------------------------------------------------------

    def _read_file_content(self, rel_path: str) -> str:
        """Read file content, returning empty string on failure."""
        try:
            resolved = sanitize_agent_path(rel_path, self._source_dir)
        except ValueError:
            return ""

        if not resolved.is_file():
            return ""

        try:
            raw = resolved.read_bytes()[:self._config.max_file_bytes]
            return raw.decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _bm25_screen(
        self,
        file_paths: list[str],
        job_description: str,
        top_k: int,
    ) -> tuple[list[str], list[float]]:
        """Score all files by BM25 relevance to job description.

        Pure Python Okapi BM25 implementation. Also applies a path bonus
        for files whose path segments match query terms.

        Returns:
            Tuple of (sorted_paths, sorted_scores) for top K candidates.
            Only includes files with score > 0.
        """
        query_terms = _tokenize(job_description)
        if not query_terms:
            return [], []

        # Build corpus: tokenize each file's content (skip text-heavy extensions)
        corpus: list[tuple[str, list[str]]] = []
        for path in file_paths:
            ext = Path(path).suffix.lower()
            if ext in _BM25_SKIP_EXTS:
                continue
            content = self._read_file_content(path)
            if not content.strip():
                continue
            tokens = _tokenize(content)
            # Add path segments as bonus tokens
            path_tokens = _tokenize(path.replace("/", " ").replace(".", " "))
            tokens.extend(path_tokens * int(_PATH_BONUS_WEIGHT))
            corpus.append((path, tokens))

        if not corpus:
            return [], []

        # Compute document lengths and average
        doc_lengths = [len(tokens) for _, tokens in corpus]
        avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0
        n_docs = len(corpus)

        # Compute IDF for query terms
        query_set = set(query_terms)
        doc_freq: Counter[str] = Counter()
        for _, tokens in corpus:
            present = query_set & set(tokens)
            for term in present:
                doc_freq[term] += 1

        idf: dict[str, float] = {}
        for term in query_set:
            df = doc_freq.get(term, 0)
            # IDF with smoothing (prevents negative values)
            idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        # Score each document
        scores: list[tuple[str, float]] = []
        for i, (path, tokens) in enumerate(corpus):
            tf_counter = Counter(tokens)
            dl = doc_lengths[i]
            score = 0.0
            for term in query_terms:
                tf = tf_counter.get(term, 0)
                if tf == 0:
                    continue
                numerator = tf * (_BM25_K1 + 1)
                denominator = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avgdl)
                score += idf.get(term, 0) * numerator / denominator
            scores.append((path, score))

        # Sort by score descending, filter zero scores
        scores.sort(key=lambda x: x[1], reverse=True)
        scores = [(p, s) for p, s in scores if s > 0]

        # Take top K
        top = scores[:top_k]
        if not top:
            return [], []

        return [p for p, _ in top], [s for _, s in top]

    # ------------------------------------------------------------------
    # Pass 3: Per-file LLM confirm (batched in pairs)
    # ------------------------------------------------------------------

    async def _screen_batch(
        self,
        client: Any,
        model: str,
        batch: list[str],
        job_description: str,
    ) -> list[tuple[str, bool]]:
        """Screen a batch of files in one LLM call.

        Returns list of (path, is_relevant) tuples.
        On failure, all files in the batch are treated as not relevant.
        """
        file_blocks: list[str] = []
        valid_paths: list[str] = []
        for rel_path in batch:
            content = self._read_file_content(rel_path)
            if not content.strip():
                continue
            file_blocks.append(
                f"FILE: {rel_path}\n\nCONTENT:\n{content}"
            )
            valid_paths.append(rel_path)

        if not valid_paths:
            return [(p, False) for p in batch]

        prompt = load_prompt("agent_screen").format(
            job_description=job_description,
            file_blocks="\n\n---\n\n".join(file_blocks),
        )

        try:
            response = await client.generate(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0,
            )
            verdicts = self._parse_batch_response(response, valid_paths)
            return verdicts
        except Exception:
            return [(p, False) for p in batch]

    async def _screen_all(
        self,
        client: Any,
        model: str,
        candidates: list[str],
        job_description: str,
    ) -> list[str]:
        """Screen all candidates in parallel batches of 2."""
        sem = asyncio.Semaphore(_SCREEN_CONCURRENCY)
        batches = [
            candidates[i:i + 2]
            for i in range(0, len(candidates), 2)
        ]

        async def _bounded(batch: list[str]) -> list[tuple[str, bool]]:
            async with sem:
                return await self._screen_batch(
                    client, model, batch, job_description,
                )

        tasks = [_bounded(b) for b in batches]
        batch_results = await asyncio.gather(*tasks)
        return [
            path
            for results in batch_results
            for path, relevant in results
            if relevant
        ]

    @staticmethod
    def _parse_batch_response(
        response: str, paths: list[str],
    ) -> list[tuple[str, bool]]:
        """Parse multi-line batch response into per-file verdicts.

        Expected format: "file_path: YES" or "file_path: NO" per line.
        Falls back to scanning for YES/NO by line position.
        """
        lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
        verdicts: dict[str, bool] = {}

        # Try parsing "path: YES/NO" format
        for line in lines:
            for path in paths:
                if path in line:
                    upper = line.upper()
                    if "YES" in upper:
                        verdicts[path] = True
                    elif "NO" in upper:
                        verdicts[path] = False
                    break

        # Fall back to positional matching if we didn't get all paths
        if len(verdicts) < len(paths):
            yes_no_lines = [
                ln for ln in lines
                if ln.strip().upper().startswith(("YES", "NO"))
                or ": YES" in ln.upper()
                or ": NO" in ln.upper()
            ]
            for i, path in enumerate(paths):
                if path not in verdicts and i < len(yes_no_lines):
                    verdicts[path] = "YES" in yes_no_lines[i].upper()

        return [(p, verdicts.get(p, False)) for p in paths]

    # ------------------------------------------------------------------
    # Pass 4: Import expansion
    # ------------------------------------------------------------------

    def _import_expand(
        self, relevant: list[str], file_paths: list[str],
    ) -> list[str]:
        """Recursive import expansion (BFS, depth 2, both directions).

        Traces forward imports ("file A imports B") and reverse imports
        ("B is imported by A") from each relevant file. This catches
        architecturally central files (e.g. base.py, protocols) that
        are imported by relevant files but contain no task keywords,
        AND files that depend on relevant files.

        Args:
            relevant: Paths the LLM marked as relevant.
            file_paths: All file paths in the codebase.

        Returns:
            Merged list: relevant + newly discovered imports.
        """
        forward_map, _module_lookup = build_import_graph(
            self._source_dir, file_paths, self._config.max_file_bytes,
        )

        # Build reverse map: {file → set of files that import it}
        reverse_map: dict[str, set[str]] = {}
        for src, deps in forward_map.items():
            for dep in deps:
                reverse_map.setdefault(dep, set()).add(src)

        # BFS both directions, depth 2
        relevant_set = set(relevant)
        frontier = set(relevant)
        for depth in range(2):
            next_frontier: set[str] = set()
            for rel_path in frontier:
                # Forward: files this file imports
                for dep in forward_map.get(rel_path, set()):
                    if dep not in relevant_set:
                        relevant_set.add(dep)
                        next_frontier.add(dep)
                # Reverse: files that import this file
                for importer in reverse_map.get(rel_path, set()):
                    if importer not in relevant_set:
                        relevant_set.add(importer)
                        next_frontier.add(importer)
            frontier = next_frontier
            if not frontier:
                break
            logger.info(
                f"AgentContextGatherer: import expand depth {depth + 1} "
                f"added {len(frontier)} files"
            )

        return list(relevant) + sorted(relevant_set - set(relevant))

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
    # Pass 5: Read raw source
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

        # Prepend library API reference (after interface signatures)
        lib_sigs = extract_library_signatures(
            self._source_dir, included, all_paths, self._config.max_file_bytes,
        )
        if lib_sigs:
            lib_header = (
                "--- LIBRARY API REFERENCE (installed packages, ground truth) ---\n"
                + lib_sigs
            )
            # Insert after interface signatures (position 1) or at 0
            insert_pos = 1 if signatures else 0
            blocks.insert(insert_pos, lib_header)

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


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, split on non-alphanumeric, remove stopwords."""
    tokens = re.findall(r"[a-z][a-z0-9_]*", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
