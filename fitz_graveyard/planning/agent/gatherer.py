# fitz_graveyard/planning/agent/gatherer.py
"""
AgentContextGatherer — Multi-signal retrieval pipeline.

Pipeline:
  Pass 1:  Map              (pure Python — pathlib walk, build file list)
  Pass 2:  Query expand      (LLM — generate search terms + HyDE code)
  Pass 3:  Structural scan   (LLM — review structural index for relevant files)
  Pass 4:  BM25 screen       (pure Python — expanded query against all files)
  Pass 5:  Embedding recall   (sentence-transformers — semantic similarity)
  Pass 6:  Cross-encoder rerank (sentence-transformers — rerank merged candidates)
  Pass 7:  Import expand      (pure Python — BFS depth 2, both directions)
  Pass 8:  Neighbor expand    (pure Python — same-directory files)
  Pass 9:  Read raw source    (pure Python — stuff into context)

Two LLM calls total (query expand, structural scan).
Structural scan hits are protected — they always survive to final selection.
Embedding and reranking use lightweight in-process models (~360MB VRAM total).
Falls back to BM25 if sentence-transformers is unavailable.
"""

import json
import logging
import math
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from fitz_graveyard.planning.agent.indexer import (
    INDEXABLE_EXTENSIONS,
    build_import_graph,
    build_structural_index,
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
_BM25_SKIP_EXTS = frozenset({".md"})

# Maximum chars of file content for embedding / reranking input
_EMBED_MAX_CHARS = 4000

# Maximum candidates sent to cross-encoder reranking
_MAX_CANDIDATES_FOR_RERANK = 200

# Cross-encoder reranking output size
_RERANK_TOP_K = 20

# Neighbor expansion: screen siblings via LLM if a directory adds more than this many
_NEIGHBOR_SCREEN_THRESHOLD = 10

# VRAM headroom (MB) required to keep LLM loaded during embedding/reranking.
# Embedding (~275MB) + reranker (~85MB) + working memory ≈ 2GB total.
# 6GB threshold gives comfortable margin for CUDA context overhead.
_VRAM_HEADROOM_MB = 6_000

# Seed-and-fetch: only a small seed set goes into the prompt.
# The rest are available via read_file/read_files tools during reasoning.
# This forces the LLM to actively explore the codebase rather than
# passively consuming a context dump.
_DEFAULT_MAX_SEED_FILES = 30



class AgentContextGatherer:
    """
    Multi-signal retrieval pipeline to gather codebase context.

    Combines BM25 keyword search, embedding-based semantic search,
    structural index scanning, cross-encoder reranking, and LLM
    judgment to select the most relevant files for a planning task.
    """

    def __init__(self, config: "AgentConfig", source_dir: str) -> None:
        self._config = config
        self._source_dir = source_dir
        self._file_cache: dict[str, str] = {}

    async def gather(
        self,
        client: Any,
        job_description: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, str]:
        """
        Run the multi-signal retrieval pipeline and return context dict.

        Returns:
            Dict with "synthesized", "raw_summaries", and "agent_files".
            Both strings "" on failure.
        """
        empty = {"synthesized": "", "raw_summaries": ""}

        if not self._config.enabled:
            logger.info("Agent context gathering disabled by config")
            return empty

        smart = self._config.agent_model or client.smart_model
        t_pipeline = time.monotonic()

        try:
            # Pass 1: Map (pure Python)
            await self._report(progress_callback, 0.06, "agent:mapping")
            t0 = time.monotonic()
            _file_tree, file_paths = self._build_file_tree()
            if not file_paths:
                logger.info("AgentContextGatherer: empty or invalid source dir")
                return empty

            # Pre-warm file cache (single pass over disk)
            for p in file_paths:
                self._read_file_content(p)

            logger.info(
                f"AgentContextGatherer: mapped {len(file_paths)} files, "
                f"cached {len(self._file_cache)} "
                f"({time.monotonic() - t0:.1f}s)"
            )

            # Pass 2: Query expansion + HyDE (1 LLM call)
            await self._report(progress_callback, 0.062, "agent:expanding_query")
            t0 = time.monotonic()
            expanded_terms, hyde_code = await self._expand_query(
                client, smart, job_description,
            )
            expanded_query = f"{job_description} {expanded_terms}"
            logger.info(
                f"AgentContextGatherer: query expanded — "
                f"{len(expanded_terms.splitlines())} terms, "
                f"{len(hyde_code)} chars HyDE "
                f"({time.monotonic() - t0:.1f}s)"
            )

            # Pass 3: Structural index scan (1 LLM call)
            await self._report(progress_callback, 0.065, "agent:scanning_index")
            t0 = time.monotonic()
            structural_index = build_structural_index(
                self._source_dir, file_paths, self._config.max_file_bytes,
            )
            scan_hits = await self._structural_scan(
                client, smart, structural_index, job_description,
            )
            # Filter scan hits to only valid paths
            valid_paths_set = set(file_paths)
            scan_hits = [p for p in scan_hits if p in valid_paths_set]
            logger.info(
                f"AgentContextGatherer: structural scan found "
                f"{len(scan_hits)} files ({time.monotonic() - t0:.1f}s)"
            )

            # Pass 4: BM25 screen with expanded query (pure Python)
            await self._report(progress_callback, 0.068, "agent:bm25")
            t0 = time.monotonic()
            bm25_top_k = min(200, len(file_paths))
            bm25_candidates, bm25_scores = self._bm25_screen(
                file_paths, expanded_query, bm25_top_k,
            )
            logger.info(
                f"AgentContextGatherer: BM25 selected "
                f"{len(bm25_candidates)} candidates "
                f"({time.monotonic() - t0:.1f}s)"
            )

            # VRAM router: if >= 6GB free, keep LLM loaded during
            # embedding/reranking (~360MB). Otherwise unload first.
            llm_unloaded = False
            unload = getattr(client, "unload_model", None)
            if unload is not None:
                from fitz_graveyard.llm.gpu_monitor import GPUTemperatureGuard
                free_mb = GPUTemperatureGuard.get_free_vram_mb()
                if free_mb is not None and free_mb >= _VRAM_HEADROOM_MB:
                    logger.info(
                        f"AgentContextGatherer: {free_mb} MB VRAM free "
                        f"(>= {_VRAM_HEADROOM_MB} MB), keeping LLM loaded"
                    )
                else:
                    reason = (
                        f"{free_mb} MB free" if free_mb is not None
                        else "VRAM unknown"
                    )
                    logger.info(
                        f"AgentContextGatherer: {reason} "
                        f"(< {_VRAM_HEADROOM_MB} MB), unloading LLM"
                    )
                    await self._report(
                        progress_callback, 0.069, "agent:unloading_llm",
                    )
                    llm_unloaded = await unload()

            # Pass 5: Embedding recall (if sentence-transformers available)
            embedding_candidates: list[str] = []
            try:
                await self._report(progress_callback, 0.070, "agent:embedding")
                t0 = time.monotonic()
                embedding_candidates = self._embedding_recall(
                    file_paths, job_description, hyde_code,
                    top_k=100,
                )
                logger.info(
                    f"AgentContextGatherer: embedding recall found "
                    f"{len(embedding_candidates)} candidates "
                    f"({time.monotonic() - t0:.1f}s)"
                )
            except ImportError:
                logger.info(
                    "AgentContextGatherer: sentence-transformers not installed, "
                    "skipping embedding recall"
                )
            except MemoryError as e:
                logger.warning(
                    f"AgentContextGatherer: skipping embedding — {e}"
                )
            except Exception:
                logger.warning(
                    "AgentContextGatherer: embedding recall failed",
                    exc_info=True,
                )

            # Merge candidates from all signals
            merged = self._merge_candidates(
                bm25_candidates, embedding_candidates, scan_hits, file_paths,
            )
            logger.info(
                f"AgentContextGatherer: merged {len(merged)} unique candidates "
                f"(BM25={len(bm25_candidates)}, embed={len(embedding_candidates)}, "
                f"scan={len(scan_hits)})"
            )

            # Pass 6: Cross-encoder rerank (if sentence-transformers available)
            reranked = merged
            try:
                if len(merged) > _RERANK_TOP_K:
                    await self._report(
                        progress_callback, 0.073, "agent:reranking",
                    )
                    t0 = time.monotonic()
                    reranked = self._rerank_candidates(
                        merged, job_description, top_k=_RERANK_TOP_K,
                    )
                    logger.info(
                        f"AgentContextGatherer: reranked to "
                        f"{len(reranked)} files "
                        f"({time.monotonic() - t0:.1f}s)"
                    )
            except ImportError:
                logger.info(
                    "AgentContextGatherer: sentence-transformers not installed, "
                    "skipping reranking"
                )
            except MemoryError as e:
                logger.warning(
                    f"AgentContextGatherer: skipping reranking — {e}"
                )
            except Exception:
                logger.warning(
                    "AgentContextGatherer: reranking failed",
                    exc_info=True,
                )

            # Reload LLM if it was unloaded
            if llm_unloaded:
                await self._report(progress_callback, 0.075, "agent:reloading_llm")
                reload = getattr(client, "reload_model", None)
                if reload is not None:
                    await reload()

            # Build import graph once (reused by import expansion + read raw source)
            forward_map, _module_lookup = build_import_graph(
                self._source_dir, file_paths, self._config.max_file_bytes,
            )

            # Pass 7: Import expansion (scan hits only — reranked files are
            # noisy keyword matches whose imports cascade into irrelevant code)
            scan_expanded = self._import_expand(scan_hits, file_paths, forward_map)
            import_added = len(scan_expanded) - len(scan_hits)
            if import_added > 0:
                logger.info(
                    f"AgentContextGatherer: import expansion added "
                    f"{import_added} files (from {len(scan_hits)} scan hits)"
                )
            # Merge: scan hits + their imports first (high-confidence), then reranked
            expanded_set = set(scan_expanded)
            expanded = list(scan_expanded)
            for path in reranked:
                if path not in expanded_set:
                    expanded_set.add(path)
                    expanded.append(path)
            logger.info(
                f"AgentContextGatherer: combined {len(expanded)} candidates "
                f"(scan+imports={len(scan_expanded)}, reranked={len(reranked)})"
            )

            # Pass 8: Neighbor expansion (pure Python, same-directory files)
            # Only expand from high-confidence sources: scan hits (LLM-picked
            # from structural index) and files import-reachable from scan hits
            # (structurally connected). BM25/embedding-only hits are noisy
            # keyword matches that would expand irrelevant directories.
            await self._report(progress_callback, 0.078, "agent:neighbor_expand")
            high_confidence = self._import_reachable(
                scan_hits, forward_map, set(expanded),
            )
            neighbor_expanded = self._neighbor_expand(
                expanded, file_paths, expand_from=high_confidence,
            )
            neighbor_added = len(neighbor_expanded) - len(expanded)
            if neighbor_added > 0:
                logger.info(
                    f"AgentContextGatherer: neighbor expansion added "
                    f"{neighbor_added} files (from {len(high_confidence)} "
                    f"high-confidence triggers)"
                )

            # Pass 8b: Screen large-directory neighbors (1 LLM call per large dir)
            screened = await self._screen_neighbors(
                client, smart, job_description, expanded, neighbor_expanded,
            )
            screen_removed = len(neighbor_expanded) - len(screened)
            if screen_removed > 0:
                logger.info(
                    f"AgentContextGatherer: neighbor screening removed "
                    f"{screen_removed} irrelevant siblings"
                )

            selected = self._prioritize_for_summary(screened)

            # Pass 9: Read selected files + compress for planning context
            await self._report(progress_callback, 0.080, "agent:reading")
            file_contents, included = self._read_selected_files(selected)

            if not file_contents:
                logger.warning("AgentContextGatherer: no readable source files")
                return empty

            # Build structural overview (compact, covers ALL selected files)
            selected_index = build_structural_index(
                self._source_dir, included, self._config.max_file_bytes,
            )
            signatures = extract_interface_signatures(
                self._source_dir, included, self._config.max_file_bytes,
            )
            lib_sigs = extract_library_signatures(
                self._source_dir, included, file_paths,
                self._config.max_file_bytes,
            )

            overview_parts: list[str] = []
            if signatures:
                overview_parts.append(
                    "--- INTERFACE SIGNATURES (auto-extracted, ground truth) ---\n"
                    + signatures
                )
            if lib_sigs:
                overview_parts.append(
                    "--- LIBRARY API REFERENCE (installed packages, ground truth) ---\n"
                    + lib_sigs
                )
            overview_parts.append(
                "--- STRUCTURAL OVERVIEW (all selected files) ---\n"
                + selected_index
            )
            structural_overview = "\n\n".join(overview_parts)

            # Compress file contents for planning context.
            # AST-based: strips docstrings, comments, collapses long bodies.
            # Structural index was built from raw disk files (unaffected).
            from fitz_graveyard.planning.agent.compressor import compress_file

            raw_chars = sum(len(c) for c in file_contents.values())
            file_contents = {
                path: compress_file(content, path)
                for path, content in file_contents.items()
            }
            comp_chars = sum(len(c) for c in file_contents.values())
            logger.info(
                f"AgentContextGatherer: compressed {len(file_contents)} files "
                f"({raw_chars} -> {comp_chars} chars, "
                f"{100 * (1 - comp_chars / raw_chars):.0f}% reduction)"
            )

            # Seed-and-fetch: include only a small seed set in the prompt.
            # The LLM gets the full structural overview (signatures + index)
            # plus seed files as full source. Remaining files are available
            # via read_file/read_files tools — the LLM must actively explore.
            max_seeds = getattr(self._config, "max_seed_files", _DEFAULT_MAX_SEED_FILES)

            # Structural overview is always included (compact, ground truth)
            raw_summaries = structural_overview

            # Priority order: scan hits first (LLM-picked from index),
            # then rest of included files in retrieval order
            scan_set = set(scan_hits)
            priority_files: list[str] = list(scan_hits)
            for path in included:
                if path not in scan_set:
                    priority_files.append(path)

            seed_files: list[str] = []
            tool_pool_files: list[str] = []
            included_in_prompt: list[str] = []

            for path in priority_files:
                if path not in file_contents:
                    continue
                if len(seed_files) < max_seeds:
                    seed_files.append(path)
                    block = f"### {path}\n```\n{file_contents[path]}\n```"
                    included_in_prompt.append(block)
                else:
                    tool_pool_files.append(path)

            if included_in_prompt:
                raw_summaries += (
                    f"\n\n--- SEED FILES ({len(seed_files)}/{len(included)} — "
                    f"use read_file/read_files for the rest) ---\n\n"
                    + "\n\n".join(included_in_prompt)
                )

            seed_chars = sum(len(file_contents.get(p, "")) for p in seed_files)
            pool_chars = sum(len(file_contents.get(p, "")) for p in tool_pool_files)
            logger.info(
                f"AgentContextGatherer: seed={len(seed_files)} files "
                f"({seed_chars} chars), tool_pool={len(tool_pool_files)} files "
                f"({pool_chars} chars), max_seeds={max_seeds}"
            )

            # Compute reverse import counts for diagnostics
            reverse_count: dict[str, int] = {}
            for deps in forward_map.values():
                for dep in deps:
                    reverse_count[dep] = reverse_count.get(dep, 0) + 1

            # Build per-file provenance for traceability.
            # For each included file, record which retrieval signals found it
            # and whether it's a seed file (in prompt) or tool-pool (deferred).
            scan_set = set(scan_hits)
            bm25_set = set(bm25_candidates)
            embed_set = set(embedding_candidates)
            rerank_set = set(reranked)
            import_set = set(scan_expanded) - scan_set
            neighbor_set = set(neighbor_expanded) - set(expanded)
            seed_set = set(seed_files)

            file_provenance: dict[str, dict] = {}
            for path in included:
                signals: list[str] = []
                if path in scan_set:
                    signals.append("scan")
                if path in bm25_set:
                    signals.append("bm25")
                if path in embed_set:
                    signals.append("embed")
                if path in rerank_set:
                    signals.append("rerank")
                if path in import_set:
                    signals.append("import")
                if path in neighbor_set:
                    signals.append("neighbor")
                file_provenance[path] = {
                    "signals": signals,
                    "in_prompt": path in seed_set,
                }

            t_total = time.monotonic() - t_pipeline
            logger.info(
                f"AgentContextGatherer: {len(included)} files "
                f"({comp_chars} chars compressed, "
                f"{len(structural_overview)} chars overview) — "
                f"pipeline total {t_total:.1f}s"
            )
            serializable_fwd = {k: sorted(v) for k, v in forward_map.items()}
            return {
                "synthesized": structural_overview,
                "raw_summaries": raw_summaries,
                "file_contents": file_contents,
                "agent_files": {
                    "total_screened": len(file_paths),
                    "bm25_candidates": bm25_candidates[:20],
                    "scan_hits": scan_hits,
                    "embedding_candidates": embedding_candidates[:20],
                    "reranked": reranked[:20],
                    "selected": selected,
                    "included": included,
                    "forward_map": serializable_fwd,
                    "reverse_count": reverse_count,
                    "file_provenance": file_provenance,
                },
            }

        except Exception:
            logger.exception("AgentContextGatherer: pipeline failed")
            return empty
        finally:
            self._file_cache.clear()

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
        return name in _SKIP_DIRS or name.startswith(".")

    # ------------------------------------------------------------------
    # Pass 2: Query expansion + HyDE
    # ------------------------------------------------------------------

    async def _expand_query(
        self,
        client: Any,
        model: str,
        job_description: str,
    ) -> tuple[str, str]:
        """LLM query expansion: generate search terms + hypothetical code.

        Returns (terms_text, hyde_code). Empty strings on failure.
        """
        prompt = load_prompt("agent_expand").format(
            job_description=job_description,
        )
        try:
            response = await client.generate(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0,
            )
            return self._parse_expand_response(response)
        except Exception:
            logger.warning(
                "AgentContextGatherer: query expansion failed", exc_info=True,
            )
            return "", ""

    # ------------------------------------------------------------------
    # Pass 3: Structural index scan
    # ------------------------------------------------------------------

    async def _structural_scan(
        self,
        client: Any,
        model: str,
        structural_index: str,
        job_description: str,
    ) -> list[str]:
        """LLM scan of structural index to find relevant files.

        Returns list of file paths. Empty list on failure.
        """
        prompt = load_prompt("agent_scan").format(
            job_description=job_description,
            structural_index=structural_index,
        )
        try:
            response = await client.generate(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0,
            )
            files = self._parse_file_list(response)
            return files if files is not None else []
        except Exception:
            logger.warning(
                "AgentContextGatherer: structural scan failed", exc_info=True,
            )
            return []

    # ------------------------------------------------------------------
    # Pass 4: BM25 screen
    # ------------------------------------------------------------------

    def _read_file_content(self, rel_path: str) -> str:
        """Read file content with caching, returning empty string on failure."""
        cached = self._file_cache.get(rel_path)
        if cached is not None:
            return cached

        try:
            resolved = sanitize_agent_path(rel_path, self._source_dir)
        except ValueError:
            self._file_cache[rel_path] = ""
            return ""

        if not resolved.is_file():
            self._file_cache[rel_path] = ""
            return ""

        try:
            raw = resolved.read_bytes()[:self._config.max_file_bytes]
            content = raw.decode("utf-8", errors="replace")
        except OSError:
            content = ""

        self._file_cache[rel_path] = content
        return content

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
    # Pass 5: Embedding recall
    # ------------------------------------------------------------------

    def _embedding_recall(
        self,
        file_paths: list[str],
        job_description: str,
        hyde_code: str,
        top_k: int = 100,
    ) -> list[str]:
        """Semantic file retrieval using sentence-transformers embeddings.

        Encodes query + HyDE code and all file contents, returns top-K
        by cosine similarity. Raises ImportError if sentence-transformers
        is not installed.
        """
        import numpy as np

        from fitz_graveyard.planning.agent.models import EmbeddingModel

        embedder = EmbeddingModel()
        try:
            embedder.load(self._config.embedding_model)

            is_nomic = "nomic" in self._config.embedding_model.lower()
            q_prefix = "search_query: " if is_nomic else ""
            d_prefix = "search_document: " if is_nomic else ""

            # Encode queries (original + HyDE)
            queries = [f"{q_prefix}{job_description}"]
            if hyde_code:
                queries.append(f"{q_prefix}{hyde_code[:_EMBED_MAX_CHARS]}")
            query_embs = embedder.encode(queries)

            # Encode all file contents
            doc_texts = []
            valid_paths = []
            for path in file_paths:
                content = self._read_file_content(path)
                if not content.strip():
                    continue
                doc_text = f"{d_prefix}{path}\n{content[:_EMBED_MAX_CHARS]}"
                doc_texts.append(doc_text)
                valid_paths.append(path)

            if not doc_texts:
                return []

            doc_embs = embedder.encode(doc_texts)

            # Cosine similarity (embeddings are normalized → dot product)
            # Take max similarity across query representations
            similarities = np.max(query_embs @ doc_embs.T, axis=0)

            # Top-K by similarity
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [valid_paths[i] for i in top_indices if similarities[i] > 0]

        finally:
            embedder.unload()

    # ------------------------------------------------------------------
    # Merge + Pass 6: Cross-encoder rerank
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_candidates(
        bm25: list[str],
        embedding: list[str],
        scan: list[str],
        all_paths: list[str],
    ) -> list[str]:
        """Merge and deduplicate candidates from all retrieval signals.

        Order: scan hits first (LLM-selected from structural index),
        then BM25 (keyword match), then embedding (semantic match).
        Deduplication preserves first appearance.
        """
        seen: set[str] = set()
        merged: list[str] = []
        all_paths_set = set(all_paths)

        for path in scan + bm25 + embedding:
            if path not in seen and path in all_paths_set:
                seen.add(path)
                merged.append(path)

        return merged[:_MAX_CANDIDATES_FOR_RERANK]

    def _rerank_candidates(
        self,
        candidates: list[str],
        job_description: str,
        top_k: int = 50,
    ) -> list[str]:
        """Cross-encoder reranking of merged candidates.

        Raises ImportError if sentence-transformers is not installed.
        """
        from fitz_graveyard.planning.agent.models import RerankerModel

        reranker = RerankerModel()
        try:
            reranker.load(self._config.reranker_model)

            doc_texts = []
            valid_paths = []
            for path in candidates:
                content = self._read_file_content(path)
                if not content.strip():
                    continue
                doc_text = f"{path}\n{content[:_EMBED_MAX_CHARS]}"
                doc_texts.append(doc_text)
                valid_paths.append(path)

            if not doc_texts:
                return candidates

            ranked = reranker.rank(job_description, doc_texts, top_k=top_k)
            return [valid_paths[idx] for idx, _score in ranked]

        finally:
            reranker.unload()

    # ------------------------------------------------------------------
    # Pass 7: Import expansion
    # ------------------------------------------------------------------

    def _import_expand(
        self,
        relevant: list[str],
        file_paths: list[str],
        forward_map: dict[str, set[str]],
    ) -> list[str]:
        """Forward-only import expansion, depth 1.

        Traces forward imports ("file A imports B") from each relevant file.
        This catches dependencies (e.g. base.py, protocols) that relevant
        files import but that contain no task keywords.

        Reverse imports ("B is imported by A") are excluded — hub files
        like engine.py are imported by half the codebase, causing explosion.

        Args:
            relevant: Paths the pipeline marked as relevant.
            file_paths: All file paths in the codebase.
            forward_map: Pre-computed {file: {imported_files}} from build_import_graph.

        Returns:
            Merged list: relevant + newly discovered forward imports.
        """
        relevant_set = set(relevant)
        added: list[str] = []
        for rel_path in relevant:
            for dep in forward_map.get(rel_path, set()):
                if dep not in relevant_set:
                    relevant_set.add(dep)
                    added.append(dep)

        if added:
            logger.info(
                f"AgentContextGatherer: import expand added {len(added)} files"
            )

        return list(relevant) + sorted(added)

    @staticmethod
    def _import_reachable(
        seeds: list[str],
        forward_map: dict[str, set[str]],
        candidates: set[str],
    ) -> list[str]:
        """Return files from candidates that are forward-imported by seeds.

        Forward-only, depth 1: "seed imports X" → X is reachable.
        Reverse imports ("Y imports seed") are excluded — they explode
        because hub files like engine.py are imported by half the codebase.
        Always includes the seeds themselves.
        """
        reachable = set(seeds)
        for path in seeds:
            for dep in forward_map.get(path, set()):
                if dep in candidates:
                    reachable.add(dep)

        return sorted(reachable)

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
    # Pass 8: Neighbor expansion
    # ------------------------------------------------------------------

    @staticmethod
    def _neighbor_expand(
        selected: list[str],
        all_paths: list[str],
        expand_from: list[str] | None = None,
    ) -> list[str]:
        """Add sibling files immediately after a high-confidence trigger file.

        Only directories containing an ``expand_from`` file get expanded.
        This prevents noisy BM25/embedding hits from pulling in entire
        irrelevant directories. Siblings are inserted right after the
        first trigger file in each directory.

        Args:
            selected:    All selected files (full list to output).
            all_paths:   Every file in the codebase.
            expand_from: Subset of selected that triggers expansion
                         (default: all of selected for backwards compat).
        """
        triggers = expand_from if expand_from is not None else selected
        selected_set = set(selected)

        # Directories eligible for expansion (only from triggers)
        expand_dirs: set[str] = set()
        for path in triggers:
            parent = str(PurePosixPath(path).parent)
            if parent != ".":
                expand_dirs.add(parent)

        # Group all_paths by directory for fast lookup
        dir_files: dict[str, list[str]] = {}
        for path in all_paths:
            if path in selected_set:
                continue
            parent = str(PurePosixPath(path).parent)
            if parent in expand_dirs:
                dir_files.setdefault(parent, []).append(path)

        # Insert siblings right after the first trigger file in each dir
        result: list[str] = []
        expanded_dirs: set[str] = set()
        for path in selected:
            result.append(path)
            parent = str(PurePosixPath(path).parent)
            if parent in expand_dirs and parent not in expanded_dirs:
                expanded_dirs.add(parent)
                for sibling in dir_files.get(parent, []):
                    result.append(sibling)

        return result

    async def _screen_neighbors(
        self,
        client: Any,
        model: str,
        job_description: str,
        before_neighbors: list[str],
        after_neighbors: list[str],
    ) -> list[str]:
        """LLM-screen neighbor siblings in large directories.

        When a directory contributes more than ``_NEIGHBOR_SCREEN_THRESHOLD``
        new siblings, asks the LLM which ones are actually relevant.
        Directories below the threshold are kept as-is.

        Args:
            client:           LLM client.
            model:            Model name for the screening call.
            job_description:  Task description for relevance judgment.
            before_neighbors: File list before neighbor expansion.
            after_neighbors:  File list after neighbor expansion.

        Returns:
            Filtered file list with irrelevant large-directory siblings removed.
        """
        before_set = set(before_neighbors)
        # Group new siblings by directory
        dir_new: dict[str, list[str]] = {}
        for path in after_neighbors:
            if path not in before_set:
                parent = str(PurePosixPath(path).parent)
                dir_new.setdefault(parent, []).append(path)

        # Find directories that need screening
        large_dirs = {
            d: files for d, files in dir_new.items()
            if len(files) > _NEIGHBOR_SCREEN_THRESHOLD
        }
        if not large_dirs:
            return after_neighbors

        # Screen each large directory with one LLM call
        prompt_template = load_prompt("agent_neighbor_screen")
        remove: set[str] = set()

        for parent_dir, siblings in large_dirs.items():
            # Find the trigger file (first file in before_neighbors from this dir)
            trigger = ""
            for path in before_neighbors:
                if str(PurePosixPath(path).parent) == parent_dir:
                    trigger = path
                    break
            if not trigger:
                continue

            # Build sibling list with brief descriptions
            sibling_lines = []
            for sib in siblings:
                desc = self._brief_description(sib)
                sibling_lines.append(f"- {sib}{desc}")

            prompt = prompt_template.format(
                job_description=job_description,
                trigger_file=trigger,
                sibling_list="\n".join(sibling_lines),
            )

            try:
                response = await client.generate(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0,
                )
                keep = self._parse_file_list(response)
                if keep is not None:
                    keep_set = set(keep)
                    removed = [s for s in siblings if s not in keep_set]
                    remove.update(removed)
                    logger.info(
                        f"AgentContextGatherer: screened {parent_dir}/ — "
                        f"kept {len(keep_set)}/{len(siblings)} siblings"
                    )
                else:
                    logger.warning(
                        f"AgentContextGatherer: neighbor screen parse failed "
                        f"for {parent_dir}/, keeping all {len(siblings)} siblings"
                    )
            except Exception:
                logger.warning(
                    f"AgentContextGatherer: neighbor screen failed for "
                    f"{parent_dir}/, keeping all siblings",
                    exc_info=True,
                )

        if not remove:
            return after_neighbors
        return [p for p in after_neighbors if p not in remove]

    def _brief_description(self, rel_path: str) -> str:
        """Extract a one-line description from a cached file for screening prompts."""
        content = self._file_cache.get(rel_path, "")
        if not content:
            return ""
        # Try to find module docstring
        match = re.search(r'"""(.*?)"""', content[:500], re.DOTALL)
        if not match:
            match = re.search(r"'''(.*?)'''", content[:500], re.DOTALL)
        if match:
            first_line = match.group(1).strip().splitlines()[0]
            return f" — {first_line}"
        return ""

    # ------------------------------------------------------------------
    # Pass 9: Read raw source
    # ------------------------------------------------------------------

    def _read_selected_files(
        self,
        selected: list[str],
    ) -> tuple[dict[str, str], list[str]]:
        """Read all selected files into a dict.

        Returns:
            (file_contents_dict, included_paths)
        """
        file_contents: dict[str, str] = {}
        included: list[str] = []

        for rel_path in selected:
            content = self._read_file_content(rel_path)
            if not content.strip():
                continue
            file_contents[rel_path] = content
            included.append(rel_path)

        return file_contents, included

    # ------------------------------------------------------------------
    # Parsers
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
    def _parse_expand_response(response: str) -> tuple[str, str]:
        """Parse query expansion response into (terms, hyde_code)."""
        terms = ""
        hyde = ""

        terms_match = re.search(
            r"TERMS:\s*\n(.*?)(?:HYPOTHETICAL:|$)", response, re.DOTALL,
        )
        if terms_match:
            terms = terms_match.group(1).strip()

        code_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if code_match:
            hyde = code_match.group(1).strip()

        return terms, hyde

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
