# fitz_graveyard/planning/agent/gatherer.py
"""
AgentContextGatherer — powered by fitz-ai[code] retrieval.

Core retrieval (file indexing, LLM selection, import expansion, neighbor
expansion, compression) is delegated to fitz-ai's CodeRetriever.  This module
adds planning-specific post-processing:

  - Interface / library signature extraction
  - Seed-and-fetch context delivery
  - File priority ordering
  - Provenance tracking
  - Planning-optimised compression (test file body collapse)

One retrieval engine, centrally maintained in fitz-ai.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_ai.code import CodeRetriever
from fitz_ai.code.indexer import build_structural_index

from fitz_graveyard.planning.agent.compressor import compress_file
from fitz_graveyard.planning.agent.indexer import (
    extract_interface_signatures,
    extract_library_signatures,
)

if TYPE_CHECKING:
    from fitz_graveyard.config.schema import AgentConfig

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SEED_FILES = 30


def _make_chat_factory(client: Any, loop: asyncio.AbstractEventLoop) -> Callable:
    """Bridge fitz-graveyard's async LLM client to fitz-ai's sync ChatFactory.

    Returns a factory ``(tier: str) -> ChatProvider`` where ChatProvider.chat()
    schedules the async ``client.generate()`` on *loop* and blocks for the result.
    """

    class _SyncChat:
        def __init__(self, model: str) -> None:
            self._model = model

        def chat(self, messages: list[dict]) -> str:
            future = asyncio.run_coroutine_threadsafe(
                client.generate(
                    messages=messages, model=self._model, temperature=0,
                ),
                loop,
            )
            return future.result()

    _TIER_MAP = {
        "fast": "fast_model",
        "balanced": "mid_model",
        "smart": "smart_model",
    }

    def factory(tier: str) -> _SyncChat:
        attr = _TIER_MAP.get(tier, "model")
        model = getattr(client, attr, client.model)
        return _SyncChat(model)

    return factory


class AgentContextGatherer:
    """Retrieval pipeline powered by fitz-ai's CodeRetriever.

    Bridges fitz-graveyard's async LLM client to fitz-ai's sync interface,
    runs retrieval, then adds planning-specific post-processing.
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
        """Run fitz-ai retrieval and return context dict.

        Returns:
            Dict with "synthesized", "raw_summaries", "file_contents",
            and "agent_files".  Empty strings on failure.
        """
        empty: dict[str, Any] = {"synthesized": "", "raw_summaries": ""}

        if not self._config.enabled:
            logger.info("Agent context gathering disabled by config")
            return empty

        t_pipeline = time.monotonic()

        try:
            # Step 1: Bridge async client → sync ChatFactory
            await self._report(progress_callback, 0.06, "agent:mapping")
            loop = asyncio.get_running_loop()
            chat_factory = _make_chat_factory(client, loop)

            # Step 2: Run fitz-ai CodeRetriever (in thread — it's sync)
            await self._report(progress_callback, 0.065, "agent:scanning_index")
            retriever = CodeRetriever(
                source_dir=self._source_dir,
                chat_factory=chat_factory,
                max_file_bytes=self._config.max_file_bytes,
            )

            results = await asyncio.to_thread(retriever.retrieve, job_description)
            file_paths = await asyncio.to_thread(retriever.get_file_paths)

            if not results:
                logger.warning("AgentContextGatherer: retrieval returned no results")
                return empty

            logger.info(
                f"AgentContextGatherer: fitz-ai retrieved {len(results)} files "
                f"from {len(file_paths)} indexed"
            )

            # Step 3: Categorize by origin
            await self._report(progress_callback, 0.070, "agent:import_expand")
            scan_hits: list[str] = []
            import_added: list[str] = []
            neighbor_added: list[str] = []

            for r in results:
                origin = r.address.metadata.get("origin", "neighbor")
                if origin == "selected":
                    scan_hits.append(r.file_path)
                elif origin == "import":
                    import_added.append(r.file_path)
                elif origin == "neighbor":
                    neighbor_added.append(r.file_path)

            included = [r.file_path for r in results]

            logger.info(
                f"AgentContextGatherer: {len(scan_hits)} selected, "
                f"{len(import_added)} import, {len(neighbor_added)} neighbor"
            )

            # Step 4: Build file_contents with planning compression
            await self._report(progress_callback, 0.075, "agent:neighbor_expand")
            file_contents: dict[str, str] = {}
            raw_chars = 0
            for r in results:
                raw_chars += len(r.content)
                # Apply planning-specific compression (test body collapse,
                # non-Python comment stripping).  Python AST compression
                # was already applied by CodeRetriever.
                file_contents[r.file_path] = compress_file(r.content, r.file_path)

            comp_chars = sum(len(c) for c in file_contents.values())
            if raw_chars > 0:
                logger.info(
                    f"AgentContextGatherer: compressed {len(file_contents)} files "
                    f"({raw_chars} -> {comp_chars} chars, "
                    f"{100 * (1 - comp_chars / raw_chars):.0f}% reduction)"
                )

            # Step 5: Build structural overview
            await self._report(progress_callback, 0.080, "agent:reading")
            selected_index = build_structural_index(
                Path(self._source_dir), included,
                max_file_bytes=self._config.max_file_bytes,
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

            # Step 6: Seed-and-fetch context delivery
            max_seeds = getattr(
                self._config, "max_seed_files", _DEFAULT_MAX_SEED_FILES,
            )
            raw_summaries = structural_overview

            # Priority order: scan hits first, then rest by tier
            selected = self._prioritize_for_summary(included)
            scan_set = set(scan_hits)
            priority_files: list[str] = list(scan_hits)
            for path in selected:
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

            # Step 7: Build provenance
            seed_set = set(seed_files)
            scan_set_prov = set(scan_hits)
            import_set = set(import_added)
            neighbor_set = set(neighbor_added)

            file_provenance: dict[str, dict] = {}
            for path in included:
                signals: list[str] = []
                if path in scan_set_prov:
                    signals.append("scan")
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

            return {
                "synthesized": structural_overview,
                "raw_summaries": raw_summaries,
                "file_contents": file_contents,
                "agent_files": {
                    "total_screened": len(file_paths),
                    "scan_hits": scan_hits,
                    "selected": included,
                    "included": included,
                    "forward_map": {},
                    "reverse_count": {},
                    "file_provenance": file_provenance,
                },
            }

        except Exception:
            logger.exception("AgentContextGatherer: pipeline failed")
            return empty

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
