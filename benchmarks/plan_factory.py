# benchmarks/plan_factory.py
"""
Benchmark factory for rapid retrieval and reasoning evaluation.

Retrieval benchmark:
    python -m benchmarks.plan_factory retrieval --runs 10 --source-dir ../fitz-ai

Reasoning benchmark (uses pre-gathered "perfect" context):
    python -m benchmarks.plan_factory reasoning --runs 3 --source-dir ../fitz-ai --context-file benchmarks/ideal_context.json

Both write results to benchmarks/results/<timestamp>/.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import typer

sys.stderr.write("")  # force stderr init before logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("bench")

app = typer.Typer(no_args_is_help=True)


class _NullCheckpointManager:
    """No-op checkpoint manager for benchmarks (no persistence needed)."""

    async def save_stage(self, job_id: str, stage_name: str, output: dict) -> None:
        pass

    async def load_checkpoint(self, job_id: str) -> dict | None:
        return None

    async def clear_checkpoint(self, job_id: str) -> None:
        pass


def _build_agent_context(context: dict, source_dir: str) -> dict:
    """Build the exact dict that AgentContextGatherer.gather() returns.

    Uses the pre-computed file list from context, reads and compresses
    files from disk, builds structural overview — same post-processing
    as the real gatherer but without the LLM retrieval step.
    """
    from fitz_ai.code.indexer import build_structural_index
    from fitz_ai.engines.fitz_krag.context.compressor import compress_python
    from fitz_graveyard.planning.agent.compressor import compress_file
    from fitz_graveyard.planning.agent.indexer import (
        extract_interface_signatures,
        extract_library_signatures,
    )

    file_list = context.get("file_list", [])
    src = Path(source_dir)

    # Read and compress (mirrors gatherer Step 4)
    file_contents: dict[str, str] = {}
    for rel in file_list:
        full = src / rel
        if not full.is_file():
            continue
        try:
            text = full.read_bytes()[:50_000].decode("utf-8", errors="replace")
            if rel.endswith(".py"):
                text = compress_python(text)
            file_contents[rel] = compress_file(text, rel)
        except OSError:
            pass

    # Build structural overview (mirrors gatherer Step 5)
    selected_index = build_structural_index(src, file_list, max_file_bytes=50_000)
    signatures = extract_interface_signatures(source_dir, file_list, 50_000)
    lib_sigs = extract_library_signatures(source_dir, file_list, [], 50_000)

    parts: list[str] = []
    if signatures:
        parts.append("--- INTERFACE SIGNATURES (auto-extracted, ground truth) ---\n" + signatures)
    if lib_sigs:
        parts.append("--- LIBRARY API REFERENCE (installed packages, ground truth) ---\n" + lib_sigs)
    parts.append("--- STRUCTURAL OVERVIEW (all selected files) ---\n" + selected_index)
    structural_overview = "\n\n".join(parts)

    # Build raw_summaries with seed files (mirrors gatherer Step 6)
    seed_blocks = []
    for rel in file_list:
        if rel in file_contents:
            seed_blocks.append(f"### {rel}\n```\n{file_contents[rel]}\n```")
    raw_summaries = structural_overview
    if seed_blocks:
        raw_summaries += (
            f"\n\n--- SEED FILES ({len(seed_blocks)}/{len(file_list)}) ---\n\n"
            + "\n\n".join(seed_blocks)
        )

    return {
        "synthesized": structural_overview,
        "raw_summaries": raw_summaries,
        "file_contents": file_contents,
        "agent_files": {
            "total_screened": 711,
            "scan_hits": file_list[:11],
            "selected": file_list,
            "included": file_list,
            "forward_map": {},
            "reverse_count": {},
            "file_provenance": {p: {"signals": ["bench"], "in_prompt": True} for p in file_list},
        },
    }


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _results_dir(label: str) -> Path:
    d = Path(__file__).parent / "results" / f"{label}_{_ts()}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------
# Retrieval benchmark
# ------------------------------------------------------------------

async def _run_retrieval_once(
    source_dir: str,
    query: str,
    run_id: int,
) -> dict:
    """Run a single retrieval and return metadata."""
    from fitz_graveyard.config import load_config
    from fitz_graveyard.llm.factory import create_llm_client
    from fitz_graveyard.planning.agent.gatherer import (
        AgentContextGatherer,
        _make_chat_factory,
    )
    from fitz_ai.code import CodeRetriever

    config = load_config()
    client = create_llm_client(config)

    # Health check — ensure model is loaded
    if hasattr(client, "health_check"):
        await client.health_check()

    # Switch to smart_model if different
    if (
        hasattr(client, "switch_model")
        and hasattr(client, "smart_model")
        and client.smart_model != client.model
    ):
        loaded = await client.get_loaded_model() if hasattr(client, "get_loaded_model") else None
        if loaded != client.smart_model:
            await client.switch_model(client.smart_model)

    loop = asyncio.get_running_loop()
    chat_factory = _make_chat_factory(client, loop)

    retriever = CodeRetriever(
        source_dir=source_dir,
        chat_factory=chat_factory,
        llm_tier="smart",
        max_file_bytes=config.agent.max_file_bytes,
    )

    t0 = time.monotonic()
    results = await asyncio.to_thread(retriever.retrieve, query)
    elapsed = time.monotonic() - t0

    # Extract provenance
    scan_hits = []
    import_added = []
    neighbor_added = []
    all_files = []

    for r in results:
        origin = r.address.metadata.get("origin", "neighbor")
        all_files.append(r.file_path)
        if origin == "selected":
            scan_hits.append(r.file_path)
        elif origin == "import":
            import_added.append(r.file_path)
        elif origin == "neighbor":
            neighbor_added.append(r.file_path)

    return {
        "run": run_id,
        "elapsed_s": round(elapsed, 1),
        "total_files": len(all_files),
        "scan_hits": scan_hits,
        "import_added": import_added,
        "neighbor_added": neighbor_added,
        "all_files": all_files,
    }


@app.command()
def retrieval(
    runs: int = typer.Option(10, help="Number of retrieval runs"),
    source_dir: str = typer.Option(..., help="Codebase to index"),
    query: str = typer.Option(
        "Add token usage tracking so I can see how many LLM tokens each query costs",
        help="Job description / query",
    ),
):
    """Run retrieval-only benchmarks (no planning stages)."""
    out_dir = _results_dir("retrieval")
    logger.info(f"Running {runs} retrieval benchmarks -> {out_dir}")

    all_results = []

    async def _run_all():
        for i in range(runs):
            logger.info(f"--- Retrieval run {i + 1}/{runs} ---")
            result = await _run_retrieval_once(source_dir, query, i + 1)
            all_results.append(result)

            # Save each run
            run_file = out_dir / f"run_{i + 1:02d}.json"
            run_file.write_text(json.dumps(result, indent=2))

            scan = result["scan_hits"]
            logger.info(
                f"Run {i + 1}: {len(scan)} scan hits, "
                f"{result['total_files']} total, {result['elapsed_s']}s"
            )

    asyncio.run(_run_all())

    # Summary
    _print_retrieval_summary(all_results, out_dir)


def _print_retrieval_summary(results: list[dict], out_dir: Path) -> None:
    """Print and save retrieval benchmark summary."""
    lines = []
    lines.append(f"# Retrieval Benchmark ({len(results)} runs)\n")

    # Timing
    times = [r["elapsed_s"] for r in results]
    lines.append(f"## Timing")
    lines.append(f"- Min: {min(times):.1f}s")
    lines.append(f"- Max: {max(times):.1f}s")
    lines.append(f"- Avg: {sum(times)/len(times):.1f}s\n")

    # File count consistency
    totals = [r["total_files"] for r in results]
    scans = [len(r["scan_hits"]) for r in results]
    lines.append(f"## File Counts")
    lines.append(f"- Total files: {min(totals)}-{max(totals)}")
    lines.append(f"- Scan hits: {min(scans)}-{max(scans)}\n")

    # Scan hit frequency
    hit_freq: dict[str, int] = {}
    for r in results:
        for f in r["scan_hits"]:
            hit_freq[f] = hit_freq.get(f, 0) + 1

    lines.append(f"## Scan Hit Frequency (across {len(results)} runs)")
    lines.append(f"| File | Hits | % |")
    lines.append(f"|------|------|---|")
    for path, count in sorted(hit_freq.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results)
        lines.append(f"| {path} | {count}/{len(results)} | {pct:.0f}% |")

    # All-files frequency
    all_freq: dict[str, int] = {}
    for r in results:
        for f in r["all_files"]:
            all_freq[f] = all_freq.get(f, 0) + 1

    lines.append(f"\n## All Selected Files Frequency")
    lines.append(f"| File | Hits | % | Signal |")
    lines.append(f"|------|------|---|--------|")
    # Determine most common signal per file
    signal_map: dict[str, str] = {}
    for r in results:
        for f in r["scan_hits"]:
            signal_map[f] = "scan"
        for f in r["import_added"]:
            if f not in signal_map:
                signal_map[f] = "import"
        for f in r["neighbor_added"]:
            if f not in signal_map:
                signal_map[f] = "neighbor"

    for path, count in sorted(all_freq.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results)
        sig = signal_map.get(path, "?")
        lines.append(f"| {path} | {count}/{len(results)} | {pct:.0f}% | {sig} |")

    # Critical file check
    critical_files = [
        "fitz_ai/engines/fitz_krag/engine.py",
        "fitz_ai/core/answer.py",
        "fitz_ai/engines/fitz_krag/query_analyzer.py",
        "fitz_ai/retrieval/detection/registry.py",
    ]
    lines.append(f"\n## Critical File Discovery")
    lines.append(f"| File | Found | % |")
    lines.append(f"|------|-------|---|")
    for cf in critical_files:
        found = all_freq.get(cf, 0)
        pct = 100 * found / len(results)
        lines.append(f"| {cf} | {found}/{len(results)} | {pct:.0f}% |")

    summary = "\n".join(lines)
    (out_dir / "SUMMARY.md").write_text(summary)
    print(summary)


# ------------------------------------------------------------------
# Reasoning benchmark
# ------------------------------------------------------------------

async def _run_reasoning_once(
    source_dir: str,
    query: str,
    context: dict,
    run_id: int,
    out_dir: Path,
) -> dict:
    """Run planning pipeline with pre-gathered context."""
    from fitz_graveyard.config import load_config
    from fitz_graveyard.llm.factory import create_llm_client
    from fitz_graveyard.planning.pipeline.orchestrator import PlanningPipeline
    from fitz_graveyard.planning.pipeline.stages import DEFAULT_STAGES

    config = load_config()
    client = create_llm_client(config)

    # Health check
    if hasattr(client, "health_check"):
        await client.health_check()

    # Ensure planning model is loaded (not the agent model)
    if hasattr(client, "switch_model"):
        loaded = await client.get_loaded_model() if hasattr(client, "get_loaded_model") else None
        if loaded != client.model:
            await client.switch_model(client.model)

    pipeline = PlanningPipeline(
        stages=DEFAULT_STAGES, checkpoint_manager=_NullCheckpointManager(),
    )
    job_id = f"bench_{run_id:03d}"

    # Build the exact _agent_context dict that the real pipeline produces,
    # then pre-inject it so the orchestrator skips agent gathering but
    # everything else runs identically (tool-use, critique, extractions).
    agent_context = _build_agent_context(context, source_dir)

    t0 = time.monotonic()
    result = await pipeline.execute(
        client=client,
        job_id=job_id,
        job_description=query,
        resume=False,
        _bench_overrides={"_agent_context": agent_context, "_source_dir": source_dir},
    )
    elapsed = time.monotonic() - t0

    # Save plan outputs if successful
    plan_text = ""
    if result.success:
        # Save raw outputs as JSON (avoids PlanOutput/PlanRenderer coupling)
        plan_data = {
            k: v for k, v in result.outputs.items()
            if not k.startswith("_")
        }
        plan_text = json.dumps(plan_data, indent=2, default=str)
        plan_file = out_dir / f"plan_{run_id:02d}.json"
        plan_file.write_text(plan_text)

    # Extract architecture decision
    arch = result.outputs.get("architecture", {})
    recommended = arch.get("recommended", "")

    return {
        "run": run_id,
        "elapsed_s": round(elapsed, 1),
        "success": result.success,
        "recommended": recommended,
        "plan_size": len(plan_text),
        "stage_timings": result.stage_timings,
        "error": result.error,
    }


@app.command()
def reasoning(
    runs: int = typer.Option(3, help="Number of reasoning runs"),
    source_dir: str = typer.Option(..., help="Codebase source dir (for file reads)"),
    context_file: str = typer.Option(..., help="JSON file with pre-gathered context"),
    query: str = typer.Option(
        "Add token usage tracking so I can see how many LLM tokens each query costs",
        help="Job description / query",
    ),
):
    """Run reasoning-only benchmarks with fixed retrieval context."""
    context = json.loads(Path(context_file).read_text())
    out_dir = _results_dir("reasoning")
    logger.info(f"Running {runs} reasoning benchmarks -> {out_dir}")

    all_results = []

    async def _run_all():
        for i in range(runs):
            logger.info(f"--- Reasoning run {i + 1}/{runs} ---")
            result = await _run_reasoning_once(
                source_dir, query, context, i + 1, out_dir,
            )
            all_results.append(result)

            run_file = out_dir / f"run_{i + 1:02d}.json"
            run_file.write_text(json.dumps(result, indent=2))

            logger.info(
                f"Run {i + 1}: {result['recommended']} "
                f"({result['elapsed_s']}s, success={result['success']})"
            )

    asyncio.run(_run_all())

    _print_reasoning_summary(all_results, out_dir)


def _print_reasoning_summary(results: list[dict], out_dir: Path) -> None:
    """Print and save reasoning benchmark summary."""
    lines = []
    lines.append(f"# Reasoning Benchmark ({len(results)} runs)\n")

    # Timing
    times = [r["elapsed_s"] for r in results if r["success"]]
    if times:
        lines.append(f"## Timing")
        lines.append(f"- Min: {min(times):.0f}s")
        lines.append(f"- Max: {max(times):.0f}s")
        lines.append(f"- Avg: {sum(times)/len(times):.0f}s\n")

    # Success rate
    successes = sum(1 for r in results if r["success"])
    lines.append(f"## Success: {successes}/{len(results)}\n")

    # Architecture decisions
    lines.append(f"## Architecture Decisions")
    lines.append(f"| Run | Recommended | Time | Size |")
    lines.append(f"|-----|-------------|------|------|")
    for r in results:
        lines.append(
            f"| {r['run']} | {r['recommended']} | {r['elapsed_s']}s | {r['plan_size']}B |"
        )

    # Decision frequency
    decisions: dict[str, int] = {}
    for r in results:
        if r["success"]:
            decisions[r["recommended"]] = decisions.get(r["recommended"], 0) + 1

    lines.append(f"\n## Decision Frequency")
    lines.append(f"| Approach | Count | % |")
    lines.append(f"|----------|-------|---|")
    for approach, count in sorted(decisions.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results)
        lines.append(f"| {approach} | {count} | {pct:.0f}% |")

    # Stage timings (average)
    stage_keys = set()
    for r in results:
        if r.get("stage_timings"):
            stage_keys.update(r["stage_timings"].keys())

    if stage_keys:
        lines.append(f"\n## Avg Stage Timings")
        lines.append(f"| Stage | Avg | Min | Max |")
        lines.append(f"|-------|-----|-----|-----|")
        for key in sorted(stage_keys):
            vals = [
                r["stage_timings"][key]
                for r in results
                if r.get("stage_timings") and key in r["stage_timings"]
            ]
            if vals:
                lines.append(
                    f"| {key} | {sum(vals)/len(vals):.0f}s | {min(vals):.0f}s | {max(vals):.0f}s |"
                )

    summary = "\n".join(lines)
    (out_dir / "SUMMARY.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    app()
