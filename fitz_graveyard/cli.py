# fitz_graveyard/cli.py
"""
CLI interface for fitz-graveyard.

Thin presentation layer over the tools/ service layer.
All commands delegate to the same functions that MCP wraps.
"""

import asyncio
import re
import sys
import time
from collections import deque


def _fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable duration (e.g. '5m17s', '42s')."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    return f"{m}m{s:02d}s"

import typer

app = typer.Typer(
    name="fitz-graveyard",
    help="Local-first AI architectural planning using local LLMs.",
    no_args_is_help=True,
)


def _run(coro):
    """Run async function from sync CLI context."""
    return asyncio.run(coro)


async def _get_store():
    """Open SQLiteJobStore directly (no lifecycle needed for read-only ops)."""
    from platformdirs import user_config_path

    from fitz_graveyard.models.sqlite_store import SQLiteJobStore

    config_dir = user_config_path("fitz-graveyard", ensure_exists=True)
    store = SQLiteJobStore(str(config_dir / "jobs.db"))
    await store.initialize()
    return store


def _state_color(state: str) -> str:
    """Return ANSI color for job state."""
    colors = {
        "complete": typer.colors.GREEN,
        "running": typer.colors.YELLOW,
        "queued": typer.colors.CYAN,
        "awaiting_review": typer.colors.MAGENTA,
        "failed": typer.colors.RED,
        "interrupted": typer.colors.RED,
    }
    return colors.get(state, typer.colors.WHITE)


# Stage list for live progress display: (display_name, progress_threshold_for_complete)
_DISPLAY_STAGES = [
    ("Health check",        0.05),
    ("Codebase analysis",   0.09),
    ("Requirements",        0.25),
    ("Architecture+Design", 0.65),
    ("Roadmap+Risk",        0.95),
    ("Finalizing",          1.00),
]


_PHASE_DESCRIPTIONS = {
    "starting": "Starting up...",
    "initializing": "Initializing...",
    "health_check": "Checking LLM connectivity...",
    "agent:mapping": "Mapping codebase...",
    "agent:selecting": "Selecting relevant files...",
    "agent:synthesizing": "Synthesizing context...",
    "context:reasoning": "Analyzing requirements and constraints...",
    "context_complete": "Requirements analysis complete",
    "architecture_design:reasoning": "Exploring architecture and design...",
    "architecture_design_complete": "Architecture+Design complete",
    "roadmap_risk:reasoning": "Planning roadmap and assessing risks...",
    "roadmap_risk_complete": "Roadmap+Risk complete",
    "coherence_check": "Checking cross-stage coherence...",
    "scoring": "Scoring plan quality...",
    "rendering": "Rendering plan to markdown...",
    "writing_file": "Saving plan to disk...",
    "finalizing": "Finalizing...",
}


def _get_phase_description(phase: str | None) -> str:
    """Map a current_phase string to a human-friendly description."""
    if not phase:
        return ""
    # Direct lookup
    if phase in _PHASE_DESCRIPTIONS:
        return _PHASE_DESCRIPTIONS[phase]
    # Agent summarizing: "agent:summarizing:path/to/file.py" → "Summarizing file.py..."
    if phase.startswith("agent:summarizing:"):
        filename = phase.split(":")[-1].rsplit("/", 1)[-1]
        return f"Summarizing {filename}..."
    # Self-critique: "architecture_design:critiquing" → "Reviewing analysis..."
    if phase.endswith(":critiquing"):
        return "Reviewing analysis for quality..."
    # Field group extraction: "architecture_design:extracting:approaches" → "Extracting approaches..."
    if ":extracting:" in phase:
        group_label = phase.split(":")[-1]
        return f"Extracting {group_label}..."
    # Stage name without sub-step
    if phase in ("context", "architecture_design", "roadmap_risk"):
        return _PHASE_DESCRIPTIONS.get(f"{phase}:generating", f"Working on {phase}...")
    return phase


def _make_live_display(
    description: str,
    progress: float,
    elapsed: float,
    current_phase: str = "",
    stage_durations: dict[int, float] | None = None,
    stage_started: dict[int, float] | None = None,
    log_lines: list[str] | None = None,
):
    """Build a rich renderable for the live progress display."""
    from rich.table import Table
    from rich.text import Text

    stage_durations = stage_durations or {}
    stage_started = stage_started or {}

    table = Table.grid(padding=(0, 2))
    table.add_column(width=3)
    table.add_column()
    table.add_column(justify="right", style="dim", width=7)

    # Determine previous threshold to detect "active" stage
    prev_threshold = 0.0
    for i, (name, threshold) in enumerate(_DISPLAY_STAGES):
        if progress >= threshold:
            icon = Text("✓", style="green")
            row_style = "dim"
            if i in stage_durations:
                duration = _fmt_duration(stage_durations[i])
            else:
                duration = ""
        elif progress >= prev_threshold:
            icon = Text("⟳", style="yellow")
            row_style = "bold"
            if i in stage_started:
                duration = _fmt_duration(time.monotonic() - stage_started[i])
            else:
                duration = _fmt_duration(elapsed)
        else:
            icon = Text("○", style="dim")
            row_style = "dim"
            duration = ""

        table.add_row(icon, Text(name, style=row_style), Text(duration, style="dim"))
        prev_threshold = threshold

    # Progress bar
    bar_width = 36
    filled = int(progress * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    pct = f"{progress * 100:.0f}%"

    from rich.panel import Panel
    from rich.console import Group
    from rich.text import Text as RichText

    title = RichText(f" {description[:60]}{'…' if len(description) > 60 else ''} ", style="bold")
    bar_text = RichText(f"\n  {bar}  {pct}", style="cyan")

    # Status line: current activity
    status_desc = _get_phase_description(current_phase)
    status_text = RichText(f"\n  {status_desc}", style="dim italic") if status_desc else RichText("")

    # Activity log
    parts: list = [table, bar_text, status_text]
    if log_lines:
        parts.append(RichText(""))  # spacer
        for line in log_lines:
            parts.append(RichText(f"  {line}", style="dim"))

    parts.append(RichText(""))  # bottom padding

    return Panel(
        Group(*parts),
        title=title,
        border_style="bright_black",
    )


async def _gather_context_for_clarification(client, config, description: str, source_dir: str, console) -> str:
    """Run agent context gathering standalone for use in clarification flow."""
    from rich.status import Status
    from fitz_graveyard.planning.agent import AgentContextGatherer
    from pathlib import Path

    if not Path(source_dir).is_dir():
        return ""

    agent = AgentContextGatherer(config=config.agent, source_dir=source_dir)
    with Status("[dim]Reading project...[/dim]", console=console, spinner="dots"):
        gathered = await agent.gather(client=client, job_description=description)
    # gather() returns {"synthesized": str, "raw_summaries": str} — extract synthesized for clarification
    if isinstance(gathered, dict):
        return gathered.get("synthesized", "")
    return gathered


async def _run_inline(
    job_id: str,
    store,
    config,
    description: str,
    pre_gathered_context: str | None = None,
    resume: bool = False,
) -> None:
    """Run a job inline with live rich progress display, then print the plan."""
    from rich.live import Live
    from rich.console import Console

    from fitz_graveyard.background.worker import BackgroundWorker
    from fitz_graveyard.llm.factory import create_llm_client

    client = create_llm_client(config)
    worker = BackgroundWorker(
        store,
        config=config,
        ollama_client=client,
        memory_threshold=config.ollama.memory_threshold,
    )

    console = Console(stderr=True)
    start = time.monotonic()

    job_task = asyncio.create_task(
        worker.process_job_direct(job_id, pre_gathered_context=pre_gathered_context, resume=resume)
    )

    # Per-stage timing tracking
    stage_started: dict[int, float] = {}
    stage_durations: dict[int, float] = {}
    prev_thresholds = [0.0] + [t for _, t in _DISPLAY_STAGES[:-1]]

    # Activity log
    log_lines: deque[str] = deque(maxlen=5)
    last_phase = ""

    try:
        with Live(
            _make_live_display(description, 0.0, 0.0),
            console=console,
            refresh_per_second=4,
        ) as live:
            while not job_task.done():
                job = await store.get(job_id)
                if job:
                    elapsed = time.monotonic() - start
                    progress = job.progress or 0.0
                    phase = job.current_phase or ""

                    # Track stage transitions for timing
                    for i, (name, threshold) in enumerate(_DISPLAY_STAGES):
                        if progress >= threshold and i not in stage_durations:
                            if i in stage_started:
                                stage_durations[i] = time.monotonic() - stage_started[i]
                        elif progress >= prev_thresholds[i] and i not in stage_started and i not in stage_durations:
                            stage_started[i] = time.monotonic()

                    # Track phase changes for activity log
                    if phase and phase != last_phase:
                        desc = _get_phase_description(phase)
                        if desc:
                            log_lines.append(f"{time.strftime('%H:%M:%S')} {desc}")
                        last_phase = phase

                    live.update(_make_live_display(
                        description, progress, elapsed,
                        current_phase=phase,
                        stage_durations=stage_durations,
                        stage_started=stage_started,
                        log_lines=list(log_lines),
                    ))
                await asyncio.sleep(0.3)

            # Final update
            job = await store.get(job_id)
            if job:
                elapsed = time.monotonic() - start
                live.update(_make_live_display(
                    description, job.progress or 0.0, elapsed,
                    current_phase=job.current_phase or "",
                    stage_durations=stage_durations,
                    stage_started=stage_started,
                    log_lines=list(log_lines),
                ))

    except (KeyboardInterrupt, asyncio.CancelledError):
        job_task.cancel()
        try:
            await job_task
        except (asyncio.CancelledError, Exception):
            pass
        raise KeyboardInterrupt

    # Check result
    exc = job_task.exception()
    if exc:
        raise exc

    # Print result
    final_job = await store.get(job_id)
    if not final_job:
        return

    state = final_job.state.value
    quality = f"{final_job.quality_score:.2f}" if final_job.quality_score else "N/A"
    elapsed = time.monotonic() - start

    console.print()
    if state == "complete":
        console.print(f"[green]✓ Done[/green]  quality: {quality}  time: {_fmt_duration(elapsed)}")
        if final_job.file_path:
            console.print(f"[dim]Saved:[/dim] {final_job.file_path}")
        console.print()
        # Print the plan to stdout
        if final_job.file_path:
            from fitz_graveyard.tools.get_plan import get_plan
            result = await get_plan(job_id, "full", store)
            typer.echo(result["content"])
    elif state == "awaiting_review":
        console.print(f"[magenta]⏸ Awaiting review[/magenta]  Run 'fitz-graveyard confirm {job_id}' to proceed.")
    else:
        error = final_job.error or "unknown error"
        console.print(f"[red]✗ Failed[/red]: {error}")
        raise typer.Exit(1)


@app.command()
def plan(
    description: str = typer.Argument(..., help="What you want to build or accomplish"),
    timeline: str = typer.Option(None, "--timeline", "-t", help="Timeline constraints"),
    context: str = typer.Option(None, "--context", "-c", help="Additional context"),
    api_review: bool = typer.Option(False, "--api-review", help="Enable API review"),
    source_dir: str = typer.Option(None, "--source-dir", help="Path to codebase for agent context"),
    detach: bool = typer.Option(False, "--detach", "-d", help="Queue only, don't run inline"),
    clarify: bool = typer.Option(False, "--clarify", help="Ask clarifying questions before planning"),
):
    """Queue and run a planning job with live progress. Use --detach to queue only."""
    from fitz_graveyard.config.loader import load_config
    from fitz_graveyard.tools.create_plan import create_plan

    async def _plan():
        import logging as _logging
        from rich.console import Console as _Console

        store = await _get_store()
        config = load_config()
        enriched_description = description
        pre_gathered_context: str | None = None

        # Clarification flow: gather context first, then ask questions with that context
        if clarify and not detach and sys.stdin.isatty():
            try:
                from fitz_graveyard.planning.clarification import get_clarifying_questions
                from fitz_graveyard.llm.factory import create_llm_client

                console = _Console(stderr=True)
                client = create_llm_client(config)
                if await client.health_check():
                    effective_source_dir = source_dir or config.agent.source_dir or "."
                    if config.agent.enabled:
                        pre_gathered_context = await _gather_context_for_clarification(
                            client, config, description, effective_source_dir, console
                        )

                    questions = await get_clarifying_questions(
                        client, description, codebase_context=pre_gathered_context or ""
                    )
                    if questions:
                        console.print("\n[bold]A few quick questions to sharpen the plan:[/bold]\n")
                        answers = []
                        for i, q in enumerate(questions, 1):
                            answer = typer.prompt(f"  {i}. {q}", default="")
                            if answer.strip():
                                answers.append(f"Q: {q}\nA: {answer}")
                        if answers:
                            enriched_description = (
                                description
                                + "\n\n## Clarifications\n\n"
                                + "\n\n".join(answers)
                            )
                        typer.echo()
            except Exception as e:
                _logging.getLogger(__name__).warning(f"Clarification skipped: {e}")

        try:
            result = await create_plan(
                description=enriched_description,
                timeline=timeline,
                context=context,
                integration_points=None,
                api_review=api_review,
                store=store,
                config=config,
                source_dir=source_dir,
            )
        except Exception:
            await store.close()
            raise

        job_id = result["job_id"]

        if detach:
            await store.close()
            typer.echo(f"Queued job {job_id}. Run 'fitz-graveyard run' to start processing.")
            return

        try:
            await _run_inline(
                job_id, store, config, enriched_description,
                pre_gathered_context=pre_gathered_context,
            )
        finally:
            await store.close()

    try:
        _run(_plan())
    except KeyboardInterrupt:
        typer.echo("\nCancelled.", err=True)
        raise typer.Exit(130)


@app.command("run")
def run_worker():
    """Start the worker to process queued jobs. Ctrl+C to stop."""
    import logging
    import sys

    from fitz_graveyard.background.lifecycle import ServerLifecycle
    from fitz_graveyard.config.loader import load_config
    from platformdirs import user_config_path

    # Simple human-readable logging to stderr for CLI mode
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    config = load_config()
    config_dir = user_config_path("fitz-graveyard", ensure_exists=True)
    db_path = str(config_dir / "jobs.db")

    async def _run_worker():
        lifecycle = ServerLifecycle(db_path, config=config)
        await lifecycle.startup()
        typer.echo("Worker started. Processing queued jobs... (Ctrl+C to stop)\n")
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            typer.echo("\nShutting down...")
            await lifecycle.shutdown()

    try:
        _run(_run_worker())
    except KeyboardInterrupt:
        pass


@app.command("list")
def list_jobs():
    """List all planning jobs."""
    from fitz_graveyard.tools.list_plans import list_plans

    async def _list():
        store = await _get_store()
        try:
            return await list_plans(store=store)
        finally:
            await store.close()

    result = _run(_list())
    plans = result["plans"]

    if not plans:
        typer.echo("No plans found.")
        return

    # Print table header
    typer.echo(f"{'JOB ID':<14} {'STATE':<18} {'QUALITY':<9} DESCRIPTION")
    typer.echo("-" * 80)

    for p in plans:
        state = p["state"]
        quality = f"{p['quality_score']:.2f}" if p["quality_score"] is not None else "-"
        desc = p["description"]

        # Derive project name from plan file path (.../project/.fitz-graveyard/plans/plan_*.md)
        project = ""
        if p.get("file_path"):
            from pathlib import Path as _Path
            parts = _Path(p["file_path"]).parts
            # Find the part before .fitz-graveyard
            for i, part in enumerate(parts):
                if part == ".fitz-graveyard" and i > 0:
                    project = parts[i - 1]
                    break

        typer.echo(
            typer.style(f"{p['job_id']:<14} ", fg=_state_color(state))
            + typer.style(f"{state:<18} ", fg=_state_color(state))
            + f"{quality:<9} {desc}"
        )
        if project:
            typer.echo(f"{'':14} {'':18} {'':9} " + typer.style(f"↳ {project}", fg=typer.colors.BRIGHT_BLACK))


@app.command()
def status(job_id: str = typer.Argument(..., help="Job ID to check")):
    """Check the status of a planning job."""
    from fitz_graveyard.tools.check_status import check_status

    async def _status():
        store = await _get_store()
        try:
            return await check_status(job_id, store=store)
        finally:
            await store.close()

    try:
        result = _run(_status())
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    state = result["state"]
    typer.echo(f"Job:      {result['job_id']}")
    typer.echo(typer.style(f"State:    {state}", fg=_state_color(state)))
    typer.echo(f"Progress: {result['progress'] * 100:.0f}%")
    if result.get("current_phase"):
        typer.echo(f"Phase:    {result['current_phase']}")
    if result.get("message"):
        typer.echo(f"Message:  {result['message']}")
    if result.get("error"):
        typer.echo(typer.style(f"Error:    {result['error']}", fg=typer.colors.RED))


@app.command()
def get(
    job_id: str = typer.Argument(..., help="Job ID to retrieve"),
    format: str = typer.Option("full", "--format", "-f", help="Output format: full, summary, roadmap_only"),
):
    """Retrieve a completed plan."""
    from fitz_graveyard.tools.get_plan import get_plan

    async def _get():
        store = await _get_store()
        try:
            return await get_plan(job_id, format, store=store)
        finally:
            await store.close()

    try:
        result = _run(_get())
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Print raw markdown to stdout (pipeable)
    typer.echo(result["content"])


@app.command()
def resume(job_id: str = typer.Argument(..., help="Job ID to resume")):
    """Resume a failed or interrupted job with live progress display."""
    from fitz_graveyard.config.loader import load_config
    from fitz_graveyard.tools.retry_job import retry_job

    async def _resume():
        store = await _get_store()
        config = load_config()

        job = await store.get(job_id)
        if not job:
            await store.close()
            typer.echo(f"Job '{job_id}' not found.", err=True)
            raise typer.Exit(1)

        # Re-queue if needed (failed/interrupted), skip if already queued
        if job.state.value in ("failed", "interrupted"):
            await retry_job(job_id, store=store)
        elif job.state.value == "running":
            # Stale running state from killed worker — reset to queued
            from fitz_graveyard.models.jobs import JobState
            await store.update(job_id, state=JobState.QUEUED, progress=0.0, error=None, current_phase=None)
        elif job.state.value == "complete":
            await store.close()
            typer.echo(f"Job '{job_id}' is already complete. Use 'fitz-graveyard get {job_id}' to view.", err=True)
            raise typer.Exit(1)
        # queued / awaiting_review — just run it

        description = job.description

        try:
            await _run_inline(job_id, store, config, description, resume=True)
        finally:
            await store.close()

    try:
        _run(_resume())
    except KeyboardInterrupt:
        typer.echo("\nCancelled.", err=True)
        raise typer.Exit(130)


@app.command()
def retry(job_id: str = typer.Argument(..., help="Job ID to retry")):
    """Retry a failed or interrupted job (queue only, use 'resume' for live UI)."""
    from fitz_graveyard.tools.retry_job import retry_job

    async def _retry():
        store = await _get_store()
        try:
            return await retry_job(job_id, store=store)
        finally:
            await store.close()

    try:
        result = _run(_retry())
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Job {result['job_id']} re-queued. Run 'fitz-graveyard resume {result['job_id']}' or 'fitz-graveyard run' to process.")


@app.command()
def confirm(job_id: str = typer.Argument(..., help="Job ID to approve API review")):
    """Approve API review after seeing cost estimate."""
    from fitz_graveyard.tools.confirm_review import confirm_review

    async def _confirm():
        store = await _get_store()
        try:
            return await confirm_review(job_id, store=store)
        finally:
            await store.close()

    try:
        result = _run(_confirm())
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"API review approved for {result['job_id']}. Run 'fitz-graveyard run' to process.")


@app.command()
def cancel(job_id: str = typer.Argument(..., help="Job ID to skip API review")):
    """Skip API review, finalize plan without it."""
    from fitz_graveyard.tools.cancel_review import cancel_review

    async def _cancel():
        store = await _get_store()
        try:
            return await cancel_review(job_id, store=store)
        finally:
            await store.close()

    try:
        result = _run(_cancel())
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"API review skipped for {result['job_id']}. Plan finalized.")


@app.command()
def serve():
    """Start the MCP server (for Claude Code integration)."""
    from fitz_graveyard.__main__ import main

    asyncio.run(main())


if __name__ == "__main__":
    app()
