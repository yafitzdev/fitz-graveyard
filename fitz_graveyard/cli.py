# fitz_graveyard/cli.py
"""
CLI interface for fitz-graveyard.

Thin presentation layer over the tools/ service layer.
All commands delegate to the same functions that MCP wraps.
"""

import asyncio
import sys

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


@app.command()
def plan(
    description: str = typer.Argument(..., help="What you want to build or accomplish"),
    timeline: str = typer.Option(None, "--timeline", "-t", help="Timeline constraints"),
    context: str = typer.Option(None, "--context", "-c", help="Additional context"),
    api_review: bool = typer.Option(False, "--api-review", help="Enable API review"),
):
    """Queue a new planning job."""
    from fitz_graveyard.config.loader import load_config
    from fitz_graveyard.tools.create_plan import create_plan

    async def _plan():
        store = await _get_store()
        config = load_config()
        try:
            result = await create_plan(
                description=description,
                timeline=timeline,
                context=context,
                integration_points=None,
                api_review=api_review,
                store=store,
                config=config,
            )
        finally:
            await store.close()
        return result

    result = _run(_plan())
    job_id = result["job_id"]
    typer.echo(f"Queued job {job_id}. Run 'fitz-graveyard run' to start processing.")


@app.command("run")
def run_worker():
    """Start the worker to process queued jobs. Ctrl+C to stop."""
    from fitz_graveyard.background.lifecycle import ServerLifecycle
    from fitz_graveyard.config.loader import load_config

    config = load_config()

    from platformdirs import user_config_path

    config_dir = user_config_path("fitz-graveyard", ensure_exists=True)
    db_path = str(config_dir / "jobs.db")

    async def _run_worker():
        lifecycle = ServerLifecycle(db_path, config=config)
        await lifecycle.startup()
        typer.echo("Worker started. Processing queued jobs... (Ctrl+C to stop)")
        try:
            # Block on the worker loop — it runs as an asyncio task,
            # so we just wait forever until cancelled
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            typer.echo("\nShutting down worker...")
            await lifecycle.shutdown()
            typer.echo("Worker stopped.")

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
    typer.echo("-" * 72)

    for p in plans:
        state = p["state"]
        quality = f"{p['quality_score']:.2f}" if p["quality_score"] is not None else "-"
        desc = p["description"]
        typer.echo(
            typer.style(f"{p['job_id']:<14} ", fg=_state_color(state))
            + typer.style(f"{state:<18} ", fg=_state_color(state))
            + f"{quality:<9} {desc}"
        )


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
def retry(job_id: str = typer.Argument(..., help="Job ID to retry")):
    """Retry a failed or interrupted job."""
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

    typer.echo(f"Job {result['job_id']} re-queued. Run 'fitz-graveyard run' to process.")


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
