# fitz_planner_mcp/server.py
"""
FastMCP server instance with tool registration.

CRITICAL: configure_logging() is called first to prevent stdout pollution.
All logging goes to stderr as JSON.
"""

# Configure logging FIRST before any other imports
from fitz_planner_mcp.logging_config import configure_logging

configure_logging()

# Now safe to import everything else
import logging
from pathlib import Path

from fastmcp import FastMCP
from platformdirs import user_config_path

from fitz_planner_mcp.background.lifecycle import ServerLifecycle
from fitz_planner_mcp.config.loader import load_config
from fitz_planner_mcp.models.store import JobStore
from fitz_planner_mcp.tools.check_status import check_status as _check_status
from fitz_planner_mcp.tools.create_plan import create_plan as _create_plan
from fitz_planner_mcp.tools.get_plan import get_plan as _get_plan
from fitz_planner_mcp.tools.list_plans import list_plans as _list_plans
from fitz_planner_mcp.tools.retry_job import retry_job as _retry_job

logger = logging.getLogger(__name__)

# Create FastMCP instance
mcp = FastMCP("fitz-planner-mcp")

# Load configuration
_config = load_config()
logger.info(f"Loaded configuration: model={_config.model}")

# Lifecycle manager (will be initialized by __main__.py)
_lifecycle: ServerLifecycle | None = None


async def get_store() -> JobStore:
    """
    Get the job store from the lifecycle manager.

    Returns:
        JobStore instance

    Raises:
        RuntimeError: If lifecycle not initialized (should never happen)
    """
    if _lifecycle is None:
        raise RuntimeError("Server lifecycle not initialized. Call initialize_lifecycle() first.")
    return _lifecycle.store


async def initialize_lifecycle() -> None:
    """
    Initialize the server lifecycle (DB + crash recovery + worker + signals).

    Must be called before any tool calls. Called by __main__.py on startup.
    """
    global _lifecycle

    # Get DB path from platformdirs
    config_dir = user_config_path("fitz-planner", ensure_exists=True)
    db_path = config_dir / "jobs.db"

    logger.info(f"Initializing lifecycle with db_path={db_path}")

    # Create and start lifecycle
    _lifecycle = ServerLifecycle(str(db_path))
    await _lifecycle.startup()

    logger.info("Lifecycle initialized: SQLite + worker + signals ready")


@mcp.tool()
async def create_plan(
    description: str,
    timeline: str | None = None,
    context: str | None = None,
    integration_points: list[str] | None = None,
) -> dict:
    """Create a new architectural planning job. Queues work for local LLM processing."""
    store = await get_store()
    return await _create_plan(
        description, timeline, context, integration_points, store=store, config=_config
    )


@mcp.tool()
async def check_status(job_id: str) -> dict:
    """Check the status of a planning job. Returns state, progress, and current phase."""
    store = await get_store()
    return await _check_status(job_id, store=store)


@mcp.tool()
async def get_plan(job_id: str, format: str = "full") -> dict:
    """Retrieve a completed plan. Supports full, summary, or roadmap_only format."""
    store = await get_store()
    return await _get_plan(job_id, format, store=store)


@mcp.tool()
async def list_plans() -> dict:
    """List all planning jobs with their status and quality scores."""
    store = await get_store()
    return await _list_plans(store=store)


@mcp.tool()
async def retry_job(job_id: str) -> dict:
    """Retry a failed or interrupted job by re-queuing it for processing."""
    store = await get_store()
    return await _retry_job(job_id, store=store)


logger.info("MCP server initialized with 5 tools")
