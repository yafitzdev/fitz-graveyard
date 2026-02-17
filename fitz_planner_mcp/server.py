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

from fastmcp import FastMCP

from fitz_planner_mcp.config.loader import load_config
from fitz_planner_mcp.models.jobs import InMemoryJobStore
from fitz_planner_mcp.tools.check_status import check_status as _check_status
from fitz_planner_mcp.tools.create_plan import create_plan as _create_plan
from fitz_planner_mcp.tools.get_plan import get_plan as _get_plan
from fitz_planner_mcp.tools.list_plans import list_plans as _list_plans

logger = logging.getLogger(__name__)

# Create FastMCP instance
mcp = FastMCP("fitz-planner-mcp")

# Load configuration
_config = load_config()
logger.info(f"Loaded configuration: model={_config.model}")

# Create shared job store
_store = InMemoryJobStore()


@mcp.tool()
def create_plan(
    description: str,
    timeline: str | None = None,
    context: str | None = None,
    integration_points: list[str] | None = None,
) -> dict:
    """Create a new architectural planning job. Queues work for local LLM processing."""
    return _create_plan(
        description, timeline, context, integration_points, store=_store, config=_config
    )


@mcp.tool()
def check_status(job_id: str) -> dict:
    """Check the status of a planning job. Returns state, progress, and current phase."""
    return _check_status(job_id, store=_store)


@mcp.tool()
def get_plan(job_id: str, format: str = "full") -> dict:
    """Retrieve a completed plan. Supports full, summary, or roadmap_only format."""
    return _get_plan(job_id, format, store=_store)


@mcp.tool()
def list_plans() -> dict:
    """List all planning jobs with their status and quality scores."""
    return _list_plans(store=_store)


logger.info("MCP server initialized with 4 tools")
