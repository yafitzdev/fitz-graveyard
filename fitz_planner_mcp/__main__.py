# fitz_planner_mcp/__main__.py
"""
Entry point for fitz-planner-mcp MCP server.

CRITICAL: configure_logging() MUST be called first to prevent stdout pollution.
"""

# Configure logging BEFORE any other imports that might log
from fitz_planner_mcp.logging_config import configure_logging

configure_logging()

# Now safe to import other modules
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the MCP server via stdio transport."""
    logger.info("Starting fitz-planner-mcp server")

    # TODO: Import and run server once it's created in Plan 02
    # from fitz_planner_mcp.server import mcp
    # mcp.run(transport="stdio")

    # For now, just log that we're ready
    logger.info("Server initialization complete (placeholder for Plan 02)")


if __name__ == "__main__":
    main()
