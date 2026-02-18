# fitz_graveyard/__main__.py
"""
Entry point for fitz-graveyard MCP server.

CRITICAL: Server imports configure_logging() first to prevent stdout pollution.

This module coordinates lifecycle initialization before starting the MCP server.
FastMCP doesn't have built-in lifecycle hooks, so we handle it manually.
"""

import asyncio
import logging

# Import server (which configures logging before anything else)
from fitz_graveyard.server import initialize_lifecycle, mcp

logger = logging.getLogger(__name__)


async def main() -> None:
    """
    Main entry point.

    Initializes lifecycle (DB + worker + signals) and then runs MCP server.
    """
    # Initialize lifecycle first
    await initialize_lifecycle()

    # Now run the MCP server via stdio transport
    # Note: mcp.run() is synchronous and blocks, so this must be the last call
    logger.info("Starting MCP server on stdio transport")
    await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
