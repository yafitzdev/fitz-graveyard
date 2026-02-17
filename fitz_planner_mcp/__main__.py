# fitz_planner_mcp/__main__.py
"""
Entry point for fitz-planner-mcp MCP server.

CRITICAL: Server imports configure_logging() first to prevent stdout pollution.
"""

# Import server (which configures logging before anything else)
from fitz_planner_mcp.server import mcp

if __name__ == "__main__":
    # Run server via stdio transport
    mcp.run(transport="stdio")
