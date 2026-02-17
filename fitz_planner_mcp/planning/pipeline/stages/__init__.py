# fitz_planner_mcp/planning/pipeline/stages/__init__.py
"""
Pipeline stage implementations.

This package will be populated with the five planning stages in Plan 04.
For now, it only exports the PipelineStage ABC.
"""

from fitz_planner_mcp.planning.pipeline.stages.base import PipelineStage

__all__ = ["PipelineStage"]
