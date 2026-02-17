# fitz_planner_mcp/planning/pipeline/stages/__init__.py
"""
Pipeline stage implementations.

Exports all five planning stages and the DEFAULT_STAGES list
for use by PlanningPipeline orchestrator.
"""

from fitz_planner_mcp.planning.pipeline.stages.base import PipelineStage, extract_json
from fitz_planner_mcp.planning.pipeline.stages.context import ContextStage
from fitz_planner_mcp.planning.pipeline.stages.architecture import ArchitectureStage
from fitz_planner_mcp.planning.pipeline.stages.design import DesignStage
from fitz_planner_mcp.planning.pipeline.stages.roadmap import RoadmapStage
from fitz_planner_mcp.planning.pipeline.stages.risk import RiskStage

# Default stage sequence for full planning pipeline
DEFAULT_STAGES: list[PipelineStage] = [
    ContextStage(),
    ArchitectureStage(),
    DesignStage(),
    RoadmapStage(),
    RiskStage(),
]

__all__ = [
    "PipelineStage",
    "extract_json",
    "ContextStage",
    "ArchitectureStage",
    "DesignStage",
    "RoadmapStage",
    "RiskStage",
    "DEFAULT_STAGES",
]
