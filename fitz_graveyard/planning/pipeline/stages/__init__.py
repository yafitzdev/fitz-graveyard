# fitz_graveyard/planning/pipeline/stages/__init__.py
"""
Pipeline stage implementations.

Exports all five planning stages and the DEFAULT_STAGES list
for use by PlanningPipeline orchestrator.
"""

from typing import TYPE_CHECKING

from fitz_graveyard.planning.pipeline.stages.base import PipelineStage, extract_json
from fitz_graveyard.planning.pipeline.stages.context import ContextStage
from fitz_graveyard.planning.pipeline.stages.architecture import ArchitectureStage
from fitz_graveyard.planning.pipeline.stages.design import DesignStage
from fitz_graveyard.planning.pipeline.stages.roadmap import RoadmapStage
from fitz_graveyard.planning.pipeline.stages.risk import RiskStage

if TYPE_CHECKING:
    from fitz_graveyard.config.schema import FitzPlannerConfig

# Default stage sequence for full planning pipeline (no KRAG)
DEFAULT_STAGES: list[PipelineStage] = [
    ContextStage(),
    ArchitectureStage(),
    DesignStage(),
    RoadmapStage(),
    RiskStage(),
]


def create_stages(
    config: "FitzPlannerConfig | None" = None, source_dir: str | None = None
) -> list[PipelineStage]:
    """
    Create pipeline stages with optional KRAG configuration.

    Args:
        config: FitzPlannerConfig with KRAG settings (None disables KRAG)
        source_dir: Source directory to point fitz to (None skips pointing)

    Returns:
        List of configured pipeline stages.

    When config is None, returns stages equivalent to DEFAULT_STAGES (no KRAG).
    When config is provided, ContextStage creates KragClient for later stages.
    """
    return [
        ContextStage(config=config, source_dir=source_dir),
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
    "create_stages",
]
