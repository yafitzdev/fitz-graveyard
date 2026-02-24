# fitz_graveyard/planning/pipeline/stages/__init__.py
"""
Pipeline stage implementations.

Exports all five planning stages and the create_stages factory function.
"""

from fitz_graveyard.planning.pipeline.stages.architecture import ArchitectureStage
from fitz_graveyard.planning.pipeline.stages.base import PipelineStage, extract_json
from fitz_graveyard.planning.pipeline.stages.context import ContextStage
from fitz_graveyard.planning.pipeline.stages.design import DesignStage
from fitz_graveyard.planning.pipeline.stages.risk import RiskStage
from fitz_graveyard.planning.pipeline.stages.roadmap import RoadmapStage

DEFAULT_STAGES: list[PipelineStage] = [
    ContextStage(),
    ArchitectureStage(),
    DesignStage(),
    RoadmapStage(),
    RiskStage(),
]


def create_stages() -> list[PipelineStage]:
    """
    Create the five planning pipeline stages.

    Context gathering is handled by AgentContextGatherer before the pipeline runs.
    Stages read gathered context from prior_outputs['_gathered_context'].

    Returns:
        List of configured pipeline stages.
    """
    return [
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
    "create_stages",
]
