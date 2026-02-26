# fitz_graveyard/planning/pipeline/stages/__init__.py
"""
Pipeline stage implementations.

Exports the three planning stages and the create_stages factory function.
"""

from fitz_graveyard.planning.pipeline.stages.architecture_design import ArchitectureDesignStage
from fitz_graveyard.planning.pipeline.stages.base import PipelineStage, extract_json
from fitz_graveyard.planning.pipeline.stages.context import ContextStage
from fitz_graveyard.planning.pipeline.stages.roadmap_risk import RoadmapRiskStage

DEFAULT_STAGES: list[PipelineStage] = [
    ContextStage(),
    ArchitectureDesignStage(),
    RoadmapRiskStage(),
]


def create_stages() -> list[PipelineStage]:
    """
    Create the three planning pipeline stages.

    Context gathering is handled by AgentContextGatherer before the pipeline runs.
    Stages read gathered context from prior_outputs['_gathered_context'].

    Returns:
        List of configured pipeline stages.
    """
    return [
        ContextStage(),
        ArchitectureDesignStage(),
        RoadmapRiskStage(),
    ]


__all__ = [
    "PipelineStage",
    "extract_json",
    "ContextStage",
    "ArchitectureDesignStage",
    "RoadmapRiskStage",
    "DEFAULT_STAGES",
    "create_stages",
]
