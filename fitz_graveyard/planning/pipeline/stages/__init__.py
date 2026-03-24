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


def create_stages(*, split_reasoning: bool = False) -> list[PipelineStage]:
    """
    Create the three planning pipeline stages.

    Args:
        split_reasoning: If True, arch+design and roadmap+risk each use
            two sequential reasoning calls instead of one. Reduces peak
            context from ~29K to ~8K tokens per call.

    Returns:
        List of configured pipeline stages.
    """
    return [
        ContextStage(),
        ArchitectureDesignStage(split_reasoning=split_reasoning),
        RoadmapRiskStage(split_reasoning=split_reasoning),
    ]


def create_decomposed_stages() -> list[PipelineStage]:
    """Create the three decomposed pipeline stages.

    Returns:
        List of [DecisionDecompositionStage, DecisionResolutionStage, SynthesisStage].
    """
    from fitz_graveyard.planning.pipeline.stages.decision_decomposition import (
        DecisionDecompositionStage,
    )
    from fitz_graveyard.planning.pipeline.stages.decision_resolution import (
        DecisionResolutionStage,
    )
    from fitz_graveyard.planning.pipeline.stages.synthesis import SynthesisStage

    return [
        DecisionDecompositionStage(),
        DecisionResolutionStage(),
        SynthesisStage(),
    ]


__all__ = [
    "PipelineStage",
    "extract_json",
    "ContextStage",
    "ArchitectureDesignStage",
    "RoadmapRiskStage",
    "DEFAULT_STAGES",
    "create_stages",
    "create_decomposed_stages",
]
