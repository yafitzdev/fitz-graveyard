# fitz_graveyard/planning/pipeline/stages/context.py
"""
Context understanding stage: Extract project requirements and constraints.
"""

from typing import Any, TYPE_CHECKING

from fitz_graveyard.planning.pipeline.stages.base import PipelineStage, extract_json
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas import ContextOutput
from fitz_graveyard.planning.krag import KragClient

if TYPE_CHECKING:
    from fitz_graveyard.config.schema import FitzPlannerConfig


class ContextStage(PipelineStage):
    """
    First stage: Understand project context.

    Extracts:
    - Project description
    - Key requirements
    - Constraints
    - Stakeholders
    - Scope boundaries

    Also creates KragClient and stores in prior_outputs['_krag_client']
    for use by later stages.
    """

    def __init__(
        self,
        config: "FitzPlannerConfig | None" = None,
        source_dir: str | None = None,
    ):
        """
        Initialize ContextStage.

        Args:
            config: FitzPlannerConfig with KRAG settings (None disables KRAG)
            source_dir: Source directory to point fitz to (None skips pointing)
        """
        self._config = config
        self._source_dir = source_dir

    @property
    def name(self) -> str:
        return "context"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.10, 0.25)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any]
    ) -> list[dict]:
        """Build context understanding prompt with KRAG context."""
        # Create KragClient if config provided (only ContextStage creates the client)
        if self._config is not None and "_krag_client" not in prior_outputs:
            krag_client = KragClient.from_config(self._config.krag, self._source_dir)
            # Store in prior_outputs for later stages
            prior_outputs["_krag_client"] = krag_client

        # Query KRAG for architecture overview and module purposes
        krag_queries = [
            "What is the project architecture overview and main modules?",
            "What key interfaces and integration points exist in the codebase?",
        ]
        krag_context = self._get_krag_context(krag_queries, prior_outputs)

        prompt_template = load_prompt("context")
        prompt = prompt_template.format(
            description=job_description, krag_context=krag_context
        )

        return [{"role": "user", "content": prompt}]

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        """Parse context output into ContextOutput schema."""
        data = extract_json(raw_output)
        # Validate with Pydantic
        context = ContextOutput(**data)
        # Return as dict for checkpoint serialization
        return context.model_dump()
