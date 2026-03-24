# fitz_graveyard/planning/pipeline/stages/decision_decomposition.py
"""
Decision decomposition stage: one cheap LLM call to break the task into
atomic decisions.

Input: task description + call graph + one-line file summaries (NOT full source)
Output: ordered list of AtomicDecision objects with dependencies
"""

import logging
import time
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import (
    PipelineStage,
    StageResult,
    extract_json,
)
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas.decisions import (
    DecisionDecompositionOutput,
)

logger = logging.getLogger(__name__)


class DecisionDecompositionStage(PipelineStage):
    """Break a planning task into atomic, ordered decisions.

    Uses:
    - Task description (from user)
    - Call graph (from call_graph.py -- deterministic AST extraction)
    - One-line file summaries (from agent manifest -- NOT full source)

    Output: list of AtomicDecision with depends_on ordering.

    Token budget: ~2-4K input (task + call graph + manifest).
    This is a CHEAP call -- the model isn't reasoning about architecture,
    just identifying what questions need answering.
    """

    @property
    def name(self) -> str:
        return "decision_decomposition"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.10, 0.20)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any],
    ) -> list[dict]:
        call_graph_text = prior_outputs.get("_call_graph_text", "")
        manifest = prior_outputs.get("_raw_summaries", "")
        impl_check = self._get_implementation_check(prior_outputs)

        prompt_template = load_prompt("decision_decomposition")
        prompt = prompt_template.format(
            task_description=job_description,
            call_graph=call_graph_text,
            file_manifest=manifest,
        )
        if impl_check:
            prompt = f"{impl_check}\n\n{prompt}"
        return self._make_messages(prompt)

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        data = extract_json(raw_output)
        output = DecisionDecompositionOutput(**data)
        return output.model_dump()

    async def execute(
        self,
        client: Any,
        job_description: str,
        prior_outputs: dict[str, Any],
    ) -> StageResult:
        try:
            messages = self.build_prompt(job_description, prior_outputs)
            await self._report_substep("decomposing")
            t0 = time.monotonic()
            raw = await client.generate(
                messages=messages, temperature=0, max_tokens=4096,
            )
            t1 = time.monotonic()
            logger.info(
                f"Stage '{self.name}': decomposition took "
                f"{t1 - t0:.1f}s ({len(raw)} chars)"
            )

            parsed = self.parse_output(raw)
            decisions = parsed.get("decisions", [])
            logger.info(
                f"Stage '{self.name}': produced {len(decisions)} decisions"
            )

            # Validate dependency references
            ids = {d["id"] for d in decisions}
            for d in decisions:
                bad_deps = [
                    dep for dep in d.get("depends_on", [])
                    if dep not in ids
                ]
                if bad_deps:
                    logger.warning(
                        f"Decision {d['id']}: removing invalid deps {bad_deps}"
                    )
                    d["depends_on"] = [
                        dep for dep in d["depends_on"]
                        if dep in ids
                    ]

            # Validate completeness and inject synthetic decisions
            call_graph = prior_outputs.get("_call_graph")
            if call_graph:
                from fitz_graveyard.planning.pipeline.decomposition_validator import (
                    validate_and_augment,
                )
                decisions = validate_and_augment(decisions, call_graph)
                parsed["decisions"] = decisions
                logger.info(
                    f"Stage '{self.name}': {len(decisions)} decisions after validation"
                )

            return StageResult(
                stage_name=self.name,
                success=True,
                output=parsed,
                raw_output=raw,
            )
        except Exception as e:
            logger.error(f"Stage '{self.name}' failed: {e}", exc_info=True)
            return StageResult(
                stage_name=self.name,
                success=False,
                output={},
                raw_output="",
                error=str(e),
            )
