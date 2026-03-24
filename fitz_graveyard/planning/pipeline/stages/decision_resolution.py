# fitz_graveyard/planning/pipeline/stages/decision_resolution.py
"""
Per-decision resolution stage: one LLM call per atomic decision.

Processes decisions in topological order (respecting depends_on).
Each call gets focused context: decision statement + call graph segment +
full source of 1-3 files + constraints from resolved dependencies.
"""

import json
import logging
import time
from collections import defaultdict
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import (
    SYSTEM_PROMPT,
    PipelineStage,
    StageResult,
    extract_json,
)
from fitz_graveyard.planning.pipeline.call_graph import CallGraph
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas.decisions import (
    DecisionResolution,
    DecisionResolutionOutput,
)

logger = logging.getLogger(__name__)


def _topological_sort(decisions: list[dict]) -> list[dict]:
    """Sort decisions in topological order based on depends_on.

    If there are cycles, breaks them by dropping the back-edge
    dependency and logging a warning.

    Returns decisions in execution order (dependencies first).
    """
    id_to_decision = {d["id"]: d for d in decisions}
    in_degree: dict[str, int] = defaultdict(int)
    dependents: dict[str, list[str]] = defaultdict(list)

    for d in decisions:
        d_id = d["id"]
        in_degree.setdefault(d_id, 0)
        for dep in d.get("depends_on", []):
            if dep in id_to_decision:
                in_degree[d_id] += 1
                dependents[dep].append(d_id)

    # Kahn's algorithm
    queue = [d_id for d_id in in_degree if in_degree[d_id] == 0]
    result = []

    while queue:
        d_id = queue.pop(0)
        result.append(id_to_decision[d_id])
        for dep_id in dependents.get(d_id, []):
            in_degree[dep_id] -= 1
            if in_degree[dep_id] == 0:
                queue.append(dep_id)

    # Handle cycles: add remaining decisions with warning
    remaining = [d for d in decisions if d["id"] not in {r["id"] for r in result}]
    if remaining:
        logger.warning(
            f"Dependency cycle detected among decisions: "
            f"{[d['id'] for d in remaining]}. Processing in original order."
        )
        result.extend(remaining)

    return result


class DecisionResolutionStage(PipelineStage):
    """Resolve each atomic decision with focused context.

    For each decision:
    1. Build focused context: decision + call graph segment + source of 1-3 files
    2. Inject constraints from already-resolved dependencies
    3. One LLM call to commit a decision
    4. Extract constraints for downstream decisions

    Token budget per call: ~4-8K input
    """

    @property
    def name(self) -> str:
        return "decision_resolution"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.20, 0.75)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any],
    ) -> list[dict]:
        raise NotImplementedError("Use _build_decision_prompt instead")

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        data = extract_json(raw_output)
        resolution = DecisionResolution(**data)
        return resolution.model_dump()

    def _build_decision_prompt(
        self,
        decision: dict,
        job_description: str,
        call_graph: CallGraph,
        file_contents: dict[str, str],
        upstream_constraints: list[str],
        file_index_entries: dict[str, str],
    ) -> list[dict]:
        """Build prompt for resolving one atomic decision."""
        relevant_files = decision.get("relevant_files", [])
        if relevant_files and call_graph.nodes:
            segment = call_graph.segment_for_files(relevant_files)
            graph_text = segment.format_for_prompt()
        else:
            graph_text = "(no call graph segment for this decision)"

        # Read source of relevant files (1-3 files, compressed)
        source_blocks = []
        for fpath in relevant_files[:3]:
            content = file_contents.get(fpath)
            if content:
                source_blocks.append(f"### {fpath}\n```python\n{content}\n```")
            else:
                entry = file_index_entries.get(fpath, "")
                if entry:
                    source_blocks.append(f"### {fpath} (structural overview)\n{entry}")
        source_text = "\n\n".join(source_blocks) if source_blocks else "(no source available)"

        constraint_text = ""
        if upstream_constraints:
            constraint_text = (
                "CONSTRAINTS FROM PREVIOUS DECISIONS (you MUST respect these):\n"
                + "\n".join(f"- {c}" for c in upstream_constraints)
            )

        prompt_template = load_prompt("decision_resolution")
        prompt = prompt_template.format(
            task_description=job_description,
            decision_id=decision["id"],
            decision_question=decision["question"],
            decision_category=decision.get("category", "technical"),
            call_graph_segment=graph_text,
            source_code=source_text,
            upstream_constraints=constraint_text,
        )
        return self._make_messages(prompt)

    async def execute(
        self,
        client: Any,
        job_description: str,
        prior_outputs: dict[str, Any],
    ) -> StageResult:
        try:
            decomp = prior_outputs.get("decision_decomposition", {})
            decisions = decomp.get("decisions", [])
            if not decisions:
                return StageResult(
                    stage_name=self.name,
                    success=False,
                    output={},
                    raw_output="",
                    error="No decisions from decomposition stage",
                )

            sorted_decisions = _topological_sort(decisions)

            call_graph = prior_outputs.get("_call_graph")
            if call_graph is None:
                call_graph = CallGraph(
                    nodes=[], edges=[], entry_points=[], max_depth=0,
                )
            file_contents = prior_outputs.get("_file_contents", {})
            file_index_entries = prior_outputs.get("_file_index_entries", {})

            # Restore partial resolutions from checkpoint (crash recovery)
            already_resolved: dict[str, dict] = {}
            for key, val in prior_outputs.items():
                if key.startswith("_resolution_partial_"):
                    d_id = key[len("_resolution_partial_"):]
                    already_resolved[d_id] = val

            resolutions: list[dict] = []
            constraint_map: dict[str, list[str]] = {}

            # Seed constraint_map from already-resolved decisions
            for d_id, resolution in already_resolved.items():
                constraint_map[d_id] = resolution.get(
                    "constraints_for_downstream", []
                )

            for i, decision in enumerate(sorted_decisions):
                d_id = decision["id"]

                # Skip already-resolved decisions (crash recovery)
                if d_id in already_resolved:
                    resolutions.append(already_resolved[d_id])
                    logger.info(
                        f"Stage '{self.name}': skipping {d_id} "
                        f"(restored from checkpoint)"
                    )
                    continue

                await self._report_substep(f"resolving:{d_id}")

                upstream = []
                for dep_id in decision.get("depends_on", []):
                    upstream.extend(constraint_map.get(dep_id, []))

                messages = self._build_decision_prompt(
                    decision=decision,
                    job_description=job_description,
                    call_graph=call_graph,
                    file_contents=file_contents,
                    upstream_constraints=upstream,
                    file_index_entries=file_index_entries,
                )

                t0 = time.monotonic()
                raw = await client.generate(
                    messages=messages, temperature=0, max_tokens=4096,
                )
                t1 = time.monotonic()
                logger.info(
                    f"Stage '{self.name}': resolved {d_id} in "
                    f"{t1 - t0:.1f}s ({len(raw)} chars)"
                )

                try:
                    resolution = self.parse_output(raw)
                    resolution["decision_id"] = d_id
                except Exception as e:
                    logger.warning(
                        f"Stage '{self.name}': failed to parse resolution "
                        f"for {d_id}: {e}. Using raw text as decision."
                    )
                    resolution = {
                        "decision_id": d_id,
                        "decision": raw[:500],
                        "reasoning": raw,
                        "evidence": [],
                        "constraints_for_downstream": [],
                    }

                resolutions.append(resolution)
                constraint_map[d_id] = resolution.get(
                    "constraints_for_downstream", []
                )

                # Save partial progress for crash recovery
                prior_outputs[f"_resolution_partial_{d_id}"] = resolution

                await self._report_substep(
                    f"resolved:{d_id} ({i + 1}/{len(sorted_decisions)})"
                )

            output = DecisionResolutionOutput(
                resolutions=[
                    DecisionResolution(**r) for r in resolutions
                ],
            )

            return StageResult(
                stage_name=self.name,
                success=True,
                output=output.model_dump(),
                raw_output=json.dumps(resolutions, indent=2),
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
