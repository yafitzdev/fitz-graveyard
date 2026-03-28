# benchmarks/eval_plans.py
"""
Sonnet-as-Judge plan evaluator.

Assembles scoring prompts from plan JSON + codebase context, writes them
to files. Scoring is done by Claude Code subagents — no Anthropic SDK needed.

Usage:
    # Prepare scoring prompts
    python -m benchmarks.plan_factory prepare-scoring \
        --results-dir benchmarks/results/decomposed_20260324_180926 \
        --source-dir ../fitz-ai \
        --context-file benchmarks/ideal_context.json

    # Then score via Claude Code conversation or subagents
"""

import json
import logging
import statistics
from pathlib import Path

from .eval_prompt import (
    SYSTEM_PROMPT,
    build_scoring_prompt,
    extract_referenced_paths,
    load_file_contents,
)
from .eval_schemas import (
    BatchScore,
    ConsistencyCheck,
    DimensionScore,
    PlanScore,
)

logger = logging.getLogger(__name__)

DIMENSIONS = [
    "file_identification",
    "contract_preservation",
    "internal_consistency",
    "codebase_alignment",
    "implementability",
    "scope_calibration",
]


def prepare_scoring_prompt(
    plan_path: Path,
    query: str,
    structural_index: str,
    source_dir: Path,
) -> str:
    """Build the full scoring prompt for a single plan.

    Returns the assembled prompt text ready to be evaluated by
    Claude Code or any other LLM.
    """
    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_json = json.dumps(plan_data, indent=2, default=str)

    referenced_paths = extract_referenced_paths(plan_data)
    referenced_files = load_file_contents(source_dir, referenced_paths)
    logger.info(
        f"  {plan_path.name}: {len(referenced_paths)} paths referenced, "
        f"{len(referenced_files)} files loaded from codebase"
    )

    return build_scoring_prompt(
        query=query,
        structural_index=structural_index,
        referenced_files=referenced_files,
        plan_json=plan_json,
    )


def prepare_batch(
    plan_dir: Path,
    query: str,
    structural_index: str,
    source_dir: Path,
) -> list[tuple[str, str]]:
    """Prepare scoring prompts for all plans in a directory.

    Returns list of (plan_filename, prompt_text) tuples.
    Also writes prompts to score_prompt_NN.md files for reference.
    """
    plan_files = sorted(plan_dir.glob("plan_*.json"))
    if not plan_files:
        raise FileNotFoundError(f"No plan_*.json files in {plan_dir}")

    logger.info(f"Preparing scoring prompts for {len(plan_files)} plans in {plan_dir}")
    prompts = []
    for pf in plan_files:
        prompt = prepare_scoring_prompt(pf, query, structural_index, source_dir)
        # Extract run number from filename
        num = pf.stem.replace("plan_", "")
        prompt_file = plan_dir / f"score_prompt_{num}.md"
        prompt_file.write_text(prompt, encoding="utf-8")
        prompts.append((pf.name, prompt))
        logger.info(f"  Wrote {prompt_file.name} ({len(prompt)} chars)")

    return prompts


def parse_scores(raw_text: str) -> dict:
    """Parse JSON response into DimensionScore objects.

    Returns dict with dimension names as keys and DimensionScore as values,
    plus '_notes' key for overall_notes.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    data = json.loads(text)

    result = {}
    for dim in DIMENSIONS:
        if dim in data and isinstance(data[dim], dict):
            result[dim] = DimensionScore(
                score=int(data[dim].get("score", 1)),
                justification=str(data[dim].get("justification", "")),
            )
        else:
            result[dim] = DimensionScore(score=1, justification="Missing from response")

    result["_notes"] = str(data.get("overall_notes", ""))
    return result


def build_plan_score(plan_file: str, query: str, raw_response: str) -> PlanScore:
    """Parse a raw LLM response into a PlanScore object."""
    scores = parse_scores(raw_response)
    dim_scores = {d: scores[d] for d in DIMENSIONS}
    total = sum(d.score for d in dim_scores.values())

    return PlanScore(
        plan_file=plan_file,
        query=query,
        **dim_scores,
        overall_notes=scores.get("_notes", ""),
        total_score=total,
        normalized_score=round(total / 60, 3),
    )


def build_batch_score(
    query: str,
    scores: list[PlanScore],
    model: str = "claude-code",
    consistency_check: ConsistencyCheck | None = None,
) -> BatchScore:
    """Aggregate individual plan scores into a batch result."""
    totals = [s.total_score for s in scores]
    dim_avgs = {}
    for dim in DIMENSIONS:
        vals = [getattr(s, dim).score for s in scores]
        dim_avgs[dim] = round(statistics.mean(vals), 1)

    return BatchScore(
        query=query,
        model=model,
        plans_scored=len(scores),
        dimension_averages=dim_avgs,
        total_average=round(statistics.mean(totals), 1),
        total_std_dev=round(statistics.stdev(totals), 1) if len(totals) > 1 else 0.0,
        total_min=min(totals),
        total_max=max(totals),
        total_cost_usd=0.0,
        scores=scores,
        consistency_check=consistency_check,
    )


def build_consistency_check(
    run_1: PlanScore,
    run_2: PlanScore,
) -> ConsistencyCheck:
    """Compare two scores of the same plan."""
    deltas = {}
    for dim in DIMENSIONS:
        s1 = getattr(run_1, dim).score
        s2 = getattr(run_2, dim).score
        deltas[dim] = abs(s1 - s2)

    max_delta = max(deltas.values())
    return ConsistencyCheck(
        plan_file=run_1.plan_file,
        run_1_total=run_1.total_score,
        run_2_total=run_2.total_score,
        dimension_deltas=deltas,
        max_delta=max_delta,
        acceptable=max_delta <= 2,
    )


def save_results(batch: BatchScore, out_dir: Path) -> None:
    """Save scores.json and SCORE_SUMMARY.md to the results directory."""
    scores_file = out_dir / "scores.json"
    scores_file.write_text(
        json.dumps(batch.model_dump(mode="json"), indent=2, default=str)
    )

    summary = format_score_summary(batch)
    summary_file = out_dir / "SCORE_SUMMARY.md"
    summary_file.write_text(summary)


def format_score_summary(batch: BatchScore) -> str:
    """Format batch scores as a human-readable markdown summary."""
    lines = [f"# Plan Evaluation ({batch.plans_scored} plans scored)\n"]

    lines.append(f"## Query\n{batch.query}\n")

    # Dimension averages
    lines.append("## Dimension Averages (1-10 scale)")
    lines.append("| Dimension | Avg | Min | Max |")
    lines.append("|-----------|-----|-----|-----|")
    for dim in DIMENSIONS:
        vals = [getattr(s, dim).score for s in batch.scores]
        label = dim.replace("_", " ").title()
        lines.append(
            f"| {label} | {statistics.mean(vals):.1f} | {min(vals)} | {max(vals)} |"
        )

    # Overall
    lines.append(f"\n## Overall")
    lines.append(f"- Average total: {batch.total_average}/60 ({batch.total_average/60*100:.1f}%)")
    lines.append(f"- Range: {batch.total_min}-{batch.total_max}")
    if batch.total_std_dev > 0:
        lines.append(f"- Std dev: {batch.total_std_dev:.1f}")
    lines.append(f"- Model: {batch.model}")

    # Consistency
    if batch.consistency_check:
        cc = batch.consistency_check
        lines.append(f"\n## Scorer Consistency")
        lines.append(
            f"- {cc.plan_file} scored twice: "
            f"{cc.run_1_total} vs {cc.run_2_total} "
            f"(max dimension delta: {cc.max_delta})"
        )
        lines.append(f"- Acceptable: {'yes' if cc.acceptable else 'NO — rubric may need tightening'}")
        if not cc.acceptable:
            worst = max(cc.dimension_deltas, key=cc.dimension_deltas.get)
            lines.append(f"- Worst dimension: {worst} (delta: {cc.dimension_deltas[worst]})")

    # Per-plan scores
    lines.append(f"\n## Per-Plan Scores")
    lines.append("| Plan | Total | Norm | Best | Worst | Notes |")
    lines.append("|------|-------|------|------|-------|-------|")
    for s in batch.scores:
        dims = s.dimensions
        best = max(dims, key=lambda d: dims[d].score)
        worst = min(dims, key=lambda d: dims[d].score)
        best_label = best.replace("_", " ").title()
        worst_label = worst.replace("_", " ").title()
        notes = s.overall_notes[:60] + "..." if len(s.overall_notes) > 60 else s.overall_notes
        lines.append(
            f"| {s.plan_file} | {s.total_score}/60 | {s.normalized_score:.2f} | "
            f"{best_label} ({dims[best].score}) | "
            f"{worst_label} ({dims[worst].score}) | {notes} |"
        )

    return "\n".join(lines)
