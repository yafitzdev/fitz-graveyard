# benchmarks/eval_prompt.py
"""
Prompt construction for Sonnet-as-Judge plan evaluation.

Builds the scoring prompt with codebase context and rubric anchors,
and extracts referenced file paths from plan JSON for targeted reads.
"""

import json
import re
from pathlib import Path

DIMENSION_ANCHORS = {
    "file_identification": (
        "Does the plan identify the correct files to modify/create? "
        "Check against the structural index.\n"
        "  1-3: Most files are wrong or hallucinated\n"
        "  4-6: Core files identified but misses important ones or includes irrelevant ones\n"
        "  7-8: Identifies the right files with minor omissions\n"
        "  9-10: All critical files correctly identified, no hallucinations"
    ),
    "contract_preservation": (
        "Does the plan preserve existing public APIs and method signatures? "
        "Check proposed changes against real signatures in the codebase.\n"
        "  1-3: Proposes changes that break existing callers without acknowledging it\n"
        "  4-6: Some interface changes acknowledged but impact analysis is incomplete\n"
        "  7-8: Correctly identifies which interfaces are stable vs internal, preserves public APIs\n"
        "  9-10: Precisely references real signatures, explicitly addresses backward compatibility"
    ),
    "internal_consistency": (
        "Do the plan's own decisions contradict each other? "
        "Do roadmap phases reference components from design? Do risks reference real phases?\n"
        "  1-3: Major contradictions (architecture says X, design does Y)\n"
        "  4-6: Minor inconsistencies between sections\n"
        "  7-8: Sections are coherent and cross-reference correctly\n"
        "  9-10: Perfect internal alignment, all cross-references valid"
    ),
    "codebase_alignment": (
        "Does the plan work with the patterns that actually exist in the codebase? "
        "Does it propose patterns consistent with the existing code style?\n"
        "  1-3: Ignores existing patterns entirely, proposes incompatible approaches\n"
        "  4-6: Partially aligned but misses key patterns or conventions\n"
        "  7-8: Follows existing patterns, correctly extends them\n"
        "  9-10: Deep understanding of codebase idioms, leverages existing infrastructure"
    ),
    "implementability": (
        "Are the steps concrete enough that a developer could follow them? "
        "Are deliverables specific?\n"
        "  1-3: Vague hand-waving ('add error handling', 'improve performance')\n"
        "  4-6: Directionally correct but missing specifics (which function, which file)\n"
        "  7-8: Specific enough to implement without guessing\n"
        "  9-10: Could be handed to a junior developer and they'd know exactly what to do"
    ),
    "scope_calibration": (
        "Is the plan's scope appropriate for the task? "
        "Not ballooning into a rewrite, not missing critical pieces?\n"
        "  1-3: Wildly over/under-scoped\n"
        "  4-6: Scope is roughly right but includes unnecessary work or misses something\n"
        "  7-8: Well-calibrated scope with clear boundaries\n"
        "  9-10: Precisely scoped, explicitly states what's in/out with good justification"
    ),
}

SYSTEM_PROMPT = (
    "You are an expert software architect evaluating implementation plans "
    "for quality and feasibility. You will examine a plan against the actual "
    "codebase it targets, then score the plan on 6 dimensions.\n\n"
    "Be rigorous but fair. A plan doesn't need to match any particular 'ideal' "
    "approach -- there are multiple valid ways to implement any feature. Judge "
    "whether THIS plan would work, not whether it matches YOUR preferred approach."
)


def build_scoring_prompt(
    query: str,
    structural_index: str,
    referenced_files: dict[str, str],
    plan_json: str,
) -> str:
    """Build the full scoring prompt for Sonnet."""
    parts = []

    parts.append(f"## Task\n{query}")

    parts.append(f"## Target Codebase Structure\n{structural_index}")

    if referenced_files:
        file_sections = []
        for path, content in sorted(referenced_files.items()):
            file_sections.append(f"### {path}\n```\n{content}\n```")
        parts.append("## Referenced File Contents\n" + "\n\n".join(file_sections))

    parts.append(f"## Plan to Evaluate\n```json\n{plan_json}\n```")

    # Build dimension instructions
    dim_instructions = []
    for i, (name, anchor) in enumerate(DIMENSION_ANCHORS.items(), 1):
        label = name.replace("_", " ").title()
        dim_instructions.append(f"{i}. **{label}** ({name}): {anchor}")

    parts.append(
        "## Scoring Instructions\n"
        "Score this plan on each dimension using the scale 1-10. For each "
        "dimension, provide:\n"
        "1. A score (integer 1-10)\n"
        "2. A brief justification (1-3 sentences citing specific evidence)\n\n"
        "### Dimensions\n\n"
        + "\n\n".join(dim_instructions)
        + "\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        "```json\n"
        "{\n"
        '  "file_identification": {"score": N, "justification": "..."},\n'
        '  "contract_preservation": {"score": N, "justification": "..."},\n'
        '  "internal_consistency": {"score": N, "justification": "..."},\n'
        '  "codebase_alignment": {"score": N, "justification": "..."},\n'
        '  "implementability": {"score": N, "justification": "..."},\n'
        '  "scope_calibration": {"score": N, "justification": "..."},\n'
        '  "overall_notes": "1-2 sentence summary of main strengths/weaknesses"\n'
        "}\n"
        "```"
    )

    return "\n\n".join(parts)


# Pattern to match file paths like fitz_ai/core/answer.py
_FILE_PATH_RE = re.compile(r"[a-zA-Z_][\w/\\.-]*\.(?:py|yaml|yml|json|toml|txt|md)")


def extract_referenced_paths(plan_data: dict) -> set[str]:
    """Extract all file paths referenced anywhere in the plan JSON.

    Walks the entire plan structure recursively, looking for strings
    that match file path patterns.
    """
    paths: set[str] = set()

    def _walk(obj: object) -> None:
        if isinstance(obj, str):
            for match in _FILE_PATH_RE.findall(obj):
                # Normalize backslashes
                paths.add(match.replace("\\", "/"))
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(plan_data)
    return paths


def load_file_contents(
    source_dir: Path,
    paths: set[str],
    max_bytes: int = 50_000,
) -> dict[str, str]:
    """Read referenced files from the target codebase.

    Returns a dict of path -> content for files that exist.
    Each file is truncated to max_bytes.
    """
    contents: dict[str, str] = {}
    for rel_path in sorted(paths):
        full_path = source_dir / rel_path
        if full_path.is_file():
            try:
                raw = full_path.read_bytes()[:max_bytes]
                contents[rel_path] = raw.decode("utf-8", errors="replace")
            except OSError:
                continue
    return contents
