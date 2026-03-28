# fitz_graveyard/planning/pipeline/stages/artifact_resolution.py
"""
Artifact resolution — generates implementation artifacts directly from
decision resolutions + actual source code.

Replaces the synthesis → extraction two-step for artifacts, which caused
prose-to-code fabrication (synthesis writes "build context", extraction
materializes as self._build_context() which doesn't exist).

Each artifact gets its own focused LLM call with:
  1. The source code of the file being modified (ground truth)
  2. Resolutions that reference this file (what to change)
  3. All constraints from all resolutions (what not to break)
  4. Instance attributes from the cheat sheet (what self._ attrs exist)

This mirrors how decision_resolution.py works: one focused call per decision
with the relevant source code, instead of one big synthesis pass.
"""

import json
import logging
import time
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import extract_json

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a senior software architect writing precise implementation code. "
    "Use ONLY real method names, attributes, and signatures from the source code provided."
)

_ARTIFACT_PROMPT = """Write ONLY the new code to add for the file described below.

TASK: {task_description}

## File to write: {filename}
Purpose: {purpose}

## Source code of the EXISTING file (ground truth — use real method names from this)
{source_code}

## Relevant decisions (what to change)
{relevant_decisions}

## All constraints (what NOT to break)
{all_constraints}

## Instance attributes available on key classes
{attribute_template}

## Rules
- Write ONLY new methods/classes to ADD. Do NOT rewrite the existing file.
- Do NOT reproduce existing methods — they stay unchanged.
- Use ONLY method names and attributes from the source code or attribute list above.
- Do NOT invent helper methods like _build_context(), _prepare_messages(), etc.
- If mirroring an existing method, use the SAME parameter names and types.
- Keep artifacts SHORT — only the new code, with a comment showing where it goes.

Return ONLY valid JSON:
{{
  "filename": "{filename}",
  "content": "ONLY the new code to add (not a full file rewrite)",
  "purpose": "{purpose}"
}}"""


async def resolve_artifacts(
    client: Any,
    job_description: str,
    prior_outputs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate artifacts directly from resolutions + source code.

    Instead of extracting artifacts from synthesis prose (which causes
    fabrication), this generates each artifact in a focused LLM call
    with the actual source code and relevant decisions.

    Returns list of artifact dicts (filename, content, purpose).
    """
    resolutions = prior_outputs.get(
        "decision_resolution", {},
    ).get("resolutions", [])
    if not resolutions:
        return []

    # ONLY use context stage's needed_artifacts — do NOT infer from resolutions.
    # Inferring from resolutions caused 12-21 artifacts (rewrote entire codebase).
    needed = prior_outputs.get("context", {}).get("needed_artifacts", [])
    if not needed:
        return []
    # Cap at 5 artifacts to prevent budget exhaustion
    needed = needed[:5]

    # Get source code pool
    file_contents = prior_outputs.get("_agent_context", {}).get("file_contents", {})
    file_index = prior_outputs.get("_file_index_entries", {})
    source_dir = prior_outputs.get("_source_dir")

    # Get attribute template from cheat sheet builder
    from fitz_graveyard.planning.pipeline.stages.synthesis import (
        SynthesisStage,
        _build_attribute_template,
    )

    # Build sections dict for attribute template
    full_index = prior_outputs.get(
        "_agent_context", {},
    ).get("full_structural_index", "")
    if not full_index:
        full_index = prior_outputs.get("_gathered_context", "")
    sections: dict[str, list[str]] = {}
    current_file = ""
    for line in full_index.split("\n"):
        if line.startswith("## "):
            current_file = line[3:].strip()
        elif current_file and line.strip():
            sections.setdefault(current_file, []).append(line)

    referenced_files = set()
    for r in resolutions:
        for ev in r.get("evidence", []):
            if ":" in ev:
                path = ev.split(":")[0].strip()
                if path.endswith(".py"):
                    referenced_files.add(path)

    attr_template = _build_attribute_template(
        referenced_files, prior_outputs, sections,
    )

    # Collect all constraints
    all_constraints = []
    for r in resolutions:
        for c in r.get("constraints_for_downstream", []):
            all_constraints.append(c)
    constraint_text = "\n".join(f"- {c}" for c in all_constraints) if all_constraints else "(none)"

    # Generate one artifact per needed file
    artifacts = []
    for artifact_spec in needed:
        # Parse "filename -- purpose" format
        if " -- " in artifact_spec:
            filename, purpose = artifact_spec.split(" -- ", 1)
        elif " - " in artifact_spec:
            filename, purpose = artifact_spec.split(" - ", 1)
        else:
            filename = artifact_spec.strip()
            purpose = "Implementation artifact"
        filename = filename.strip()
        purpose = purpose.strip()

        # Get source code of the file (if it exists — new files won't have source)
        source = _get_source(filename, file_contents, file_index, source_dir)

        # Find resolutions that reference this file
        relevant = _find_relevant_resolutions(filename, resolutions)
        decision_text = _format_decisions(relevant) if relevant else "(no specific decisions for this file)"

        prompt = _ARTIFACT_PROMPT.format(
            task_description=job_description,
            filename=filename,
            purpose=purpose,
            source_code=source,
            relevant_decisions=decision_text,
            all_constraints=constraint_text,
            attribute_template=attr_template or "(not available)",
        )

        t0 = time.monotonic()
        try:
            raw = await client.generate(
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
            )
            data = extract_json(raw)
            elapsed = time.monotonic() - t0
            logger.info(
                f"Artifact resolution: {filename} ({elapsed:.1f}s, "
                f"{len(data.get('content', ''))} chars)"
            )
            artifacts.append({
                "filename": data.get("filename", filename),
                "content": data.get("content", ""),
                "purpose": data.get("purpose", purpose),
            })
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.warning(
                f"Artifact resolution failed for {filename} ({elapsed:.1f}s): {e}"
            )
            # Don't crash — return empty artifact
            artifacts.append({
                "filename": filename,
                "content": f"# Artifact generation failed: {e}",
                "purpose": purpose,
            })

    return artifacts


def _infer_needed_artifacts(resolutions: list[dict]) -> list[str]:
    """Infer which files need artifacts from resolution constraints."""
    files = set()
    for r in resolutions:
        for c in r.get("constraints_for_downstream", []):
            c_lower = c.lower()
            # Look for "add X method to Y" or "new file Z" patterns
            if "must be added" in c_lower or "new method" in c_lower or "new file" in c_lower:
                # Extract file paths from evidence
                for ev in r.get("evidence", []):
                    if ":" in ev:
                        path = ev.split(":")[0].strip()
                        if path.endswith(".py"):
                            files.add(path)
    # Also add any file with "parallel method" constraints
    for r in resolutions:
        for c in r.get("constraints_for_downstream", []):
            if "parallel" in c.lower() or "stream" in c.lower():
                for ev in r.get("evidence", []):
                    if ":" in ev:
                        path = ev.split(":")[0].strip()
                        if path.endswith(".py"):
                            files.add(path)
    return [f"{f} -- streaming implementation" for f in sorted(files)]


def _get_source(
    filename: str,
    file_contents: dict[str, str],
    file_index: dict[str, str],
    source_dir: str | None,
) -> str:
    """Get source code for a file, with fallbacks."""
    # Direct match
    content = file_contents.get(filename, "")
    if content:
        # Truncate to keep prompt manageable
        return f"```python\n{content[:15000]}\n```"

    # Partial match
    for key in file_contents:
        if key.endswith(filename) or filename.endswith(key):
            return f"```python\n{file_contents[key][:15000]}\n```"

    # Structural index fallback
    entry = file_index.get(filename, "")
    if entry:
        return f"(structural overview only)\n{entry}"

    # Disk fallback
    if source_dir:
        from pathlib import Path
        full_path = Path(source_dir) / filename
        if full_path.is_file():
            try:
                raw = full_path.read_bytes()[:15000]
                text = raw.decode("utf-8", errors="replace")
                return f"```python\n{text}\n```"
            except OSError:
                pass

    return "(new file — no existing source)"


def _find_relevant_resolutions(
    filename: str,
    resolutions: list[dict],
) -> list[dict]:
    """Find resolutions whose evidence references this file."""
    relevant = []
    basename = filename.rsplit("/", 1)[-1] if "/" in filename else filename
    for r in resolutions:
        # Check evidence for file references
        for ev in r.get("evidence", []):
            if filename in ev or basename in ev:
                relevant.append(r)
                break
        else:
            # Also check if the decision question mentions the file
            question = r.get("question", "")
            if filename in question or basename in question:
                relevant.append(r)
    return relevant


def _format_decisions(resolutions: list[dict]) -> str:
    """Format relevant resolutions for the artifact prompt."""
    lines = []
    for r in resolutions:
        lines.append(f"### {r.get('decision_id', '?')}")
        lines.append(f"Decision: {r.get('decision', '')[:500]}")
        evidence = r.get("evidence", [])
        if evidence:
            lines.append("Evidence:")
            for e in evidence[:4]:
                lines.append(f"  - {e}")
        constraints = r.get("constraints_for_downstream", [])
        if constraints:
            lines.append("Constraints:")
            for c in constraints[:3]:
                lines.append(f"  - {c}")
        lines.append("")
    return "\n".join(lines)
