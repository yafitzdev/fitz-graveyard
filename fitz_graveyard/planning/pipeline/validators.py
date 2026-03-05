# fitz_graveyard/planning/pipeline/validators.py
"""
Post-extraction quality validators.

Each validator takes a merged dict (from the field-group extraction loop),
checks a structural condition, and optionally repairs via LLM or deterministic fix.
Validators run between extraction and parse_output() — never crash the stage.
"""

import ast
import json
import logging
import re
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import SYSTEM_PROMPT, extract_json

logger = logging.getLogger(__name__)

_MIN_EXISTING_FILES = 6
_MAX_EXISTING_FILES = 25
_MIN_ADRS = 2
_MIN_RISKS = 2

# Patterns that indicate vague verification commands
_VAGUE_PATTERNS = [
    re.compile(r"^#?\s*(run tests?|verify|check that|test it|manual)", re.IGNORECASE),
    re.compile(r"^(pytest|python)\s*$", re.IGNORECASE),
    re.compile(r"^run\s+\w+\s*$", re.IGNORECASE),
]


def ensure_min_existing_files(
    merged: dict[str, Any],
    prior_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Backfill existing_files from raw summaries if extraction missed files.

    No LLM call — parses ``### path`` headers from _raw_summaries, which are
    real files the agent actually read.
    """
    existing = merged.get("existing_files", [])
    raw_summaries = prior_outputs.get("_raw_summaries", "")

    if not raw_summaries:
        return merged

    # Parse file paths from raw summary headers (### path/to/file.py)
    summary_paths = re.findall(r"^###\s+(.+?)(?:\s*$)", raw_summaries, re.MULTILINE)
    if not summary_paths:
        return merged

    # Normalize existing entries: extract path before " — " annotation
    existing_paths = set()
    for entry in existing:
        path = entry.split(" — ")[0].split(" - ")[0].strip()
        existing_paths.add(path)

    added = 0
    for path in summary_paths:
        path = path.strip()
        if path not in existing_paths and len(existing) < _MAX_EXISTING_FILES:
            existing.append(f"{path} — discovered by agent")
            existing_paths.add(path)
            added += 1

    if added:
        logger.info(f"ensure_min_existing_files: backfilled {added} files from raw summaries")
        merged["existing_files"] = existing

    return merged


async def ensure_min_adrs(
    merged: dict[str, Any],
    client: Any,
    prior_outputs: dict[str, Any],
    reasoning: str,
) -> dict[str, Any]:
    """Add ADRs via LLM if fewer than minimum.

    One LLM call asking for design tradeoffs based on the architecture.
    Falls back silently on failure.
    """
    adrs = merged.get("adrs", [])
    if len(adrs) >= _MIN_ADRS:
        return merged

    recommended = merged.get("recommended", "")
    arch_reasoning = merged.get("reasoning", "")
    components = merged.get("components", [])

    # Also try prior_outputs for architecture data (merged stage splits output)
    if not recommended and "architecture" in prior_outputs:
        arch = prior_outputs["architecture"]
        recommended = arch.get("recommended", "")
        arch_reasoning = arch.get("reasoning", "")
    if not components and "design" in prior_outputs:
        components = prior_outputs["design"].get("components", [])

    needed = _MIN_ADRS - len(adrs)
    comp_summary = ", ".join(c.get("name", c) if isinstance(c, dict) else str(c) for c in components[:5])

    prompt = (
        f"Architecture: {recommended}\n"
        f"Reasoning: {arch_reasoning[:500]}\n"
        f"Components: {comp_summary}\n\n"
        f"Name {needed} design tradeoffs for this architecture that someone could reasonably disagree with. "
        "Consider: what data crosses boundaries, what failure mode is accepted, what v1 excludes.\n\n"
        "Return ONLY a JSON array of ADR objects:\n"
        '[{"title": "ADR: ...", "context": "...", "decision": "...", "rationale": "...", '
        '"consequences": ["..."], "alternatives_considered": ["..."]}]'
    )

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw = await client.generate(messages=messages)
        data = extract_json(raw)
        new_adrs = data if isinstance(data, list) else data.get("adrs", [])
        if new_adrs:
            adrs.extend(new_adrs[:needed])
            merged["adrs"] = adrs
            logger.info(f"ensure_min_adrs: added {len(new_adrs[:needed])} ADRs via LLM")
    except Exception as e:
        logger.warning(f"ensure_min_adrs: LLM repair failed: {e}")

    return merged


def ensure_phase_zero(
    merged: dict[str, Any],
    prior_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Inject Phase 0 (read-before-writing) if missing.

    No LLM call — uses discovered files from _raw_summaries.
    Bumps existing phase numbers and adjusts dependencies.
    """
    phases = merged.get("phases", [])
    if any(p.get("number") == 0 for p in phases):
        return merged

    raw_summaries = prior_outputs.get("_raw_summaries", "")
    if not raw_summaries:
        return merged

    # Extract top 5 file paths from raw summaries
    summary_paths = re.findall(r"^###\s+(.+?)(?:\s*$)", raw_summaries, re.MULTILINE)
    top_files = [p.strip() for p in summary_paths[:5]]
    if not top_files:
        return merged

    phase_zero = {
        "number": 0,
        "name": "Read Before Writing",
        "objective": "Verify integration points by reading source",
        "deliverables": [f"Verified: {path}" for path in top_files],
        "dependencies": [],
        "estimated_complexity": "low",
        "key_risks": [],
        "verification_command": "# Manual: confirm each file's role",
        "estimated_effort": "~15 min",
    }

    # Bump existing phase numbers +1 and adjust dependencies
    from fitz_graveyard.planning.schemas.roadmap import _coerce_phase_number

    for phase in phases:
        phase["number"] = _coerce_phase_number(phase.get("number", 0)) + 1
        coerced_deps = []
        for d in phase.get("dependencies", []):
            try:
                coerced_deps.append(_coerce_phase_number(d) + 1)
            except ValueError:
                continue
        phase["dependencies"] = coerced_deps

    merged["phases"] = [phase_zero] + phases

    # Update scheduling fields
    if "critical_path" in merged:
        coerced_cp = []
        for p in merged.get("critical_path", []):
            try:
                coerced_cp.append(_coerce_phase_number(p) + 1)
            except ValueError:
                continue
        merged["critical_path"] = [0] + coerced_cp
    if "parallel_opportunities" in merged:
        result = []
        for group in merged.get("parallel_opportunities", []):
            coerced_group = []
            for p in group:
                try:
                    coerced_group.append(_coerce_phase_number(p) + 1)
                except ValueError:
                    continue
            result.append(coerced_group)
        merged["parallel_opportunities"] = result
    if "total_phases" in merged:
        merged["total_phases"] = len(merged["phases"])

    logger.info(f"ensure_phase_zero: injected Phase 0 with {len(top_files)} files")
    return merged


def _is_vague_verification(cmd: str) -> bool:
    """Check if a verification command is too vague to be useful."""
    cmd = cmd.strip()
    if not cmd:
        return True
    for pattern in _VAGUE_PATTERNS:
        if pattern.match(cmd):
            return True
    return False


def _fallback_verification(phase: dict) -> str:
    """Generate a template verification command from the first deliverable."""
    deliverables = phase.get("deliverables", [])
    if not deliverables:
        return f"python -m pytest tests/ -v -k phase_{phase.get('number', 0)}"

    first = deliverables[0]
    # Try to extract a module/file stem from the deliverable
    match = re.search(r"(\w+(?:/\w+)*\.py)", first)
    if match:
        path = match.group(1)
        stem = path.replace("/", "_").replace(".py", "")
        return f"python -m pytest tests/unit/test_{stem}.py -v"
    return f"python -m pytest tests/ -v -k phase_{phase.get('number', 0)}"


async def ensure_concrete_verification(
    merged: dict[str, Any],
    client: Any,
    reasoning: str,
) -> dict[str, Any]:
    """Replace vague verification commands with concrete ones.

    Batches all vague phases into one LLM call. Falls back to template on failure.
    """
    phases = merged.get("phases", [])
    vague_phases = []
    for phase in phases:
        cmd = phase.get("verification_command", "")
        if _is_vague_verification(cmd):
            vague_phases.append(phase)

    if not vague_phases:
        return merged

    # Try LLM repair for all vague phases at once
    phase_descriptions = "\n".join(
        f"Phase {p['number']} ({p.get('name', '')}): deliverables = {p.get('deliverables', [])}"
        for p in vague_phases
    )

    prompt = (
        f"For each phase below, write the exact shell command to verify it worked.\n"
        f"Use pytest, python -c, curl, or similar. Must include a file path or module name.\n\n"
        f"{phase_descriptions}\n\n"
        f"Return ONLY a JSON object mapping phase number to command string:\n"
        '{{"0": "pytest tests/...", "1": "python -c ..."}}'
    )

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw = await client.generate(messages=messages)
        data = extract_json(raw)
        applied = 0
        for phase in vague_phases:
            num_str = str(phase["number"])
            if num_str in data and not _is_vague_verification(data[num_str]):
                phase["verification_command"] = data[num_str]
                applied += 1
        if applied:
            logger.info(f"ensure_concrete_verification: LLM fixed {applied} vague commands")
        # Apply fallback for any still-vague phases
        for phase in vague_phases:
            if _is_vague_verification(phase.get("verification_command", "")):
                phase["verification_command"] = _fallback_verification(phase)
                logger.info(f"ensure_concrete_verification: fallback for phase {phase['number']}")
    except Exception as e:
        logger.warning(f"ensure_concrete_verification: LLM repair failed: {e}")
        for phase in vague_phases:
            phase["verification_command"] = _fallback_verification(phase)

    return merged


def ensure_grounded_risks(
    merged: dict[str, Any],
    prior_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Remove risks that reference files/modules not found in context.

    No LLM call. Never drops below _MIN_RISKS.
    """
    risks = merged.get("risks", [])
    if not risks:
        return merged

    # Build set of known references from gathered context and prior outputs
    known_text = prior_outputs.get("_gathered_context", "")
    known_text += " " + prior_outputs.get("_raw_summaries", "")
    for key in ("context", "architecture", "design"):
        if key in prior_outputs:
            known_text += " " + json.dumps(prior_outputs[key])

    # If no context available, can't verify grounding — keep all
    if not known_text.strip():
        return merged

    # Extract file-like references from risk descriptions
    # Pattern: word/word.ext or word.ext (looks like a file path)
    _FILE_REF = re.compile(r"\b([\w/]+\.(?:py|js|ts|yaml|yml|json|sql|toml|cfg|ini|md))\b")

    grounded = []
    ungrounded = []
    for risk in risks:
        desc = risk.get("description", "") + " " + risk.get("mitigation", "")
        refs = _FILE_REF.findall(desc)
        if not refs:
            # No file references — keep (it's about concepts, not specific files)
            grounded.append(risk)
            continue
        # Check if all referenced files appear somewhere in known context
        all_found = all(ref in known_text for ref in refs)
        if all_found:
            grounded.append(risk)
        else:
            ungrounded.append(risk)

    if not ungrounded:
        return merged

    # Never drop below minimum
    if len(grounded) < _MIN_RISKS:
        needed = _MIN_RISKS - len(grounded)
        grounded.extend(ungrounded[:needed])
        ungrounded = ungrounded[needed:]

    if ungrounded:
        removed_descs = [r.get("description", "")[:60] for r in ungrounded]
        logger.info(f"ensure_grounded_risks: removed {len(ungrounded)} ungrounded risks: {removed_descs}")

    merged["risks"] = grounded
    return merged


# ---------------------------------------------------------------------------
# Artifact code validators
# ---------------------------------------------------------------------------

def _extract_known_methods(reference_text: str) -> set[str]:
    """Extract known method/function names from signatures + library reference blocks."""
    methods: set[str] = set()
    for line in reference_text.splitlines():
        line = line.strip()
        # Match "  method_name(..." or "method_name(..." or "async method_name(..."
        m = re.match(r"(?:async\s+)?(\w+)\(", line)
        if m:
            methods.add(m.group(1))
        # Match "class Name: method1(...), method2(...), ..."
        if line.startswith("class ") and ":" in line:
            after_colon = line.split(":", 1)[1]
            for call_match in re.finditer(r"(\w+)\(", after_colon):
                methods.add(call_match.group(1))
    return methods


def _extract_method_calls_from_ast(tree: ast.Module) -> list[tuple[str, int]]:
    """Extract (method_name, line_number) from attribute calls in AST."""
    calls: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.append((node.func.attr, node.lineno))
    return calls


def ensure_valid_artifacts(
    merged: dict[str, Any],
    prior_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Validate Python artifacts: syntax check + method call cross-reference.

    Pure Python, no LLM. For each .py artifact:
    1. ast.parse() — catches syntax errors
    2. Cross-reference method calls against known signatures
    3. Annotate suspicious calls with [VERIFY] comments

    Never crashes — silently skips unparseable artifacts.
    """
    artifacts = merged.get("artifacts", [])
    if not artifacts:
        return merged

    # Build known methods from signatures + library reference
    raw_summaries = prior_outputs.get("_raw_summaries", "")
    # Extract the reference blocks (before first ### source file)
    first_source = raw_summaries.find("\n\n### ")
    reference = raw_summaries[:first_source] if first_source > 0 else ""
    known_methods = _extract_known_methods(reference)
    # Add common builtins/stdlib that are always valid
    known_methods.update({
        "append", "extend", "insert", "remove", "pop", "get", "set",
        "update", "items", "keys", "values", "strip", "split", "join",
        "format", "replace", "startswith", "endswith", "lower", "upper",
        "encode", "decode", "read", "write", "close", "open",
        "dumps", "loads", "dump", "load",  # json
        "info", "warning", "error", "debug", "exception",  # logging
        "add", "discard", "copy", "clear",
        "resolve", "exists", "is_file", "is_dir",  # pathlib
        "isinstance", "hasattr", "getattr", "setattr", "len", "str",
        "int", "float", "bool", "list", "dict", "tuple", "type",
        "print", "range", "enumerate", "zip", "map", "filter", "sorted",
        "run", "gather", "create_task", "sleep",  # asyncio
        "model_dump", "model_validate",  # pydantic
    })

    annotated = 0
    for artifact in artifacts:
        if not artifact.get("filename", "").endswith(".py"):
            continue

        content = artifact.get("content", "")
        if not content.strip():
            continue

        # 1. Syntax check
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            line_num = e.lineno or 1
            lines = content.splitlines()
            if 0 < line_num <= len(lines):
                lines[line_num - 1] += f"  # [SYNTAX ERROR: {e.msg}]"
                artifact["content"] = "\n".join(lines)
                annotated += 1
            continue

        # 2. Cross-reference method calls
        if not known_methods:
            continue

        calls = _extract_method_calls_from_ast(tree)
        suspicious: set[int] = set()
        for method_name, line_num in calls:
            if method_name.startswith("_"):
                continue
            if method_name not in known_methods:
                suspicious.add(line_num)

        if suspicious:
            lines = content.splitlines()
            for line_num in sorted(suspicious):
                if 0 < line_num <= len(lines):
                    if "[VERIFY]" not in lines[line_num - 1]:
                        lines[line_num - 1] += "  # [VERIFY] method not found in signatures"
                        annotated += 1
            artifact["content"] = "\n".join(lines)

    if annotated:
        logger.info(f"ensure_valid_artifacts: annotated {annotated} suspicious lines")

    return merged


async def ensure_correct_artifacts(
    merged: dict[str, Any],
    client: Any,
    prior_outputs: dict[str, Any],
) -> dict[str, Any]:
    """LLM review of Python artifacts for API correctness.

    One focused LLM call reviews all .py artifacts against the signatures
    reference. Returns corrected artifacts. Silent fallback on failure.
    """
    artifacts = merged.get("artifacts", [])
    py_artifacts = [a for a in artifacts if a.get("filename", "").endswith(".py")]
    if not py_artifacts:
        return merged

    # Get reference blocks (signatures + library API)
    raw_summaries = prior_outputs.get("_raw_summaries", "")
    first_source = raw_summaries.find("\n\n### ")
    reference = raw_summaries[:first_source] if first_source > 0 else ""

    if not reference:
        return merged

    artifact_text = "\n\n".join(
        f"--- {a['filename']} ---\n{a['content']}" for a in py_artifacts
    )

    prompt = (
        "Review these Python code artifacts for API correctness.\n"
        "For each artifact, check:\n"
        "1. Do all method calls use methods that actually exist on the object?\n"
        "2. Are parameter names and types correct?\n"
        "3. Are return types handled correctly (e.g., not treating str as dict)?\n"
        "4. Are stdlib/third-party APIs used correctly (real methods, not hallucinated)?\n"
        "5. Any lines marked [VERIFY] — are those calls correct or wrong?\n\n"
        f"REFERENCE (ground truth — these are the real APIs):\n{reference}\n\n"
        f"ARTIFACTS TO REVIEW:\n{artifact_text}\n\n"
        "Return corrected artifacts as JSON array:\n"
        '[{"filename": "path/to/file.py", "content": "corrected full file content"}]\n'
        "Only include artifacts that needed corrections. Return [] if all are correct."
    )

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw = await client.generate(messages=messages)
        corrections = extract_json(raw)
        if isinstance(corrections, list):
            fix_map = {
                c["filename"]: c["content"]
                for c in corrections
                if isinstance(c, dict) and "filename" in c and "content" in c
            }
            applied = 0
            for artifact in artifacts:
                if artifact.get("filename") in fix_map:
                    artifact["content"] = fix_map[artifact["filename"]]
                    applied += 1
            if applied:
                logger.info(f"ensure_correct_artifacts: LLM corrected {applied} artifacts")
    except Exception as e:
        logger.warning(f"ensure_correct_artifacts: LLM review failed: {e}")

    return merged
