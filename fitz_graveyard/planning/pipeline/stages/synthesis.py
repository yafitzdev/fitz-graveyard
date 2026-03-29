# fitz_graveyard/planning/pipeline/stages/synthesis.py
"""
Synthesis stage: narrate pre-solved decisions into the final plan.

Receives all committed decision records + constraints. The model is narrating
pre-solved problems, not discovering anything new. Uses existing per-field
extraction, self-critique, and coherence checking.

Output: same PlanOutput format (ContextOutput + ArchitectureOutput + DesignOutput
+ RoadmapOutput + RiskOutput).
"""

import difflib
import json
import logging
import time
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import (
    SYSTEM_PROMPT,
    PipelineStage,
    StageResult,
    extract_json,
)
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas import (
    ArchitectureOutput,
    ContextOutput,
    DesignOutput,
    RiskOutput,
    RoadmapOutput,
)

logger = logging.getLogger(__name__)

# Field groups for per-field extraction (same schemas as classic pipeline).

_CONTEXT_FIELD_GROUPS = [
    {
        "label": "description",
        "fields": ["project_description", "key_requirements", "constraints", "existing_context"],
        "schema": json.dumps({
            "project_description": "1-3 sentence specific description of what is being built",
            "key_requirements": ["concrete testable requirement 1", "requirement 2"],
            "constraints": ["real binding constraint 1", "constraint 2"],
            "existing_context": "existing codebase or tech context, or empty string if none",
        }, indent=2),
    },
    {
        "label": "stakeholders",
        "fields": ["stakeholders", "scope_boundaries"],
        "schema": json.dumps({
            "stakeholders": ["stakeholder with specific concern"],
            "scope_boundaries": {
                "in_scope": ["specific feature or capability"],
                "out_of_scope": ["explicitly excluded feature"],
            },
        }, indent=2),
    },
    {
        "label": "files",
        "fields": ["existing_files", "needed_artifacts"],
        "schema": json.dumps({
            "existing_files": ["path/to/relevant/file.py -- what it does"],
            "needed_artifacts": ["new_file.py -- what it produces (empty list [] if already implemented)"],
        }, indent=2),
    },
    {
        "label": "assumptions",
        "fields": ["assumptions"],
        "schema": json.dumps({
            "assumptions": [
                {"assumption": "what you assumed", "impact": "what changes if wrong", "confidence": "low|medium|high"}
            ],
        }, indent=2),
    },
]

_ARCH_FIELD_GROUPS = [
    {
        "label": "approaches",
        "fields": ["approaches", "recommended", "reasoning", "scope_statement"],
        "schema": json.dumps({
            "approaches": [
                {
                    "name": "Approach A",
                    "description": "What it looks like in production",
                    "pros": ["advantage"],
                    "cons": ["disadvantage"],
                    "complexity": "low|medium|high",
                    "best_for": ["scenario"],
                },
            ],
            "recommended": "must match one approach name exactly",
            "reasoning": "why this approach is right AND why the other is wrong",
            "scope_statement": "1-2 sentences characterizing the effort",
        }, indent=2),
    },
    {
        "label": "tradeoffs",
        "fields": ["key_tradeoffs", "technology_considerations"],
        "schema": json.dumps({
            "key_tradeoffs": {"tradeoff_name": "description"},
            "technology_considerations": ["technology with reason"],
        }, indent=2),
    },
]

_DESIGN_FIELD_GROUPS = [
    {
        "label": "adrs",
        "fields": ["adrs"],
        "schema": json.dumps({
            "adrs": [
                {
                    "title": "ADR: Decision Title",
                    "context": "What problem this solves",
                    "decision": "What was decided",
                    "rationale": "Why this is right",
                    "consequences": ["consequence"],
                    "alternatives_considered": ["Alternative -- rejected because reason"],
                }
            ],
        }, indent=2),
    },
    {
        "label": "components",
        "fields": ["components", "data_model"],
        "schema": json.dumps({
            "components": [
                {
                    "name": "ComponentName",
                    "purpose": "What it does",
                    "responsibilities": ["responsibility"],
                    "interfaces": ["methodName(param: Type) -> ReturnType"],
                    "dependencies": ["OtherComponent"],
                }
            ],
            "data_model": {"EntityName": ["field: type"]},
        }, indent=2),
    },
    {
        "label": "integrations",
        "fields": ["integration_points"],
        "schema": json.dumps({
            "integration_points": ["ExternalSystem -- what and how"],
        }, indent=2),
    },
    {
        "label": "artifacts",
        "fields": ["artifacts"],
        "schema": json.dumps({
            "artifacts": [
                {
                    "filename": "path/to/file",
                    "content": "complete file content",
                    "purpose": "why this artifact exists",
                }
            ],
        }, indent=2),
    },
]

_ROADMAP_FIELD_GROUPS = [
    {
        "label": "phases",
        "fields": ["phases"],
        "schema": json.dumps({
            "phases": [
                {
                    "number": 1,
                    "name": "Phase Name",
                    "objective": "What this phase achieves",
                    "deliverables": ["specific deliverable"],
                    "dependencies": [],
                    "estimated_complexity": "low|medium|high",
                    "key_risks": ["risk"],
                    "verification_command": "pytest tests/test_something.py -v",
                    "estimated_effort": "~2 hours",
                }
            ],
        }, indent=2),
    },
    {
        "label": "scheduling",
        "fields": ["critical_path", "parallel_opportunities", "total_phases"],
        "schema": json.dumps({
            "critical_path": [1, 2, 4],
            "parallel_opportunities": [[3, 5]],
            "total_phases": 5,
        }, indent=2),
    },
]

_RISK_FIELD_GROUPS = [
    {
        "label": "risks",
        "fields": ["risks", "overall_risk_level", "recommended_contingencies"],
        "schema": json.dumps({
            "risks": [
                {
                    "category": "technical|external|resource|schedule|quality|security",
                    "description": "What could go wrong",
                    "impact": "low|medium|high|critical",
                    "likelihood": "low|medium|high",
                    "mitigation": "Specific mitigation action",
                    "contingency": "What to do if it happens",
                    "affected_phases": [1, 3],
                    "verification": "assert something",
                }
            ],
            "overall_risk_level": "low|medium|high",
            "recommended_contingencies": ["contingency action"],
        }, indent=2),
    },
]


def _build_attribute_template(
    referenced_files: set[str],
    prior_outputs: dict[str, Any],
    sections: dict[str, list[str]],
) -> str:
    """Extract instance attributes from source code of referenced files.

    Parses __init__ and setup methods to find self._xxx = ClassName(...)
    assignments. Produces a compact template telling the model what
    attributes actually exist on each class.

    This prevents the model from fabricating method names like
    self._build_context() when the real attribute is self._assembler.
    """
    import ast as _ast

    agent_ctx = prior_outputs.get("_agent_context", {})
    file_contents = agent_ctx.get("file_contents", {})
    if not file_contents:
        return ""

    lines = [
        "\n## INSTANCE ATTRIBUTES — real self._ attributes on key classes\n"
        "When writing new methods on these classes, use ONLY the attributes "
        "listed below. Do NOT invent helper methods or attributes.\n"
    ]
    found_any = False

    for ref_path in sorted(referenced_files):
        # Find source content
        content = file_contents.get(ref_path, "")
        if not content:
            for key in file_contents:
                if key.endswith(ref_path) or ref_path.endswith(key):
                    content = file_contents[key]
                    break
        if not content:
            continue

        try:
            tree = _ast.parse(content)
        except SyntaxError:
            continue

        for cls_node in _ast.iter_child_nodes(tree):
            if not isinstance(cls_node, _ast.ClassDef):
                continue

            # Extract self._xxx = ... assignments from __init__ and setup methods
            attrs: dict[str, str] = {}  # attr_name -> type_hint
            for method in _ast.iter_child_nodes(cls_node):
                if not isinstance(method, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    continue
                if method.name not in ("__init__", "_init_components", "setup", "_setup"):
                    continue
                for node in _ast.walk(method):
                    if not isinstance(node, _ast.Assign):
                        continue
                    for target in node.targets:
                        if (isinstance(target, _ast.Attribute)
                                and isinstance(target.value, _ast.Name)
                                and target.value.id == "self"
                                and target.attr.startswith("_")
                                and not target.attr.startswith("__")):
                            # Resolve type from RHS
                            rhs = ""
                            if isinstance(node.value, _ast.Call):
                                if isinstance(node.value.func, _ast.Name):
                                    rhs = node.value.func.id
                                elif isinstance(node.value.func, _ast.Attribute):
                                    rhs = node.value.func.attr
                            if rhs:
                                attrs[target.attr] = rhs

            if not attrs:
                continue

            # Also get method names from the structural index for this class
            class_methods = []
            for idx_path, idx_lines in sections.items():
                if ref_path.endswith(idx_path) or idx_path.endswith(ref_path):
                    for line in idx_lines:
                        if line.startswith("classes:") and cls_node.name in line:
                            # Extract method list from brackets
                            import re
                            bracket_match = re.search(
                                rf'{cls_node.name}[^[]*\[([^\]]+)\]',
                                line,
                            )
                            if bracket_match:
                                for m in bracket_match.group(1).split(","):
                                    m = m.strip().split("->")[0].split("(")[0].strip()
                                    if m and not m.startswith("@"):
                                        class_methods.append(m)
                    break

            # Look up public methods for each component type.
            # Search both the structural index AND source code (for classes
            # not in the 30-file selected index).
            import re as _re
            component_methods: dict[str, list[str]] = {}  # type_name -> [method sigs]

            # Strategy 1: structural index
            for _idx_lines in sections.values():
                for _line in _idx_lines:
                    if not _line.startswith("classes:"):
                        continue
                    for type_name in set(attrs.values()):
                        if type_name not in _line:
                            continue
                        pattern = rf'{type_name}(?:\([^)]*\))?\s*(?:\[[^\]]*\])?\s*\[([^\]]+)\]'
                        _match = _re.search(pattern, _line)
                        if _match:
                            methods = []
                            for m in _match.group(1).split(","):
                                m = m.strip()
                                if m and not m.startswith("@") and not m.startswith("__"):
                                    methods.append(m)
                            if methods:
                                component_methods[type_name] = methods

            # Strategy 2: parse source files for classes not found in index
            # Search file_contents first, then fall back to disk via source_dir
            missing_types = set(attrs.values()) - set(component_methods.keys())
            source_dir = prior_outputs.get("_source_dir", "")
            all_sources = list((agent_ctx.get("file_contents") or {}).values())

            # Also read from disk for files not in the agent's pool
            if missing_types and source_dir:
                from pathlib import Path as _Path
                for _py in _Path(source_dir).rglob("*.py"):
                    if not missing_types:
                        break
                    # Quick check: does the filename hint at a missing type?
                    _stem = _py.stem.lower()
                    if not any(t.lower() in _stem or _stem in t.lower() for t in missing_types):
                        continue
                    try:
                        all_sources.append(_py.read_text(encoding="utf-8", errors="replace"))
                    except OSError:
                        continue

            for _src in all_sources:
                if not missing_types:
                    break
                try:
                    _tree = _ast.parse(_src)
                except SyntaxError:
                    continue
                for _node in _ast.walk(_tree):
                    if isinstance(_node, _ast.ClassDef) and _node.name in missing_types:
                        meths = []
                        for _child in _ast.iter_child_nodes(_node):
                            if isinstance(_child, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                                if _child.name.startswith("__"):
                                    continue
                                params = [a.arg for a in _child.args.args if a.arg != "self"]
                                sig = f"{_child.name}({', '.join(params)})"
                                if _child.returns:
                                    try:
                                        sig += f" -> {_ast.unparse(_child.returns)}"
                                    except Exception:
                                        pass
                                meths.append(sig)
                        if meths:
                            component_methods[_node.name] = meths
                            missing_types.discard(_node.name)

            lines.append(f"### {cls_node.name} ({ref_path})")
            lines.append("  Attributes:")
            for attr_name, type_name in sorted(attrs.items()):
                # Append component's public methods as inline comment
                comp_meths = component_methods.get(type_name, [])
                if comp_meths:
                    sig_str = ", ".join(comp_meths[:5])
                    lines.append(f"    self.{attr_name} = {type_name}  # has: {sig_str}")
                else:
                    lines.append(f"    self.{attr_name} = {type_name}(...)")
            if class_methods:
                lines.append(f"  Methods: {', '.join(class_methods)}")
            lines.append("")
            found_any = True

    if not found_any:
        return ""

    return "\n".join(lines)


class SynthesisStage(PipelineStage):
    """Synthesize resolved decisions into the final PlanOutput.

    The model receives ALL committed decision records and narrates them into
    a coherent plan. Then per-field extraction pulls structured data.

    This stage does NOT do original reasoning -- it organizes pre-solved answers.
    """

    @property
    def name(self) -> str:
        return "synthesis"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.75, 0.95)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any],
    ) -> list[dict]:
        resolution_output = prior_outputs.get("decision_resolution", {})
        resolutions = resolution_output.get("resolutions", [])

        decision_text = self._format_resolutions(resolutions)
        call_graph_text = prior_outputs.get("_call_graph_text", "")
        gathered_context = self._get_gathered_context(prior_outputs)

        prompt_template = load_prompt("synthesis")
        prompt = prompt_template.format(
            task_description=job_description,
            resolved_decisions=decision_text,
            call_graph=call_graph_text,
            gathered_context=gathered_context,
        )
        return self._make_messages(prompt)

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        return extract_json(raw_output)

    @staticmethod
    def _build_artifact_source_context(
        prior_outputs: dict[str, Any],
    ) -> str:
        """Build a focused cheat sheet of real symbols for artifact writing.

        Extracts class names, method names, and function signatures from the
        structural index for files referenced in decision resolutions. Compact
        (~30-50 lines) so the model knows what exists without being overwhelmed.
        """
        # Collect file paths from decision resolutions
        resolution_output = prior_outputs.get("decision_resolution", {})
        resolutions = resolution_output.get("resolutions", [])
        referenced_files: set[str] = set()
        for r in resolutions:
            for ev in r.get("evidence", []):
                if ":" in ev:
                    path = ev.split(":")[0].strip()
                    if path.endswith(".py"):
                        referenced_files.add(path)

        if not referenced_files:
            return ""

        # Get full structural index (covers all codebase files)
        full_index = prior_outputs.get(
            "_agent_context", {},
        ).get("full_structural_index", "")
        if not full_index:
            full_index = prior_outputs.get("_gathered_context", "")
        if not full_index:
            return ""

        # Parse index into per-file sections
        sections: dict[str, list[str]] = {}
        current_file = ""
        for line in full_index.split("\n"):
            if line.startswith("## "):
                current_file = line[3:].strip()
            elif current_file and line.strip():
                sections.setdefault(current_file, []).append(line)

        # Also include service/dependency files that define how the API
        # layer connects to the engine — models consistently miss this layer
        _SERVICE_KEYWORDS = ("service", "dependencies", "factory")
        for idx_path in sections:
            base = idx_path.rsplit("/", 1)[-1].replace(".py", "").lower()
            if any(kw in base for kw in _SERVICE_KEYWORDS):
                referenced_files.add(idx_path)

        # Build compact cheat sheet — only files referenced in decisions
        parts = [
            "## ARTIFACT REFERENCE — real symbols from the codebase\n"
            "When writing artifact code, use ONLY these class names, method "
            "names, field names, and function signatures. If a method is not "
            "listed here, it does NOT exist — do not invent it.\n"
        ]
        matched = 0
        for ref_path in sorted(referenced_files):
            # Match against index sections (may need partial match)
            for idx_path, idx_lines in sections.items():
                if ref_path == idx_path or ref_path.endswith(idx_path) or idx_path.endswith(ref_path):
                    parts.append(f"\n### {idx_path}")
                    for line in idx_lines:
                        # Include classes and functions lines, skip imports/exports
                        if line.startswith(("classes:", "functions:", "doc:")):
                            parts.append(f"  {line}")
                    matched += 1
                    break

        if matched == 0:
            return ""

        # Build instance attribute template from source code
        # This tells the model what self._ attributes ACTUALLY exist
        attr_template = _build_attribute_template(
            referenced_files, prior_outputs, sections,
        )
        if attr_template:
            parts.append(attr_template)

        result = "\n".join(parts)
        logger.info(
            f"Stage 'synthesis': artifact cheat sheet: "
            f"{len(referenced_files)} files referenced, "
            f"{matched} matched ({len(result)} chars)"
        )
        return result

    @staticmethod
    def _extract_class_names(
        reasoning: str,
        prior_outputs: dict[str, Any],
    ) -> list[str]:
        """Extract CamelCase class names from resolutions and reasoning.

        Returns sorted list of likely project class names, filtering
        out stdlib/framework names. Used by pre-fill and tool history
        injection.
        """
        import re

        resolution_output = prior_outputs.get("decision_resolution", {})
        resolutions = resolution_output.get("resolutions", [])

        # Extract CamelCase class names (2+ words, e.g. FitzKragEngine)
        camel = re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+)+)\b')
        names: set[str] = set()

        for r in resolutions:
            for ev in r.get("evidence", []):
                names.update(camel.findall(ev))
            names.update(camel.findall(r.get("decision", "")))
            names.update(camel.findall(r.get("reasoning", "")))

        # Also scan reasoning (synthesis output)
        names.update(camel.findall(reasoning[:8000]))

        # Filter out stdlib / framework / generic names
        _SKIP = {
            "True", "False", "None",
            "Optional", "Dict", "List", "Tuple", "Type", "Any", "Union",
            "Callable", "Iterator", "AsyncIterator", "Generator",
            "AsyncGenerator", "Sequence", "Mapping", "Iterable",
            "Exception", "ValueError", "TypeError", "KeyError",
            "AttributeError", "NotImplementedError", "RuntimeError",
            "ImportError", "FileNotFoundError", "IOError", "OSError",
            "FastAPI", "BaseModel", "APIRouter", "StreamingResponse",
            "JSONResponse", "HTTPException", "Depends", "Response",
            "ReturnType", "TypeVar", "FieldInfo", "ConfigDict",
        }
        names -= _SKIP

        return sorted(names)

    @staticmethod
    def _build_tool_history(
        class_names: list[str],
        tools_map: dict[str, Any],
    ) -> tuple[list[dict], dict[str, str]]:
        """Pre-call lookup_class and format results as tool history.

        Returns (messages_to_inject, seen_calls_dict). The messages
        look like the model already called lookup_class for each class,
        keeping the model in verification mode rather than passive
        reading mode.
        """
        lookup_class = tools_map.get("lookup_class")
        if not lookup_class or not class_names:
            return [], {}

        tool_calls_list = []
        tool_results = []
        seen: dict[str, str] = {}
        found = 0

        for i, name in enumerate(class_names):
            result = lookup_class(class_name=name)
            call_key = (
                f'lookup_class:'
                f'{json.dumps({"class_name": name}, sort_keys=True)}'
            )
            seen[call_key] = result

            if "NOT FOUND" in result:
                continue

            tc_id = f"pre_{i}"
            tool_calls_list.append({
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": "lookup_class",
                    "arguments": json.dumps({"class_name": name}),
                },
            })
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            })
            found += 1

        if not tool_calls_list:
            return [], seen

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_list,
            },
            *tool_results,
        ]
        logger.info(
            f"Stage 'synthesis': injected {found} class lookups "
            f"as tool history"
        )
        return messages, seen

    async def _build_artifacts_with_tools(
        self,
        client: Any,
        reasoning: str,
        prior_outputs: dict[str, Any],
        extract_context: str,
    ) -> tuple[list[dict] | None, str]:
        """Build artifacts using tool-assisted generation.

        The model gets codebase lookup tools (lookup_method, lookup_class,
        read_method_source) so it can verify real interfaces before writing
        code.

        Returns (artifacts, tool_context):
        - artifacts: list of artifact dicts if model produced JSON, else None
        - tool_context: formatted string of all tool results gathered,
          usable as enriched context for template fallback
        """
        if not hasattr(client, "generate_with_tools"):
            return None, ""

        # Build tools from the codebase context
        try:
            from fitz_graveyard.planning.pipeline.tools.codebase_tools import (
                make_codebase_tools,
            )
        except ImportError:
            return None, ""

        agent_ctx = prior_outputs.get("_agent_context", {})
        full_index = agent_ctx.get("full_structural_index", "")
        if not full_index:
            full_index = prior_outputs.get("_gathered_context", "")
        file_contents = agent_ctx.get("file_contents", {})
        source_dir = prior_outputs.get("_source_dir", "")

        if not full_index:
            return None, ""

        tools = make_codebase_tools(full_index, file_contents, source_dir)
        # Remove check_exists — model over-uses it (15+ calls per run),
        # checking stdlib types and re-checking things, causing degeneration.
        # lookup_class returns richer info anyway.
        tools = [fn for fn in tools if fn.__name__ != "check_exists"]
        tools_map = {fn.__name__: fn for fn in tools}

        schema = json.dumps({
            "artifacts": [{
                "filename": "path/to/file.py",
                "content": "ONLY the new methods/classes to add",
                "purpose": "why this artifact exists",
            }]
        }, indent=2)

        # Note: pre-calling lookup_class for resolution classes was tested
        # (run 29) and HURT scores (39.0 vs 43.4) because it seeds the
        # dedup cache, causing the model's organic calls to be flagged
        # as duplicates → earlier stale exit → less research time.
        # The model's organic research is more valuable than pre-filled info.

        prompt = (
            "You are writing implementation artifacts for a software plan.\n\n"
            "IMPORTANT: Before writing ANY code, use the lookup tools to "
            "verify real signatures:\n"
            "- lookup_method(class, method): get the REAL signature of "
            "an existing method\n"
            "- lookup_class(class): see real attributes and methods on "
            "a class\n"
            "- read_method_source(class, method): read the actual source "
            "code of a method\n\n"
            "Rules:\n"
            "- When adding a parallel method (e.g. generate_stream), call "
            "lookup_method\n"
            "  on the original (generate) FIRST to get the exact parameter "
            "list.\n"
            "- The parallel method MUST accept the same parameters.\n"
            "- Only use self._ attributes that lookup_class confirms "
            "exist.\n"
            "- Do NOT fabricate method names. If you are unsure whether a "
            "method exists, call lookup_class or lookup_method to check.\n\n"
            f"Return ONLY valid JSON matching this schema:\n{schema}\n\n"
            "--- PLAN ANALYSIS (what to build) ---\n"
            f"{reasoning[:8000]}\n\n"
            f"--- CODEBASE CONTEXT ---\n{extract_context[:4000]}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        max_rounds = 10
        t0 = time.monotonic()
        seen_calls: dict[str, str] = {}
        total_tool_calls = 0
        consecutive_stale = 0  # rounds with zero new info

        def _format_tool_context(calls: dict[str, str]) -> str:
            """Format collected tool results as context for template."""
            if not calls:
                return ""
            parts = [
                "## VERIFIED CODEBASE INFO (from tool lookups)\n"
                "The following was verified by looking up real classes "
                "and methods. Use these exact signatures and attributes.\n"
            ]
            for key, result in calls.items():
                if "NOT FOUND" in result or "Error:" in result:
                    continue
                parts.append(result)
            ctx = "\n\n".join(parts)
            return ctx if len(parts) > 1 else ""

        try:
            for round_num in range(max_rounds):
                response = await client.generate_with_tools(
                    messages, tools,
                )

                if not response.tool_calls:
                    # Model voluntarily produced final answer
                    raw = response.content or ""
                    elapsed = time.monotonic() - t0
                    tool_ctx = _format_tool_context(seen_calls)
                    logger.info(
                        f"Stage 'synthesis': tool-assisted artifacts "
                        f"(voluntary, {round_num} rounds, "
                        f"{total_tool_calls} calls, "
                        f"{elapsed:.1f}s, {len(raw)} chars)"
                    )
                    try:
                        data = extract_json(raw)
                        artifacts = data.get("artifacts", [])
                        if artifacts:
                            return artifacts, tool_ctx
                    except ValueError:
                        logger.warning(
                            f"Stage 'synthesis': tool-assisted artifacts "
                            f"failed to parse JSON"
                        )
                    return None, tool_ctx

                # Execute tool calls
                messages.append(response.assistant_dict)
                new_info_this_round = 0
                for tc in response.tool_calls:
                    # Normalize cache keys: strip module paths so
                    # lookup_class("fitz_ai.x.Foo") deduplicates with
                    # lookup_class("Foo") — they return the same data.
                    norm_args = {
                        k: (v.rsplit(".", 1)[-1] if isinstance(v, str) else v)
                        for k, v in tc.arguments.items()
                    }
                    call_key = (
                        f"{tc.name}:"
                        f"{json.dumps(norm_args, sort_keys=True)}"
                    )

                    if call_key in seen_calls:
                        result = seen_calls[call_key]
                        logger.info(
                            f"Stage 'synthesis': DUPLICATE tool call "
                            f"{tc.name} — returning cached"
                        )
                    else:
                        fn = tools_map.get(tc.name)
                        if fn:
                            try:
                                result = fn(**tc.arguments)
                            except Exception as e:
                                result = f"Error: {e}"
                            seen_calls[call_key] = result
                            total_tool_calls += 1
                            new_info_this_round += 1
                            logger.info(
                                f"Stage 'synthesis': tool call {tc.name}"
                                f"({', '.join(f'{k}={v!r}' for k, v in tc.arguments.items())}) "
                                f"-> {len(result)} chars"
                            )
                        else:
                            result = f"Unknown tool: {tc.name}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                # Early exit: 2 consecutive all-duplicate rounds means
                # the model has exhausted useful research. Return the
                # collected tool results for template enrichment.
                if new_info_this_round == 0:
                    consecutive_stale += 1
                else:
                    consecutive_stale = 0

                if consecutive_stale >= 2:
                    elapsed = time.monotonic() - t0
                    tool_ctx = _format_tool_context(seen_calls)
                    logger.info(
                        f"Stage 'synthesis': tool early exit — "
                        f"{consecutive_stale} stale rounds after "
                        f"round {round_num + 1}, "
                        f"{total_tool_calls} unique calls, "
                        f"{elapsed:.1f}s, {len(tool_ctx)} chars "
                        f"of verified context for template"
                    )
                    return None, tool_ctx

            # Exhausted rounds — return collected tool results
            elapsed = time.monotonic() - t0
            tool_ctx = _format_tool_context(seen_calls)
            logger.warning(
                f"Stage 'synthesis': tool-assisted artifacts exhausted "
                f"{max_rounds} rounds without producing JSON "
                f"({total_tool_calls} calls, {elapsed:.1f}s, "
                f"{len(tool_ctx)} chars of verified context)"
            )
            return None, tool_ctx

        except Exception as e:
            logger.warning(
                f"Stage 'synthesis': tool-assisted artifacts failed: {e}"
            )
            return None, ""

    def _format_resolutions(self, resolutions: list[dict]) -> str:
        """Format resolved decisions for the synthesis prompt."""
        lines = []
        for r in resolutions:
            lines.append(f"### Decision {r.get('decision_id', '?')}")
            lines.append(f"**Decided:** {r.get('decision', '')}")
            lines.append(f"**Reasoning:** {r.get('reasoning', '')}")
            evidence = r.get("evidence", [])
            if evidence:
                lines.append("**Evidence:**")
                for e in evidence:
                    lines.append(f"  - {e}")
            constraints = r.get("constraints_for_downstream", [])
            if constraints:
                lines.append("**Constraints:**")
                for c in constraints:
                    lines.append(f"  - {c}")
            lines.append("")
        return "\n".join(lines)

    async def execute(
        self,
        client: Any,
        job_description: str,
        prior_outputs: dict[str, Any],
    ) -> StageResult:
        try:
            # 1. Synthesis reasoning -- narrate pre-solved decisions
            messages = self.build_prompt(job_description, prior_outputs)
            await self._report_substep("synthesizing")
            t0 = time.monotonic()
            reasoning = await client.generate(messages=messages)
            t1 = time.monotonic()
            logger.info(
                f"Stage '{self.name}': synthesis took "
                f"{t1 - t0:.1f}s ({len(reasoning)} chars)"
            )

            # 2. Self-critique (catches formatting issues, not architectural)
            krag_context = self._get_gathered_context(prior_outputs)
            reasoning = await self._self_critique(
                client, reasoning, job_description, krag_context=krag_context,
            )

            # 3. Per-field extraction into all five schema sections
            extract_context = krag_context

            # Context fields
            context_merged: dict[str, Any] = {}
            for group in _CONTEXT_FIELD_GROUPS:
                extra = extract_context if group["label"] in {"files", "description"} else ""
                partial = await self._extract_field_group(
                    client, reasoning, group["fields"],
                    group["schema"], group["label"],
                    extra_context=extra,
                )
                context_merged.update(partial)

            # Architecture fields
            arch_merged: dict[str, Any] = {}
            for group in _ARCH_FIELD_GROUPS:
                partial = await self._extract_field_group(
                    client, reasoning, group["fields"],
                    group["schema"], group["label"],
                    extra_context=extract_context,
                )
                arch_merged.update(partial)

            # Design fields — artifacts use tool-assisted building when available
            design_merged: dict[str, Any] = {}
            for group in _DESIGN_FIELD_GROUPS:
                if group["label"] == "artifacts":
                    continue  # handled via tool-assisted loop below
                extra = extract_context if group["label"] in {"adrs", "components", "integrations"} else ""
                partial = await self._extract_field_group(
                    client, reasoning, group["fields"],
                    group["schema"], group["label"],
                    extra_context=extra,
                )
                design_merged.update(partial)

            # Artifact building: tool loop gathers verified codebase info,
            # then template extraction uses it for grounded artifacts.
            tool_artifacts, tool_context = await self._build_artifacts_with_tools(
                client, reasoning, prior_outputs, extract_context,
            )
            if tool_artifacts is not None:
                design_merged["artifacts"] = tool_artifacts
            else:
                # Template extraction enriched with tool-verified info
                artifact_source_context = self._build_artifact_source_context(
                    prior_outputs,
                )
                # Combine: cheat sheet + tool-verified signatures
                extra_parts = [
                    p for p in [artifact_source_context, tool_context]
                    if p
                ]
                extra = "\n\n".join(extra_parts) if extra_parts else extract_context
                partial = await self._extract_field_group(
                    client, reasoning, ["artifacts"],
                    _DESIGN_FIELD_GROUPS[-1]["schema"],
                    "artifacts",
                    extra_context=extra,
                )
                design_merged.update(partial)

            # Roadmap fields
            roadmap_merged: dict[str, Any] = {}
            for group in _ROADMAP_FIELD_GROUPS:
                extra = extract_context if group["label"] == "phases" else ""
                partial = await self._extract_field_group(
                    client, reasoning, group["fields"],
                    group["schema"], group["label"],
                    extra_context=extra,
                )
                roadmap_merged.update(partial)

            # Risk fields
            risk_merged: dict[str, Any] = {}
            for group in _RISK_FIELD_GROUPS:
                partial = await self._extract_field_group(
                    client, reasoning, group["fields"],
                    group["schema"], group["label"],
                    extra_context=extract_context,
                )
                risk_merged.update(partial)

            # 4. Validate through Pydantic
            context = ContextOutput(**context_merged).model_dump()

            # Handle recommended approach matching
            approach_names = [a["name"] for a in arch_merged.get("approaches", [])]
            recommended = arch_merged.get("recommended", "")
            if recommended not in approach_names and approach_names:
                matches = difflib.get_close_matches(
                    recommended, approach_names, n=1, cutoff=0.4,
                )
                if matches:
                    arch_merged["recommended"] = matches[0]
                else:
                    arch_merged["recommended"] = approach_names[0]

            arch_merged.setdefault("approaches", [])
            arch_merged.setdefault("recommended", "")
            arch_merged.setdefault("reasoning", "")
            arch_merged.setdefault("key_tradeoffs", {})
            arch_merged.setdefault("technology_considerations", [])
            arch_merged.setdefault("scope_statement", "")
            architecture = ArchitectureOutput(**arch_merged).model_dump()

            design_merged.setdefault("adrs", [])
            design_merged.setdefault("components", [])
            design_merged.setdefault("data_model", {})
            design_merged.setdefault("integration_points", [])
            design_merged.setdefault("artifacts", [])
            design = DesignOutput(**design_merged).model_dump()

            # Fix roadmap
            from fitz_graveyard.planning.pipeline.stages.roadmap_risk import (
                _remove_dependency_cycles,
            )
            if "phases" in roadmap_merged:
                for phase in roadmap_merged["phases"]:
                    if "num" in phase and "number" not in phase:
                        phase["number"] = phase.pop("num")
                roadmap_merged["phases"] = _remove_dependency_cycles(
                    roadmap_merged["phases"]
                )
            roadmap_merged.setdefault("phases", [])
            roadmap_merged.setdefault("critical_path", [])
            roadmap_merged.setdefault("parallel_opportunities", [])
            roadmap_merged.setdefault(
                "total_phases", len(roadmap_merged.get("phases", []))
            )
            roadmap = RoadmapOutput(**roadmap_merged).model_dump()

            risk_merged.setdefault("risks", [])
            risk_merged.setdefault("overall_risk_level", "medium")
            risk_merged.setdefault("recommended_contingencies", [])
            risk = RiskOutput(**risk_merged).model_dump()

            # 5. Combine into the output format expected by the orchestrator
            output = {
                "context": context,
                "architecture": architecture,
                "design": design,
                "roadmap": roadmap,
                "risk": risk,
            }

            return StageResult(
                stage_name=self.name,
                success=True,
                output=output,
                raw_output=reasoning,
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
