# fitz_graveyard/planning/validation/grounding.py
"""
Post-synthesis grounding validator.

Two-path validation of plan artifacts against the target codebase:
  Path 1 (AST): Deterministic — parses artifacts, checks every symbol
         reference against the structural index.
  Path 2 (LLM): Architectural — checks for missing layers, unaddressed
         integration points, gaps the AST can't see.
"""

import ast
import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structural index parser
# ---------------------------------------------------------------------------

@dataclass
class IndexedMethod:
    name: str
    return_type: str | None = None


@dataclass
class IndexedClass:
    name: str
    file: str
    bases: list[str] = field(default_factory=list)
    methods: dict[str, IndexedMethod] = field(default_factory=dict)
    decorators: list[str] = field(default_factory=list)


@dataclass
class IndexedFunction:
    name: str
    file: str
    params: list[str] = field(default_factory=list)
    return_type: str | None = None


# Regex for parsing index class entries:
#   ClassName(Base1, Base2) [@dec1, @dec2] [method1 -> Type, method2]
_CLASS_RE = re.compile(
    r"([A-Za-z_]\w*)"           # class name
    r"(?:\(([^)]*)\))?"         # optional bases
    r"(?:\s*\[([^\]]*)\])?"     # optional first bracket (decorators or methods)
    r"(?:\s*\[([^\]]*)\])?"     # optional second bracket (methods if first was decorators)
)

# Regex for parsing index function entries:
#   func_name(param1, param2) -> ReturnType [@decorator]
_FUNC_RE = re.compile(
    r"([A-Za-z_]\w*)"           # function name
    r"\(([^)]*)\)"              # params
    r"(?:\s*->\s*([^[,]+))?"    # optional return type
)


class StructuralIndexLookup:
    """Parsed structural index for programmatic symbol queries."""

    def __init__(self, index_text: str):
        self.classes: dict[str, list[IndexedClass]] = {}  # name -> list (may exist in multiple files)
        self.functions: dict[str, list[IndexedFunction]] = {}
        self._all_method_names: set[str] = set()
        self._all_class_names: set[str] = set()
        self._all_function_names: set[str] = set()
        self._parse(index_text)

    def _parse(self, text: str) -> None:
        current_file = ""
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("## "):
                current_file = line[3:].strip()
                continue
            if not current_file:
                continue

            if line.startswith("classes: "):
                self._parse_classes(line[9:], current_file)
            elif line.startswith("functions: "):
                self._parse_functions(line[11:], current_file)

    def _parse_classes(self, text: str, file: str) -> None:
        # Classes separated by "; "
        for cls_text in text.split("; "):
            cls_text = cls_text.strip()
            if not cls_text:
                continue
            m = _CLASS_RE.match(cls_text)
            if not m:
                continue

            name = m.group(1)
            bases = [b.strip() for b in m.group(2).split(",") if b.strip()] if m.group(2) else []
            decorators: list[str] = []
            methods: dict[str, IndexedMethod] = {}

            # Parse bracket contents — could be decorators or methods
            for bracket in (m.group(3), m.group(4)):
                if not bracket:
                    continue
                bracket = bracket.strip()
                if bracket.startswith("@"):
                    decorators = [d.strip().lstrip("@") for d in bracket.split(",")]
                else:
                    for method_str in bracket.split(", "):
                        method_str = method_str.strip()
                        if not method_str:
                            continue
                        if " -> " in method_str:
                            mname, ret = method_str.split(" -> ", 1)
                            methods[mname.strip()] = IndexedMethod(mname.strip(), ret.strip())
                        else:
                            methods[method_str] = IndexedMethod(method_str)

            cls = IndexedClass(name, file, bases, methods, decorators)
            self.classes.setdefault(name, []).append(cls)
            self._all_class_names.add(name)
            for mname in methods:
                self._all_method_names.add(mname)

    def _parse_functions(self, text: str, file: str) -> None:
        for func_text in text.split(", "):
            # Need to handle cases where return type contains commas
            # Try matching from the start
            func_text = func_text.strip()
            if not func_text:
                continue
            m = _FUNC_RE.match(func_text)
            if not m:
                continue

            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            ret = m.group(3).strip() if m.group(3) else None

            func = IndexedFunction(name, file, params, ret)
            self.functions.setdefault(name, []).append(func)
            self._all_function_names.add(name)

    def find_class(self, name: str) -> IndexedClass | None:
        entries = self.classes.get(name, [])
        return entries[0] if entries else None

    def find_classes(self, name: str) -> list[IndexedClass]:
        return self.classes.get(name, [])

    def find_function(self, name: str) -> list[IndexedFunction]:
        return self.functions.get(name, [])

    def class_has_method(self, class_name: str, method_name: str) -> bool:
        for cls in self.classes.get(class_name, []):
            if method_name in cls.methods:
                return True
        return False

    def method_exists_anywhere(self, method_name: str) -> bool:
        return method_name in self._all_method_names

    def class_exists(self, name: str) -> bool:
        return name in self._all_class_names

    def function_exists(self, name: str) -> bool:
        return name in self._all_function_names

    def function_params(self, name: str) -> list[str] | None:
        funcs = self.functions.get(name, [])
        return funcs[0].params if funcs else None

    def suggest_method(self, name: str) -> list[str]:
        return difflib.get_close_matches(name, self._all_method_names, n=3, cutoff=0.6)

    def suggest_class(self, name: str) -> list[str]:
        return difflib.get_close_matches(name, self._all_class_names, n=3, cutoff=0.6)

    def suggest_function(self, name: str) -> list[str]:
        return difflib.get_close_matches(name, self._all_function_names, n=3, cutoff=0.6)


# ---------------------------------------------------------------------------
# Violation dataclass
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    artifact: str
    line: int
    symbol: str
    kind: str  # missing_method, missing_class, missing_function, wrong_arity
    detail: str
    suggestion: str = ""


# ---------------------------------------------------------------------------
# Path 1: AST grounding check
# ---------------------------------------------------------------------------

# Names to skip — builtins, common stdlib, test patterns
_SKIP_NAMES = frozenset({
    "None", "True", "False", "self", "cls", "super",
    "str", "int", "float", "bool", "list", "dict", "set", "tuple", "bytes",
    "type", "object", "Exception", "ValueError", "TypeError", "KeyError",
    "RuntimeError", "AttributeError", "NotImplementedError", "StopIteration",
    "OSError", "IOError", "FileNotFoundError", "ImportError", "IndexError",
    "print", "len", "range", "enumerate", "zip", "map", "filter", "sorted",
    "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "any", "all", "min", "max", "sum", "abs", "round", "hash", "id", "repr",
    "open", "iter", "next",
    "Optional", "Union", "Any", "List", "Dict", "Set", "Tuple",
    "Iterator", "Generator", "AsyncGenerator", "Callable", "Type",
    "Sequence", "Mapping", "Iterable",
    "Path", "logging", "json", "re", "os", "sys", "time",
    "dataclass", "field", "dataclasses",
    "BaseModel", "Field", "ConfigDict",
    "APIRouter", "Request", "StreamingResponse", "Depends",
    "router", "app",
})

# Common stdlib/third-party module prefixes to skip
_SKIP_PREFIXES = ("typing.", "collections.", "abc.", "functools.", "asyncio.")


def check_artifact(
    artifact: dict[str, Any],
    lookup: StructuralIndexLookup,
) -> list[Violation]:
    """Parse artifact content with AST and validate symbol references."""
    filename = artifact.get("filename", "unknown")
    content = artifact.get("content", "")

    if not content.strip():
        return []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return [Violation(filename, 0, "", "parse_error",
                         "Artifact is not valid Python — cannot validate")]

    violations: list[Violation] = []

    # Collect classes defined IN this artifact (so self.X checks use the right class)
    artifact_classes: dict[str, set[str]] = {}  # class_name -> set of method names
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = set()
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.add(child.name)
            artifact_classes[node.name] = methods

    # Determine the target class (the class in the target file being modified)
    # by matching artifact filename to structural index
    target_classes: list[IndexedClass] = []
    for cls_list in lookup.classes.values():
        for cls in cls_list:
            if cls.file == filename or filename.endswith(cls.file):
                target_classes.append(cls)

    # Walk AST and check references
    for node in ast.walk(tree):
        _check_node(node, filename, lookup, artifact_classes,
                     target_classes, violations)

    return violations


def _check_node(
    node: ast.AST,
    filename: str,
    lookup: StructuralIndexLookup,
    artifact_classes: dict[str, set[str]],
    target_classes: list[IndexedClass],
    violations: list[Violation],
) -> None:
    line = getattr(node, "lineno", 0)

    # Check: self.method() calls
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"):
        method_name = node.func.attr
        if method_name.startswith("_") and method_name.startswith("__"):
            return  # skip dunder
        # Check if method exists in artifact's own class definitions
        for cls_methods in artifact_classes.values():
            if method_name in cls_methods:
                return
        # Check if method exists in target file's classes (from index)
        for cls in target_classes:
            if method_name in cls.methods:
                return
        # Method not found — check if it exists anywhere
        if not lookup.method_exists_anywhere(method_name):
            suggestions = lookup.suggest_method(method_name)
            violations.append(Violation(
                filename, line, f"self.{method_name}()",
                "missing_method",
                f"Method '{method_name}' not found on any class in the target file "
                f"or anywhere in the codebase index",
                f"Did you mean: {', '.join(suggestions)}" if suggestions else "",
            ))

    # Check: standalone function calls (not method calls)
    elif (isinstance(node, ast.Call)
          and isinstance(node.func, ast.Name)):
        func_name = node.func.id
        if func_name in _SKIP_NAMES:
            return
        if func_name[0].isupper():
            # Looks like a class constructor
            if not lookup.class_exists(func_name):
                # Check if it's defined in the artifact itself
                if func_name not in artifact_classes:
                    suggestions = lookup.suggest_class(func_name)
                    violations.append(Violation(
                        filename, line, func_name,
                        "missing_class",
                        f"Class '{func_name}' not found in codebase index",
                        f"Did you mean: {', '.join(suggestions)}" if suggestions else "",
                    ))
        else:
            # Looks like a function call — check existence and arity
            if not lookup.function_exists(func_name):
                # Could be an imported function not in the index
                # Only flag if it looks like a codebase function (not stdlib)
                if not any(func_name.startswith(p) for p in ("_", "pytest")):
                    suggestions = lookup.suggest_function(func_name)
                    if suggestions:
                        violations.append(Violation(
                            filename, line, f"{func_name}()",
                            "missing_function",
                            f"Function '{func_name}' not found in codebase index",
                            f"Did you mean: {', '.join(suggestions)}",
                        ))
            else:
                # Check arity
                expected_params = lookup.function_params(func_name)
                if expected_params is not None:
                    actual_args = len(node.args) + len(node.keywords)
                    expected = len(expected_params)
                    # Allow some slack for *args/**kwargs and defaults
                    if actual_args > 0 and expected > 0 and abs(actual_args - expected) > 2:
                        violations.append(Violation(
                            filename, line, f"{func_name}()",
                            "wrong_arity",
                            f"Called with {actual_args} args but index shows "
                            f"{expected} params: ({', '.join(expected_params)})",
                        ))

    # Check: attribute access on known types (obj.field)
    elif (isinstance(node, ast.Attribute)
          and isinstance(node.value, ast.Name)
          and node.value.id not in _SKIP_NAMES
          and node.value.id != "self"):
        # Skip module-level attribute access (json.loads, etc)
        if node.value.id[0].islower() and node.value.id not in ("request", "response", "ctx"):
            return
        # For known variable names like 'request', we'd need type info
        # This is deferred to the LLM path


def check_all_artifacts(
    artifacts: list[dict[str, Any]],
    structural_index: str,
) -> list[Violation]:
    """Run AST grounding check + parallel method signature check."""
    if not artifacts:
        return []

    lookup = StructuralIndexLookup(structural_index)
    all_violations: list[Violation] = []

    for artifact in artifacts:
        violations = check_artifact(artifact, lookup)
        all_violations.extend(violations)

    # Check parallel method signatures (e.g. generate_stream must match generate)
    all_violations.extend(_check_parallel_signatures(artifacts, lookup))

    return all_violations


# Common streaming parallel suffixes
_PARALLEL_SUFFIXES = ("_stream", "_async", "_streaming", "stream_")


def _check_parallel_signatures(
    artifacts: list[dict[str, Any]],
    lookup: StructuralIndexLookup,
) -> list[Violation]:
    """Check that parallel methods (e.g. generate_stream) match the original's params."""
    violations: list[Violation] = []

    for artifact in artifacts:
        content = artifact.get("content", "")
        filename = artifact.get("filename", "unknown")
        if not content.strip():
            continue

        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # Check if this is a parallel method (name ends with _stream etc.)
            method_name = node.name
            original_name = None
            for suffix in _PARALLEL_SUFFIXES:
                if method_name.endswith(suffix):
                    original_name = method_name[: -len(suffix)]
                    break
                if method_name.startswith(suffix):
                    original_name = method_name[len(suffix):]
                    break

            if not original_name:
                continue

            # Find the original method in the structural index
            original_params = None
            for cls_list in lookup.classes.values():
                for cls in cls_list:
                    if original_name in cls.methods:
                        # Get param count from the index method signature
                        # Methods are stored as IndexedMethod(name, return_type)
                        # but we need params — check via the lookup
                        pass

            # Count params on the new parallel method
            new_params = [a.arg for a in node.args.args if a.arg != "self"]

            # Also check against top-level functions
            original_funcs = lookup.find_function(original_name)
            if original_funcs:
                orig_params = original_funcs[0].params
                if len(new_params) < len(orig_params) - 2:  # allow 2 fewer (defaults)
                    violations.append(Violation(
                        artifact=filename,
                        line=node.lineno,
                        symbol=method_name,
                        kind="param_mismatch",
                        detail=(
                            f"Parallel method {method_name}() has {len(new_params)} params "
                            f"but original {original_name}() has {len(orig_params)}: "
                            f"({', '.join(orig_params)}). "
                            f"Parallel methods should accept the same parameters."
                        ),
                        suggestion=f"Add missing params from {original_name}()",
                    ))

    return violations


# ---------------------------------------------------------------------------
# Path 2: LLM architectural gap check
# ---------------------------------------------------------------------------

_LLM_GROUNDING_PROMPT = """You are validating a software plan's artifacts against the actual codebase.

## AST-Detected Violations
The following symbol reference errors were found deterministically:

{violations_text}

## Artifacts Being Validated
{artifacts_summary}

## Structural Index (relevant sections)
{structural_index_excerpt}

## Resolved Decisions
{decisions_summary}

## Instructions
Based on the violations above and your understanding of the codebase structure, identify:
1. **Missing layers**: Does the plan skip any intermediate layer in the call chain? (e.g., API → Service → Engine, but plan goes API → Engine directly)
2. **Missing files**: Are there files that need modification but aren't listed as artifacts?
3. **Wrong assumptions**: Do the artifacts assume helper methods exist that would need to be created first?

Be specific. Reference real file paths and method names from the structural index.
Return a JSON object:
{{
  "missing_layers": ["description of missing layer"],
  "missing_files": ["file.py — why it needs changes"],
  "wrong_assumptions": ["what the artifact assumes vs reality"],
  "summary": "1-2 sentence overall assessment"
}}"""


def _format_violations(violations: list[Violation]) -> str:
    if not violations:
        return "(none detected)"
    lines = []
    for v in violations:
        line = f"- {v.artifact}:{v.line} — {v.kind}: {v.symbol} — {v.detail}"
        if v.suggestion:
            line += f" ({v.suggestion})"
        lines.append(line)
    return "\n".join(lines)


def _format_artifacts_summary(artifacts: list[dict]) -> str:
    lines = []
    for a in artifacts:
        content = a.get("content", "")
        line_count = content.count("\n") + 1
        lines.append(f"- {a.get('filename', '?')} ({line_count} lines): {a.get('purpose', '')}")
    return "\n".join(lines)


def _format_decisions_summary(resolutions: list[dict]) -> str:
    lines = []
    for r in resolutions:
        did = r.get("decision_id", "?")
        decision = r.get("decision", "")[:200]
        lines.append(f"- {did}: {decision}")
    return "\n".join(lines)


def build_llm_grounding_prompt(
    violations: list[Violation],
    artifacts: list[dict],
    structural_index: str,
    resolutions: list[dict],
) -> str:
    # Trim structural index to relevant sections — files mentioned in artifacts
    artifact_files = {a.get("filename", "") for a in artifacts}
    relevant_sections = []
    current_section: list[str] = []
    current_file = ""
    for line in structural_index.split("\n"):
        if line.startswith("## "):
            if current_section and current_file:
                # Check if this file is relevant
                for af in artifact_files:
                    if af in current_file or current_file in af:
                        relevant_sections.extend(current_section)
                        break
            current_file = line[3:].strip()
            current_section = [line]
        else:
            current_section.append(line)
    # Flush last
    if current_section and current_file:
        for af in artifact_files:
            if af in current_file or current_file in af:
                relevant_sections.extend(current_section)
                break

    index_excerpt = "\n".join(relevant_sections) if relevant_sections else structural_index[:5000]

    return _LLM_GROUNDING_PROMPT.format(
        violations_text=_format_violations(violations),
        artifacts_summary=_format_artifacts_summary(artifacts),
        structural_index_excerpt=index_excerpt,
        decisions_summary=_format_decisions_summary(resolutions),
    )


# ---------------------------------------------------------------------------
# Combined validator
# ---------------------------------------------------------------------------

@dataclass
class GroundingReport:
    """Combined output from both validation paths."""
    ast_violations: list[Violation]
    llm_gaps: dict[str, Any] | None = None  # parsed LLM response
    total_violations: int = 0

    def to_dict(self) -> dict:
        return {
            "ast_violations": [
                {
                    "artifact": v.artifact,
                    "line": v.line,
                    "symbol": v.symbol,
                    "kind": v.kind,
                    "detail": v.detail,
                    "suggestion": v.suggestion,
                }
                for v in self.ast_violations
            ],
            "llm_gaps": self.llm_gaps,
            "total_violations": self.total_violations,
        }


async def validate_grounding(
    artifacts: list[dict[str, Any]],
    structural_index: str,
    resolutions: list[dict[str, Any]],
    client: Any | None = None,
) -> GroundingReport:
    """Run both validation paths and return combined report.

    Args:
        artifacts: Plan artifacts (filename, content, purpose)
        structural_index: Full structural index text
        resolutions: Decision resolutions from decomposed pipeline
        client: LLM client for Path 2 (None = skip LLM validation)
    """
    # Path 1: AST
    ast_violations = check_all_artifacts(artifacts, structural_index)
    logger.info(
        f"Grounding AST check: {len(ast_violations)} violations "
        f"across {len(artifacts)} artifacts"
    )
    for v in ast_violations:
        logger.info(f"  {v.artifact}:{v.line} {v.kind}: {v.symbol} — {v.detail}")

    # Path 2: LLM (optional)
    llm_gaps = None
    if client is not None:
        try:
            prompt = build_llm_grounding_prompt(
                ast_violations, artifacts, structural_index, resolutions,
            )
            messages = [
                {"role": "system", "content": "You are a code reviewer validating plan artifacts against a real codebase."},
                {"role": "user", "content": prompt},
            ]
            raw = await client.generate(messages=messages, max_tokens=4096)
            # Try to parse as JSON
            import json
            from fitz_graveyard.planning.pipeline.stages.base import extract_json
            try:
                llm_gaps = extract_json(raw)
            except ValueError:
                llm_gaps = {"raw": raw[:2000], "parse_error": True}
            logger.info(f"Grounding LLM check: {llm_gaps.get('summary', 'no summary')}")
        except Exception as e:
            logger.warning(f"Grounding LLM check failed (non-fatal): {e}")
            llm_gaps = {"error": str(e)}

    return GroundingReport(
        ast_violations=ast_violations,
        llm_gaps=llm_gaps,
        total_violations=len(ast_violations),
    )
