# fitz_graveyard/planning/agent/indexer.py
"""
Structural index builder for codebase context gathering.

Extracts structural information (classes, functions, imports) from source
files using language-appropriate methods:
  - Python: ast module (classes with bases, functions with signatures, imports)
  - Config (YAML/JSON/TOML): safe parsers for top-level keys
  - Markdown/RST: regex heading extraction
  - Generic code (JS/TS/Go/Rust/Java/C/C++/Ruby): regex patterns
  - Fallback: no structural info line

The index gives the LLM architectural visibility into the entire codebase
without reading full file contents, breaking the circular retrieval problem.
"""

import ast
import json
import logging
import re
from pathlib import Path, PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

# Maximum total index size in characters before truncation
_MAX_INDEX_CHARS = 80_000

# Extension → extractor mapping.
# INDEXABLE_EXTENSIONS is the union — used by the tree walker to skip files
# that the indexer can't extract anything useful from.
_PYTHON_EXTS = {".py"}
_CONFIG_EXTS = {".yaml", ".yml", ".json", ".toml"}
_MARKDOWN_EXTS = {".md", ".rst"}
_GENERIC_CODE_EXTS = {
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go",
    ".rs",
    ".java", ".kt", ".scala",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
    ".rb",
    ".cs",
    ".swift",
    ".php",
    ".lua",
    ".zig",
    ".ex", ".exs",
    ".erl", ".hrl",
    ".hs",
    ".ml", ".mli",
    ".sh", ".bash", ".zsh",
}

# All extensions the indexer can extract structure from.
# Files without these extensions are invisible to the agent.
INDEXABLE_EXTENSIONS = _PYTHON_EXTS | _CONFIG_EXTS | _MARKDOWN_EXTS | _GENERIC_CODE_EXTS


def build_structural_index(
    source_dir: str,
    file_list: list[str],
    max_file_bytes: int = 50_000,
) -> str:
    """
    Build a compact structural index of all files in the codebase.

    Args:
        source_dir: Absolute path to source directory root.
        file_list: List of relative file paths (posix-style) to index.
        max_file_bytes: Maximum bytes to read per file.

    Returns:
        Multi-line text index with structural info per file.
    """
    root = Path(source_dir).resolve()
    entries: list[tuple[str, str]] = []  # (rel_path, index_text)

    for rel_path in file_list:
        full_path = root / rel_path
        if not full_path.is_file():
            continue

        try:
            raw = full_path.read_bytes()[:max_file_bytes]
            content = raw.decode("utf-8", errors="replace")
        except OSError:
            continue

        if not content.strip():
            continue

        suffix = PurePosixPath(rel_path).suffix.lower()
        info = _extract_structure(suffix, content, rel_path)
        if info:
            entries.append((rel_path, info))
        else:
            entries.append((rel_path, "(no structural info)"))

    # Format and apply size budget
    return _format_index(entries)


def _extract_structure(suffix: str, content: str, rel_path: str) -> str:
    """Dispatch to the appropriate extractor based on file extension."""
    if suffix in _PYTHON_EXTS:
        return _extract_python(content)
    if suffix in _CONFIG_EXTS:
        return _extract_config(suffix, content)
    if suffix in _MARKDOWN_EXTS:
        return _extract_markdown(content)
    if suffix in _GENERIC_CODE_EXTS:
        return _extract_generic_code(content)
    return ""


def _extract_python(content: str) -> str:
    """Extract structure from Python files using AST.

    Falls back to regex if AST parsing fails (syntax errors).
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _extract_python_regex(content)

    lines: list[str] = []

    # Classes with bases and method names
    classes = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                bases.append(_ast_name(base))
            methods = [
                n.name for n in ast.iter_child_nodes(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            cls_str = node.name
            if bases:
                cls_str += f"({', '.join(bases)})"
            if methods:
                cls_str += f" [{', '.join(methods)}]"
            classes.append(cls_str)
    if classes:
        lines.append(f"classes: {'; '.join(classes)}")

    # Top-level functions with parameter names
    functions = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [arg.arg for arg in node.args.args if arg.arg != "self"]
            func_str = f"{node.name}({', '.join(params)})"
            functions.append(func_str)
    if functions:
        lines.append(f"functions: {', '.join(functions)}")

    # Imports — walk full tree to catch TYPE_CHECKING and conditional imports.
    # Show full dotted paths for intra-project imports so the LLM can trace
    # architectural connections (e.g. "fitz_ai.llm.providers.base" not "fitz_ai").
    # External packages still use top-level name only.
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Keep full path for multi-segment (intra-project) imports,
                # top-level only for stdlib/external (single segment).
                if "." in node.module:
                    imports.add(node.module)
                else:
                    imports.add(node.module)
    if imports:
        lines.append(f"imports: {', '.join(sorted(imports))}")

    # __all__ exports
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        names = [
                            elt.value for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
                        if names:
                            lines.append(f"exports: {', '.join(names)}")

    return "\n".join(lines)


def _ast_name(node: ast.expr) -> str:
    """Get a human-readable name from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _ast_name(node.value)
        return f"{value}.{node.attr}" if value else node.attr
    if isinstance(node, ast.Subscript):
        return _ast_name(node.value)
    return "?"


def _extract_python_regex(content: str) -> str:
    """Fallback Python extraction using regex when AST fails."""
    lines: list[str] = []

    classes = re.findall(r'^class\s+(\w+)(?:\(([^)]*)\))?:', content, re.MULTILINE)
    if classes:
        cls_strs = []
        for name, bases in classes:
            cls_strs.append(f"{name}({bases})" if bases else name)
        lines.append(f"classes: {'; '.join(cls_strs)}")

    functions = re.findall(r'^(?:async\s+)?def\s+(\w+)\(([^)]*)\)', content, re.MULTILINE)
    if functions:
        func_strs = [f"{name}({params})" for name, params in functions]
        lines.append(f"functions: {', '.join(func_strs)}")

    imports = set()
    for m in re.finditer(r'^(?:from\s+(\S+)\s+)?import\s+(\S+)', content, re.MULTILINE):
        mod = m.group(1) or m.group(2)
        imports.add(mod.split(".")[0])
    if imports:
        lines.append(f"imports: {', '.join(sorted(imports))}")

    return "\n".join(lines)


def _extract_config(suffix: str, content: str) -> str:
    """Extract top-level keys from config files."""
    try:
        if suffix in (".yaml", ".yml"):
            # Only import yaml if needed — it's optional
            import yaml
            data = yaml.safe_load(content)
        elif suffix == ".json":
            data = json.loads(content)
        elif suffix == ".toml":
            import tomllib
            data = tomllib.loads(content)
        else:
            return ""
    except Exception:
        return ""

    if isinstance(data, dict):
        keys = list(data.keys())[:20]  # Cap at 20 keys
        return f"keys: {', '.join(str(k) for k in keys)}"
    return ""


def _extract_markdown(content: str) -> str:
    """Extract headings from markdown/RST files."""
    # Markdown headings
    headings = re.findall(r'^(#{1,3})\s+(.+)', content, re.MULTILINE)
    if headings:
        items = [f"{'#' * len(h[0])} {h[1].strip()}" for h in headings[:15]]
        return f"headings: {'; '.join(items)}"

    # RST headings (line of = or - under text)
    rst_headings = re.findall(r'^(.+)\n[=\-~^]+$', content, re.MULTILINE)
    if rst_headings:
        items = [h.strip() for h in rst_headings[:15]]
        return f"headings: {'; '.join(items)}"

    return ""


def _extract_generic_code(content: str) -> str:
    """Extract structure from non-Python code using regex patterns.

    Covers: JS/TS, Go, Rust, Java/Kotlin, C/C++, Ruby, C#, Swift, PHP, etc.
    """
    lines: list[str] = []

    # Classes / structs / interfaces / traits / enums
    type_defs = re.findall(
        r'^(?:export\s+)?(?:pub\s+)?(?:public\s+|private\s+|protected\s+|abstract\s+|sealed\s+)?'
        r'(?:class|struct|interface|trait|enum|type)\s+'
        r'(\w+)',
        content, re.MULTILINE,
    )
    if type_defs:
        lines.append(f"types: {', '.join(dict.fromkeys(type_defs))}")

    # Functions / methods (various languages)
    func_patterns = [
        # Go: func Name(
        r'^func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(',
        # Rust: fn name(  / pub fn name(
        r'^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*[<(]',
        # JS/TS: function name( / export function name(
        r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*[<(]',
        # JS/TS: const name = (...) => / const name = function
        r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*=>|function)',
        # Java/C#/Kotlin: public void name( / fun name(
        r'^\s+(?:public|private|protected|static|override|virtual|abstract|final|\s)*'
        r'(?:fun|void|int|string|bool|float|double|var|val|Task|async)\s+(\w+)\s*[<(]',
        # Ruby: def name
        r'^(?:\s+)?def\s+(\w+)',
        # C/C++: type name( at start of line (heuristic)
        r'^(?:static\s+)?(?:inline\s+)?(?:const\s+)?\w[\w:*&<> ]*\s+(\w+)\s*\([^;]*$',
    ]
    functions: list[str] = []
    for pat in func_patterns:
        for m in re.finditer(pat, content, re.MULTILINE):
            name = m.group(1)
            if name not in functions and name not in ("if", "for", "while", "switch", "return", "main"):
                functions.append(name)
    if functions:
        lines.append(f"functions: {', '.join(functions[:20])}")

    # Imports / requires / use statements
    imports: set[str] = set()
    import_patterns = [
        r'^import\s+["\']([^"\']+)["\']',           # JS/TS import "x"
        r'^import\s+.*\s+from\s+["\']([^"\']+)["\']', # JS/TS import x from "y"
        r'^(?:const|let|var)\s+.*=\s*require\(["\']([^"\']+)["\']\)',  # Node require
        r'^import\s+"([^"]+)"',                       # Go import
        r'^use\s+([\w:]+)',                            # Rust use
        r'^import\s+([\w.]+)',                         # Java/Kotlin
        r'^using\s+([\w.]+)',                          # C#
        r'^require\s+["\']([^"\']+)["\']',             # Ruby
        r'^#include\s+[<"]([^>"]+)[>"]',               # C/C++
    ]
    for pat in import_patterns:
        for m in re.finditer(pat, content, re.MULTILINE):
            mod = m.group(1).split("/")[0].split("::")[0].split(".")[0]
            if mod:
                imports.add(mod)
    if imports:
        lines.append(f"imports: {', '.join(sorted(imports))}")

    return "\n".join(lines)


def _format_index(entries: list[tuple[str, str]]) -> str:
    """Format entries into the final index text, truncating if over budget.

    Strategy: never drop files entirely.  If over budget, progressively
    reduce detail — first strip imports, then strip function lists — from
    the *shallowest* (least specific) files, preserving full detail on
    deeper (more architecturally specific) files.  As a last resort,
    reduce deep files too.
    """
    parts: list[str] = []
    for rel_path, info in entries:
        parts.append(f"## {rel_path}\n{info}")

    full = "\n\n".join(parts)

    if len(full) <= _MAX_INDEX_CHARS:
        return full

    # Over budget — strip detail from shallowest files first.
    # Work on a mutable copy sorted shallowest-first.
    mutable = list(entries)  # preserve original order for output
    by_depth = sorted(range(len(mutable)), key=lambda i: mutable[i][0].count("/"))

    # Pass 1: strip imports lines from shallowest files first
    for idx in by_depth:
        rel_path, info = mutable[idx]
        lines = [ln for ln in info.splitlines() if not ln.startswith("imports:")]
        mutable[idx] = (rel_path, "\n".join(lines))
        if _estimate_size(mutable) <= _MAX_INDEX_CHARS:
            break

    # Pass 2: strip functions lines from shallowest files first
    if _estimate_size(mutable) > _MAX_INDEX_CHARS:
        for idx in by_depth:
            rel_path, info = mutable[idx]
            lines = [ln for ln in info.splitlines() if not ln.startswith("functions:")]
            mutable[idx] = (rel_path, "\n".join(lines))
            if _estimate_size(mutable) <= _MAX_INDEX_CHARS:
                break

    # Pass 3: last resort — reduce to path-only for shallowest files
    if _estimate_size(mutable) > _MAX_INDEX_CHARS:
        for idx in by_depth:
            mutable[idx] = (mutable[idx][0], "")
            if _estimate_size(mutable) <= _MAX_INDEX_CHARS:
                break

    result_parts = []
    for rel_path, info in mutable:
        if info.strip():
            result_parts.append(f"## {rel_path}\n{info}")
        else:
            result_parts.append(f"## {rel_path}")

    return "\n\n".join(result_parts)


def _estimate_size(entries: list[tuple[str, str]]) -> int:
    """Estimate formatted index size without building the full string."""
    total = 0
    for rel_path, info in entries:
        total += 3 + len(rel_path) + 1 + len(info) + 2  # "## " + path + "\n" + info + "\n\n"
    return total


# ---------------------------------------------------------------------------
# Full import graph (for reverse-import caller expansion)
# ---------------------------------------------------------------------------


def _extract_full_imports(content: str) -> set[str]:
    """Extract full dotted import paths from Python source (AST, regex fallback)."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _extract_full_imports_regex(content)

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def _extract_full_imports_regex(content: str) -> set[str]:
    """Regex fallback for full import paths.

    Unlike the structural index regex (top-level only), this matches
    indented imports too — critical for lazy imports inside functions.
    """
    imports: set[str] = set()
    for m in re.finditer(r'^\s*from\s+(\S+)\s+import', content, re.MULTILINE):
        imports.add(m.group(1))
    for m in re.finditer(r'^\s*import\s+(\S+)', content, re.MULTILINE):
        imports.add(m.group(1).split(",")[0].strip())
    return imports


def _build_module_file_lookup(file_list: list[str]) -> dict[str, str]:
    """Build module_dotted_path -> relative_file_path lookup.

    E.g.: "fitz_ai.governance.governor" -> "fitz_ai/governance/governor.py"
    Also maps package inits: "fitz_ai.governance" -> "fitz_ai/governance/__init__.py"
    """
    lookup: dict[str, str] = {}
    for rel_path in file_list:
        if not rel_path.endswith(".py"):
            continue
        module = rel_path[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            lookup[module] = rel_path
            lookup[module[:-9]] = rel_path  # strip ".__init__"
        else:
            lookup[module] = rel_path
    return lookup


def build_import_graph(
    source_dir: str,
    file_list: list[str],
    max_file_bytes: int = 50_000,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Build forward import map and module lookup.

    Returns:
        (forward_map, module_lookup) where:
        - forward_map: {file_path: {resolved_file_path, ...}} — only intra-project imports
        - module_lookup: {dotted.module: file_path}
    """
    root = Path(source_dir).resolve()
    module_lookup = _build_module_file_lookup(file_list)
    forward: dict[str, set[str]] = {}

    for rel_path in file_list:
        if not rel_path.endswith(".py"):
            continue
        full_path = root / rel_path
        if not full_path.is_file():
            continue
        try:
            raw = full_path.read_bytes()[:max_file_bytes]
            content = raw.decode("utf-8", errors="replace")
        except OSError:
            continue

        full_imports = _extract_full_imports(content)
        resolved = set()
        for imp in full_imports:
            target = module_lookup.get(imp)
            if target and target != rel_path:
                resolved.add(target)

        if resolved:
            forward[rel_path] = resolved

    return forward, module_lookup


# ---------------------------------------------------------------------------
# Two-tier directory clustering (for large codebases, ≥100 files)
# ---------------------------------------------------------------------------

_CLUSTERING_THRESHOLD = 100


def _group_by_directory(
    file_list: list[str], max_depth: int = 2,
) -> dict[str, list[str]]:
    """Group file paths by their directory prefix up to *max_depth* levels.

    Root-level files (no directory) are grouped under ``"(root)"``.

    Args:
        file_list: Relative posix-style paths.
        max_depth: Maximum directory depth for grouping (default 2).

    Returns:
        Mapping of directory prefix → list of file paths in that group.
    """
    groups: dict[str, list[str]] = {}
    for rel_path in file_list:
        parts = PurePosixPath(rel_path).parts
        if len(parts) <= 1:
            key = "(root)"
        else:
            key = "/".join(parts[: min(max_depth, len(parts) - 1)]) + "/"
        groups.setdefault(key, []).append(rel_path)
    return groups


def build_directory_clusters(
    source_dir: str,
    file_list: list[str],
    max_depth: int = 2,
    max_file_bytes: int = 50_000,
) -> tuple[str, dict[str, list[str]]]:
    """Build aggregated directory-level summaries for LLM directory selection.

    For each directory cluster, aggregates class names, function names, and
    imports from the files it contains (reusing ``_extract_structure``).

    Args:
        source_dir:     Absolute path to source root.
        file_list:      Relative posix-style paths.
        max_depth:      Directory grouping depth (default 2).
        max_file_bytes: Max bytes per file for extraction.

    Returns:
        Tuple of (formatted_text, groups_dict).
        *formatted_text* is the prompt-ready cluster summary.
        *groups_dict* maps dir prefix → list of file paths.
    """
    root = Path(source_dir).resolve()
    groups = _group_by_directory(file_list, max_depth)

    lines: list[str] = []
    for dir_prefix in sorted(groups):
        files = groups[dir_prefix]
        all_classes: list[str] = []
        all_functions: list[str] = []
        all_imports: set[str] = set()

        for rel_path in files:
            full_path = root / rel_path
            if not full_path.is_file():
                continue
            try:
                raw = full_path.read_bytes()[:max_file_bytes]
                content = raw.decode("utf-8", errors="replace")
            except OSError:
                continue
            if not content.strip():
                continue

            suffix = PurePosixPath(rel_path).suffix.lower()
            info = _extract_structure(suffix, content, rel_path)
            if not info:
                continue

            for line in info.splitlines():
                if line.startswith("classes:"):
                    all_classes.extend(
                        c.strip() for c in line[len("classes:"):].split(";") if c.strip()
                    )
                elif line.startswith("functions:"):
                    all_functions.extend(
                        f.strip() for f in line[len("functions:"):].split(",") if f.strip()
                    )
                elif line.startswith("imports:"):
                    all_imports.update(
                        i.strip() for i in line[len("imports:"):].split(",") if i.strip()
                    )

        entry = f"## {dir_prefix}  ({len(files)} files)"
        detail_lines: list[str] = []
        if all_classes:
            detail_lines.append(f"classes: {'; '.join(all_classes[:15])}")
        if all_functions:
            detail_lines.append(f"functions: {', '.join(all_functions[:15])}")
        if all_imports:
            detail_lines.append(f"imports: {', '.join(sorted(all_imports)[:10])}")

        if detail_lines:
            entry += "\n" + "\n".join(detail_lines)
        lines.append(entry)

    text = "\n\n".join(lines)
    logger.info(
        f"Clustered {len(file_list)} files into {len(groups)} directories "
        f"({len(text)} chars)"
    )
    return text, groups
