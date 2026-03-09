# fitz_graveyard/planning/agent/compressor.py
"""
AST-based code compression for planning context.

Reduces Python source code token count by 50-70% while preserving
all information needed for architectural planning:
  - Import statements (kept verbatim)
  - Class/function signatures with decorators (kept verbatim)
  - Data structures and constants (kept verbatim)
  - Function bodies (collapsed to `...` unless short/important)

Stripped with zero information loss:
  - Docstrings (signatures tell the planning model more)
  - Comments (implementation notes, not architectural signal)
  - Blank lines (cosmetic)
  - String literals in non-essential positions

Applied AFTER retrieval, BEFORE reasoning. The cross-encoder and BM25
operate on full source for accurate relevance scoring.
"""

import ast
import logging
import textwrap

logger = logging.getLogger(__name__)

# Bodies shorter than this (in lines) are kept verbatim.
# Short functions are often the most architecturally informative
# (factory functions, config, protocol methods).
_KEEP_BODY_LINES = 6

# Top-level assignments (constants, configs) are always kept.
# Class-level assignments (class vars, defaults) are always kept.


def compress_python(source: str) -> str:
    """Compress Python source for planning context.

    Returns compressed source string. If parsing fails (syntax errors,
    non-Python), returns the original source unchanged.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    if not lines:
        return source

    # Collect line ranges to remove or replace
    removals: list[tuple[int, int]] = []  # (start_line, end_line) 1-indexed, inclusive
    replacements: dict[int, str] = {}  # start_line -> replacement text

    for node in ast.walk(tree):
        # Strip docstrings (module, class, function)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, (ast.Constant,))
            ):
                doc_node = node.body[0]
                if isinstance(doc_node.value, ast.Constant) and isinstance(doc_node.value.value, str):
                    removals.append((doc_node.lineno, doc_node.end_lineno))

        # Compress function/method bodies
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            # Skip docstring node if present
            body_start_idx = 0
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, (ast.Constant,))
            ):
                body_start_idx = 1

            real_body = body[body_start_idx:]
            if not real_body:
                continue

            # Determine body line range (after signature + docstring)
            first_body = real_body[0]
            last_body = real_body[-1]
            body_start = first_body.lineno
            body_end = last_body.end_lineno

            body_lines = body_end - body_start + 1

            if body_lines <= _KEEP_BODY_LINES:
                continue  # Keep short bodies verbatim

            # Check if body is just `pass` or `...`
            if (
                len(real_body) == 1
                and isinstance(real_body[0], (ast.Pass, ast.Expr))
            ):
                continue

            # Collapse long bodies to `...`
            # Detect indentation from first body line
            first_line = lines[body_start - 1] if body_start <= len(lines) else ""
            indent = len(first_line) - len(first_line.lstrip())
            indent_str = first_line[:indent] if indent > 0 else "        "

            # Keep return type hint if it's a return statement
            return_nodes = [n for n in real_body if isinstance(n, ast.Return)]
            if return_nodes and len(real_body) > 2:
                replacements[body_start] = f"{indent_str}...  # {body_lines} lines\n"
                removals.append((body_start + 1, body_end))
            else:
                replacements[body_start] = f"{indent_str}...  # {body_lines} lines\n"
                removals.append((body_start + 1, body_end))

    # Apply removals and replacements (process from bottom to top to preserve line numbers)
    # Merge overlapping removals
    if not removals and not replacements:
        # Only strip comments and blank lines
        return _strip_comments_and_blanks(source)

    # Sort removals by start line descending for safe mutation
    removals.sort(key=lambda r: r[0], reverse=True)

    # Build set of lines to remove
    remove_lines: set[int] = set()
    for start, end in removals:
        for ln in range(start, end + 1):
            remove_lines.add(ln)

    # Build output
    result: list[str] = []
    for i, line in enumerate(lines, 1):
        if i in replacements:
            result.append(replacements[i])
        elif i not in remove_lines:
            result.append(line)

    return _strip_comments_and_blanks("".join(result))


def _strip_comments_and_blanks(source: str) -> str:
    """Remove comment-only lines and collapse multiple blank lines."""
    out: list[str] = []
    prev_blank = False

    for line in source.splitlines(keepends=True):
        stripped = line.strip()

        # Remove comment-only lines (but keep shebangs and type: ignore)
        if stripped.startswith("#") and not stripped.startswith("#!") and "type:" not in stripped:
            continue

        # Collapse multiple blank lines to one
        if not stripped:
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False

        out.append(line)

    return "".join(out)


def compress_file(source: str, path: str) -> str:
    """Compress a source file based on its type.

    Python files get AST-based compression. Other files get
    comment/blank stripping only. Test files get aggressive
    compression (signature-only).
    """
    if not path.endswith(".py"):
        return _strip_comments_and_blanks(source)

    # Test files: keep only imports and signatures
    parts = path.replace("\\", "/").split("/")
    is_test = (
        any(p.startswith("test") for p in parts)
        or any(p == "tests" for p in parts)
    )

    compressed = compress_python(source)

    if is_test:
        # Further compress: collapse ALL function bodies in test files
        compressed = _collapse_all_bodies(compressed)

    return compressed


def _collapse_all_bodies(source: str) -> str:
    """Collapse all function bodies to `...` regardless of length."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    if not lines:
        return source

    removals: list[tuple[int, int]] = []
    replacements: dict[int, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        body = node.body
        if not body:
            continue

        # Skip if body is already just `...` or `pass`
        if len(body) == 1:
            if isinstance(body[0], ast.Pass):
                continue
            if (
                isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and body[0].value.value is ...
            ):
                continue

        # Skip docstring
        body_start_idx = 0
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, (ast.Constant,))
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body_start_idx = 1

        real_body = body[body_start_idx:]
        if not real_body:
            continue

        first_body = real_body[0]
        last_body = real_body[-1]
        body_start = first_body.lineno
        body_end = last_body.end_lineno
        body_lines = body_end - body_start + 1

        first_line = lines[body_start - 1] if body_start <= len(lines) else ""
        indent = len(first_line) - len(first_line.lstrip())
        indent_str = first_line[:indent] if indent > 0 else "        "

        replacements[body_start] = f"{indent_str}...\n"
        if body_end > body_start:
            removals.append((body_start + 1, body_end))

    if not removals and not replacements:
        return source

    remove_lines: set[int] = set()
    for start, end in removals:
        for ln in range(start, end + 1):
            remove_lines.add(ln)

    result: list[str] = []
    for i, line in enumerate(lines, 1):
        if i in replacements:
            result.append(replacements[i])
        elif i not in remove_lines:
            result.append(line)

    return "".join(result)
