# fitz_graveyard/planning/pipeline/tools/codebase_tools.py
"""
Codebase lookup tools for tool-assisted artifact building.

These tools let the LLM look up real method signatures, class attributes,
and source code during artifact generation — preventing fabrication by
grounding every reference in the actual codebase.

All tools are pure Python (no LLM calls). They query the structural index
and source code that was already gathered by the agent.
"""

import ast
import logging
from pathlib import Path

from fitz_graveyard.planning.validation.grounding import StructuralIndexLookup

logger = logging.getLogger(__name__)


def make_codebase_tools(
    structural_index: str,
    file_contents: dict[str, str],
    source_dir: str | None = None,
) -> list:
    """Create codebase lookup tools bound to the gathered context.

    Returns a list of callables suitable for generate_with_tools().
    """
    lookup = StructuralIndexLookup(structural_index)

    # Build a source pool: file_contents + disk fallback
    _source_pool = dict(file_contents) if file_contents else {}

    def _find_source(class_name: str) -> str | None:
        """Find source code containing a class definition."""
        # Search file_contents first — must have the actual class def, not just an import
        class_marker = f"class {class_name}"
        for path, content in _source_pool.items():
            if class_marker in content:
                return content
        # Disk fallback — match class name against filename in both directions
        if source_dir:
            cn_lower = class_name.lower()
            for py in Path(source_dir).rglob("*.py"):
                # Skip venvs and hidden directories
                parts_str = str(py)
                if ".venv" in parts_str or "__pycache__" in parts_str or "site-packages" in parts_str:
                    continue
                stem = py.stem.lower()
                # Match: class name contains stem or stem contains class name
                # Require minimum 4-char stem to avoid matching 'c.py', 'de.py' etc.
                if len(stem) >= 4 and (cn_lower in stem or stem in cn_lower):
                    try:
                        src = py.read_text(encoding="utf-8", errors="replace")
                        if f"class {class_name}" in src:
                            return src
                    except OSError:
                        continue
        return None

    def _find_class_node(src: str, class_name: str) -> ast.ClassDef | None:
        """Find AST ClassDef node by name."""
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        return None

    def _format_method_sig(method: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Format a method's full signature from AST."""
        params = []
        for arg in method.args.args:
            if arg.arg == "self":
                continue
            p = arg.arg
            if arg.annotation:
                try:
                    p += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            params.append(p)
        # Defaults
        defaults = method.args.defaults
        n_defaults = len(defaults)
        n_params = len(params)
        for i, default in enumerate(defaults):
            param_idx = n_params - n_defaults + i
            if param_idx >= 0:
                try:
                    params[param_idx] += f" = {ast.unparse(default)}"
                except Exception:
                    pass

        sig = f"{method.name}({', '.join(params)})"
        if method.returns:
            try:
                sig += f" -> {ast.unparse(method.returns)}"
            except Exception:
                pass
        return sig

    def _strip_module(name: str) -> str:
        """Strip module path from a class/function name.

        Models often pass fully-qualified names like
        'fitz_ai.engines.fitz_krag.engine.FitzKragEngine' but tools
        index by simple name 'FitzKragEngine'. Also strips method
        names passed as 'ClassName.method'.
        """
        if "." in name:
            return name.rsplit(".", 1)[-1]
        return name

    # ------------------------------------------------------------------
    # Tool 1: lookup_method
    # ------------------------------------------------------------------
    def lookup_method(class_name: str, method_name: str) -> str:
        """Look up the full signature of a method on a class in the codebase."""
        class_name = _strip_module(class_name)
        method_name = _strip_module(method_name)
        # Try AST from source code first (most accurate)
        src = _find_source(class_name)
        if src:
            cls_node = _find_class_node(src, class_name)
            if cls_node:
                for node in ast.iter_child_nodes(cls_node):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name == method_name:
                            sig = _format_method_sig(node)
                            return f"{class_name}.{sig}"

        # Fall back to structural index
        if lookup.class_has_method(class_name, method_name):
            for cls in lookup.find_classes(class_name):
                if method_name in cls.methods:
                    ret = cls.methods[method_name].return_type
                    ret_str = f" -> {ret}" if ret else ""
                    return f"{class_name}.{method_name}(...){ret_str} (params not available from index)"

        # Check top-level functions
        funcs = lookup.find_function(method_name)
        if funcs:
            f = funcs[0]
            params = ", ".join(f.params) if f.params else ""
            ret = f" -> {f.return_type}" if f.return_type else ""
            return f"{method_name}({params}){ret} [top-level function in {f.file}]"

        return f"METHOD NOT FOUND: {class_name}.{method_name}() does not exist in the codebase. Do not use it."

    # ------------------------------------------------------------------
    # Tool 2: lookup_class
    # ------------------------------------------------------------------
    def lookup_class(class_name: str) -> str:
        """Look up a class: its methods, instance attributes, and base classes."""
        class_name = _strip_module(class_name)
        parts = []

        # AST-based (most accurate)
        src = _find_source(class_name)
        if src:
            cls_node = _find_class_node(src, class_name)
            if cls_node:
                # Bases
                bases = [ast.unparse(b) if hasattr(ast, 'unparse') else '?'
                         for b in cls_node.bases]
                if bases:
                    parts.append(f"class {class_name}({', '.join(bases)})")
                else:
                    parts.append(f"class {class_name}")

                # Instance attrs from __init__ / _init_components
                attrs = []
                for method in ast.iter_child_nodes(cls_node):
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if method.name in ("__init__", "_init_components"):
                            for node in ast.walk(method):
                                if isinstance(node, ast.Assign):
                                    for target in node.targets:
                                        if (isinstance(target, ast.Attribute)
                                                and isinstance(target.value, ast.Name)
                                                and target.value.id == "self"):
                                            rhs = ""
                                            if isinstance(node.value, ast.Call):
                                                if isinstance(node.value.func, ast.Name):
                                                    rhs = node.value.func.id
                                                elif isinstance(node.value.func, ast.Attribute):
                                                    rhs = node.value.func.attr
                                            attrs.append(f"  self.{target.attr} = {rhs}(...)" if rhs
                                                         else f"  self.{target.attr}")
                if attrs:
                    parts.append("Attributes:")
                    parts.extend(attrs[:20])  # cap at 20

                # Class-level annotated fields (Pydantic models, dataclasses)
                # e.g. question: str = Field(...)
                fields = []
                for node in ast.iter_child_nodes(cls_node):
                    if isinstance(node, ast.AnnAssign) and node.target:
                        if isinstance(node.target, ast.Name):
                            name = node.target.id
                            try:
                                ann = ast.unparse(node.annotation)
                            except Exception:
                                ann = "?"
                            fields.append(f"  {name}: {ann}")
                if fields:
                    parts.append("Fields:")
                    parts.extend(fields[:20])

                # Methods with signatures
                methods = []
                for node in ast.iter_child_nodes(cls_node):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not node.name.startswith("__"):
                            methods.append(f"  {_format_method_sig(node)}")
                if methods:
                    parts.append("Methods:")
                    parts.extend(methods[:15])  # cap at 15

                return "\n".join(parts)

        # Fall back to structural index
        cls = lookup.find_class(class_name)
        if cls:
            parts.append(f"class {class_name} (from structural index, {cls.file})")
            if cls.bases:
                parts.append(f"  Bases: {', '.join(cls.bases)}")
            if cls.methods:
                parts.append("  Methods:")
                for name, m in cls.methods.items():
                    ret = f" -> {m.return_type}" if m.return_type else ""
                    parts.append(f"    {name}{ret}")
            return "\n".join(parts)

        return f"CLASS NOT FOUND: {class_name} does not exist in the codebase."

    # ------------------------------------------------------------------
    # Tool 3: check_exists
    # ------------------------------------------------------------------
    def check_exists(symbol_name: str) -> str:
        """Check if a class, method, or function exists anywhere in the codebase."""
        # Check classes
        if lookup.class_exists(symbol_name):
            cls = lookup.find_class(symbol_name)
            return f"EXISTS: class {symbol_name} in {cls.file}"

        # Check functions
        if lookup.function_exists(symbol_name):
            funcs = lookup.find_function(symbol_name)
            return f"EXISTS: function {symbol_name} in {funcs[0].file}"

        # Check as a method on any class
        if lookup.method_exists_anywhere(symbol_name):
            for cls_name, cls_list in lookup.classes.items():
                for cls in cls_list:
                    if symbol_name in cls.methods:
                        return f"EXISTS: method {cls_name}.{symbol_name}() in {cls.file}"

        return f"DOES NOT EXIST: no class, function, or method named '{symbol_name}' found in the codebase. Do not use it."

    # ------------------------------------------------------------------
    # Tool 4: read_method_source
    # ------------------------------------------------------------------
    def read_method_source(class_name: str, method_name: str) -> str:
        """Read the actual source code of a method (up to 2000 chars)."""
        class_name = _strip_module(class_name)
        method_name = _strip_module(method_name)
        src = _find_source(class_name)
        if not src:
            return f"SOURCE NOT AVAILABLE for {class_name}"

        cls_node = _find_class_node(src, class_name)
        if not cls_node:
            return f"CLASS {class_name} NOT FOUND in source"

        for node in ast.iter_child_nodes(cls_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == method_name:
                    # Extract source lines
                    lines = src.split("\n")
                    start = node.lineno - 1
                    end = node.end_lineno or (start + 50)
                    method_src = "\n".join(lines[start:end])
                    if len(method_src) > 2000:
                        method_src = method_src[:2000] + "\n... (truncated)"
                    return method_src

        return f"METHOD {method_name} NOT FOUND on {class_name}"

    logger.info(
        f"Created 4 codebase tools (index: {len(lookup.classes)} classes, "
        f"{len(lookup.functions)} functions, sources: {len(_source_pool)} files)"
    )

    return [lookup_method, lookup_class, check_exists, read_method_source]
