# tests/unit/test_indexer.py
"""Unit tests for the structural index builder."""

import json
from pathlib import Path

import pytest

from fitz_graveyard.planning.agent.indexer import (
    _MAX_INDEX_CHARS,
    _build_module_file_lookup,
    _extract_config,
    _extract_full_imports,
    _extract_full_imports_regex,
    _extract_generic_code,
    _extract_markdown,
    _extract_python,
    _extract_python_regex,
    _extract_signatures_from_python,
    _parse_class_hierarchies,
    _parse_simple_return_methods,
    _group_by_directory,
    build_directory_clusters,
    build_import_graph,
    build_structural_index,
    extract_interface_signatures,
    extract_library_signatures,
    generate_investigation_questions,
)


# ---------------------------------------------------------------------------
# Python AST extraction
# ---------------------------------------------------------------------------
class TestExtractPython:
    def test_class_with_bases_and_methods(self):
        code = """
class MyProvider(ChatProvider):
    def chat(self, messages):
        pass

    def stream(self):
        pass
"""
        result = _extract_python(code)
        assert "MyProvider(ChatProvider)" in result
        assert "chat, stream" in result

    def test_top_level_functions_with_params(self):
        code = """
def create_provider(spec, config):
    pass

async def resolve_auth(provider):
    pass
"""
        result = _extract_python(code)
        assert "create_provider(spec, config)" in result
        assert "resolve_auth(provider)" in result

    def test_imports(self):
        code = """
import os
import json
from pathlib import Path
from fitz_ai.llm.providers.base import ChatProvider
"""
        result = _extract_python(code)
        assert "imports:" in result
        assert "os" in result
        assert "json" in result
        assert "pathlib" in result
        assert "fitz_ai" in result

    def test_all_exports(self):
        code = """
__all__ = ["get_chat", "get_embedder", "ModelTier"]
"""
        result = _extract_python(code)
        assert "exports:" in result
        assert "get_chat" in result
        assert "ModelTier" in result

    def test_empty_file(self):
        assert _extract_python("") == ""

    def test_syntax_error_falls_back_to_regex(self):
        # Invalid Python but has class-like structure
        code = """
class Broken(:
    def method(self):
        >>>invalid<<<
"""
        result = _extract_python(code)
        # Should fall back to regex and still find something
        assert "Broken" in result or result == ""

    def test_self_excluded_from_params(self):
        code = """
def process(self, data, options):
    pass
"""
        result = _extract_python(code)
        assert "process(data, options)" in result  # self always filtered

    def test_class_method_self_listed(self):
        code = """
class Foo:
    def bar(self, x):
        pass
"""
        result = _extract_python(code)
        assert "Foo" in result
        assert "bar" in result


# ---------------------------------------------------------------------------
# Python regex fallback
# ---------------------------------------------------------------------------
class TestExtractPythonRegex:
    def test_finds_classes(self):
        code = "class Foo(Bar):\n    pass\nclass Baz:\n    pass"
        result = _extract_python_regex(code)
        assert "Foo(Bar)" in result
        assert "Baz" in result

    def test_finds_functions(self):
        code = "def hello(name, age):\n    pass\nasync def fetch(url):\n    pass"
        result = _extract_python_regex(code)
        assert "hello(name, age)" in result
        assert "fetch(url)" in result

    def test_finds_imports(self):
        code = "import os\nfrom pathlib import Path"
        result = _extract_python_regex(code)
        assert "os" in result
        assert "pathlib" in result


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------
class TestExtractConfig:
    def test_json_keys(self):
        data = json.dumps({"name": "test", "version": "1.0", "plugins": []})
        result = _extract_config(".json", data)
        assert "keys:" in result
        assert "name" in result
        assert "version" in result

    def test_toml_keys(self):
        content = '[project]\nname = "test"\n\n[tool.pytest]\naddopts = "-v"'
        result = _extract_config(".toml", content)
        assert "keys:" in result
        assert "project" in result

    def test_invalid_json(self):
        result = _extract_config(".json", "not json")
        assert result == ""

    def test_yaml_keys(self):
        content = "provider: ollama\nmodel: qwen\nagent:\n  enabled: true"
        result = _extract_config(".yaml", content)
        assert "keys:" in result
        assert "provider" in result

    def test_non_dict_yaml(self):
        content = "- item1\n- item2"
        result = _extract_config(".yaml", content)
        assert result == ""


# ---------------------------------------------------------------------------
# Markdown extraction
# ---------------------------------------------------------------------------
class TestExtractMarkdown:
    def test_headings(self):
        content = "# Title\n## Section One\n### Subsection\nBody text"
        result = _extract_markdown(content)
        assert "headings:" in result
        assert "# Title" in result
        assert "## Section One" in result

    def test_rst_headings(self):
        content = "Title\n=====\n\nSection\n-------\n"
        result = _extract_markdown(content)
        assert "headings:" in result
        assert "Title" in result

    def test_no_headings(self):
        result = _extract_markdown("just plain text\nno headings here")
        assert result == ""


# ---------------------------------------------------------------------------
# Generic code extraction
# ---------------------------------------------------------------------------
class TestExtractGenericCode:
    def test_go_func(self):
        code = 'package main\n\nimport "fmt"\n\nfunc handleRequest(w http.ResponseWriter) {\n}'
        result = _extract_generic_code(code)
        assert "handleRequest" in result
        assert "fmt" in result

    def test_rust_struct_and_fn(self):
        code = "pub struct Config {\n}\n\npub fn parse(input: &str) -> Config {\n}"
        result = _extract_generic_code(code)
        assert "Config" in result
        assert "parse" in result

    def test_js_function_and_import(self):
        code = 'import express from "express"\n\nexport function createApp() {\n}'
        result = _extract_generic_code(code)
        assert "createApp" in result
        assert "express" in result

    def test_java_class(self):
        code = "public class UserService {\n    public void getUser() {}\n}"
        result = _extract_generic_code(code)
        assert "UserService" in result

    def test_typescript_interface(self):
        code = "export interface Config {\n  name: string\n}"
        result = _extract_generic_code(code)
        assert "Config" in result

    def test_c_include(self):
        code = '#include <stdio.h>\n#include "myheader.h"\n\nint main() {}'
        result = _extract_generic_code(code)
        assert "stdio" in result or "myheader" in result


# ---------------------------------------------------------------------------
# build_structural_index (integration)
# ---------------------------------------------------------------------------
class TestBuildStructuralIndex:
    def test_indexes_python_file(self, tmp_path):
        (tmp_path / "main.py").write_text(
            "class App:\n    def run(self):\n        pass\n"
        )
        result = build_structural_index(str(tmp_path), ["main.py"])
        assert "## main.py" in result
        assert "App" in result
        assert "run" in result

    def test_indexes_json_config(self, tmp_path):
        (tmp_path / "config.json").write_text('{"host": "localhost", "port": 8080}')
        result = build_structural_index(str(tmp_path), ["config.json"])
        assert "## config.json" in result
        assert "host" in result

    def test_skips_missing_files(self, tmp_path):
        result = build_structural_index(str(tmp_path), ["nonexistent.py"])
        assert result == ""

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.py").write_text("")
        result = build_structural_index(str(tmp_path), ["empty.py"])
        assert result == ""

    def test_fallback_for_unknown_extension(self, tmp_path):
        (tmp_path / "data.xyz").write_text("some data")
        result = build_structural_index(str(tmp_path), ["data.xyz"])
        assert "## data.xyz" in result
        assert "(no structural info)" in result

    def test_multiple_files(self, tmp_path):
        (tmp_path / "a.py").write_text("def foo(): pass")
        (tmp_path / "b.py").write_text("class Bar: pass")
        result = build_structural_index(str(tmp_path), ["a.py", "b.py"])
        assert "## a.py" in result
        assert "## b.py" in result
        assert "foo" in result
        assert "Bar" in result

    def test_truncation_on_budget(self, tmp_path):
        # Create many files to exceed budget
        files = []
        for i in range(200):
            name = f"pkg/{'sub/' * 5}module_{i}.py"
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write enough code to have substantial index entries
            path.write_text(
                f"class LongClassName{i}(BaseClass):\n"
                f"    def method_a_{i}(self): pass\n"
                f"    def method_b_{i}(self): pass\n"
                f"import os\nimport json\nimport pathlib\n" * 10
            )
            files.append(name.replace("\\", "/"))

        result = build_structural_index(str(tmp_path), files, max_file_bytes=50_000)
        assert len(result) <= _MAX_INDEX_CHARS + 200  # small buffer for last entry


# ---------------------------------------------------------------------------
# _group_by_directory
# ---------------------------------------------------------------------------
class TestGroupByDirectory:
    def test_groups_by_two_level_prefix(self):
        files = [
            "src/pkg/main.py",
            "src/pkg/util.py",
            "src/other/app.py",
            "tests/unit/test_a.py",
        ]
        groups = _group_by_directory(files, max_depth=2)
        assert set(groups.keys()) == {"src/pkg/", "src/other/", "tests/unit/"}
        assert groups["src/pkg/"] == ["src/pkg/main.py", "src/pkg/util.py"]
        assert groups["src/other/"] == ["src/other/app.py"]

    def test_root_files_grouped_under_root(self):
        files = ["setup.py", "README.md", "src/app.py"]
        groups = _group_by_directory(files)
        assert "(root)" in groups
        assert groups["(root)"] == ["setup.py", "README.md"]
        assert "src/" in groups

    def test_single_level_dirs(self):
        files = ["pkg/mod.py", "pkg/other.py"]
        groups = _group_by_directory(files, max_depth=2)
        assert "pkg/" in groups
        assert len(groups["pkg/"]) == 2

    def test_empty_list(self):
        assert _group_by_directory([]) == {}


# ---------------------------------------------------------------------------
# build_directory_clusters
# ---------------------------------------------------------------------------
class TestBuildDirectoryClusters:
    def test_aggregates_classes_and_functions(self, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "models.py").write_text(
            "class User:\n    def save(self): pass\n\nclass Group:\n    pass\n"
        )
        (pkg / "utils.py").write_text("def helper(): pass\ndef format_name(n): pass\n")
        files = ["pkg/models.py", "pkg/utils.py"]

        text, groups = build_directory_clusters(str(tmp_path), files)
        assert "pkg/" in groups
        assert len(groups["pkg/"]) == 2
        assert "User" in text
        assert "Group" in text
        assert "helper" in text
        assert "(2 files)" in text

    def test_root_files(self, tmp_path):
        (tmp_path / "setup.py").write_text("def setup(): pass\n")
        text, groups = build_directory_clusters(str(tmp_path), ["setup.py"])
        assert "(root)" in groups
        assert "(root)" in text
        assert "setup" in text

    def test_caps_classes_at_15(self, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        classes = "\n".join(f"class C{i}:\n    pass\n" for i in range(20))
        (pkg / "big.py").write_text(classes)
        text, _ = build_directory_clusters(str(tmp_path), ["pkg/big.py"])
        # Should have at most 15 classes listed
        class_line = [l for l in text.splitlines() if l.startswith("classes:")][0]
        assert class_line.count(";") <= 14  # 15 items = 14 separators

    def test_caps_functions_at_15(self, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        funcs = "\n".join(f"def fn{i}(): pass\n" for i in range(20))
        (pkg / "big.py").write_text(funcs)
        text, _ = build_directory_clusters(str(tmp_path), ["pkg/big.py"])
        func_line = [l for l in text.splitlines() if l.startswith("functions:")][0]
        assert func_line.count(",") <= 14

    def test_caps_imports_at_10(self, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        imports = "\n".join(f"import mod{i}" for i in range(15))
        (pkg / "big.py").write_text(imports + "\ndef dummy(): pass\n")
        text, _ = build_directory_clusters(str(tmp_path), ["pkg/big.py"])
        import_line = [l for l in text.splitlines() if l.startswith("imports:")][0]
        assert import_line.count(",") <= 9

    def test_missing_files_skipped(self, tmp_path):
        text, groups = build_directory_clusters(
            str(tmp_path), ["nonexistent/file.py"]
        )
        assert "nonexistent/" in groups
        # No structural info extracted, but group exists
        assert "(1 files)" in text

    def test_empty_file_list(self, tmp_path):
        text, groups = build_directory_clusters(str(tmp_path), [])
        assert text == ""
        assert groups == {}


# ---------------------------------------------------------------------------
# Full import graph
# ---------------------------------------------------------------------------
class TestBuildModuleFileLookup:
    def test_maps_regular_file(self):
        lookup = _build_module_file_lookup(["foo/bar.py"])
        assert lookup["foo.bar"] == "foo/bar.py"

    def test_maps_init_file(self):
        lookup = _build_module_file_lookup(["foo/__init__.py"])
        assert lookup["foo.__init__"] == "foo/__init__.py"
        assert lookup["foo"] == "foo/__init__.py"

    def test_skips_non_python(self):
        lookup = _build_module_file_lookup(["config.yaml", "data.json"])
        assert lookup == {}

    def test_multiple_files(self):
        lookup = _build_module_file_lookup([
            "pkg/mod_a.py", "pkg/mod_b.py", "pkg/__init__.py",
        ])
        assert lookup["pkg.mod_a"] == "pkg/mod_a.py"
        assert lookup["pkg.mod_b"] == "pkg/mod_b.py"
        assert lookup["pkg"] == "pkg/__init__.py"


class TestExtractFullImports:
    def test_ast_extracts_full_paths(self):
        code = "from fitz_ai.governance.governor import GovernanceDecision\nimport os\n"
        result = _extract_full_imports(code)
        assert "fitz_ai.governance.governor" in result
        assert "os" in result

    def test_ast_import_statement(self):
        code = "import fitz_ai.governance.governor\n"
        result = _extract_full_imports(code)
        assert "fitz_ai.governance.governor" in result

    def test_regex_fallback_on_syntax_error(self):
        code = "from fitz_ai.governance import governor\n>>>invalid<<<\n"
        result = _extract_full_imports(code)
        assert "fitz_ai.governance" in result

    def test_lazy_imports_inside_functions(self):
        code = (
            "class Engine:\n"
            "    def init(self):\n"
            "        from pkg.governance.decider import Decider\n"
            "        from pkg.governance import run_constraints\n"
        )
        result = _extract_full_imports(code)
        assert "pkg.governance.decider" in result
        assert "pkg.governance" in result

    def test_empty_content(self):
        assert _extract_full_imports("") == set()


class TestExtractFullImportsRegex:
    def test_from_import(self):
        result = _extract_full_imports_regex("from foo.bar.baz import Qux\n")
        assert "foo.bar.baz" in result

    def test_plain_import(self):
        result = _extract_full_imports_regex("import foo.bar\n")
        assert "foo.bar" in result

    def test_indented_imports(self):
        code = (
            "class Engine:\n"
            "    def init(self):\n"
            "        from pkg.governance import run_constraints\n"
            "            from pkg.governance.decider import Decider\n"
        )
        result = _extract_full_imports_regex(code)
        assert "pkg.governance" in result
        assert "pkg.governance.decider" in result


class TestBuildImportGraph:
    def test_resolves_intra_project(self, tmp_path):
        (tmp_path / "a.py").write_text("from b import something\n")
        (tmp_path / "b.py").write_text("something = 1\n")
        forward, lookup = build_import_graph(str(tmp_path), ["a.py", "b.py"])
        assert "b.py" in forward.get("a.py", set())

    def test_excludes_external(self, tmp_path):
        (tmp_path / "a.py").write_text("import logging\nimport os\n")
        forward, _lookup = build_import_graph(str(tmp_path), ["a.py"])
        assert forward.get("a.py") is None

    def test_empty_for_non_python(self, tmp_path):
        (tmp_path / "config.yaml").write_text("key: value\n")
        forward, _lookup = build_import_graph(str(tmp_path), ["config.yaml"])
        assert forward == {}

    def test_self_import_excluded(self, tmp_path):
        (tmp_path / "a.py").write_text("from a import x\n")
        forward, _lookup = build_import_graph(str(tmp_path), ["a.py"])
        assert forward.get("a.py") is None

    def test_subpackage_imports(self, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "engine.py").write_text("from pkg.governor import GovernanceDecision\n")
        (pkg / "governor.py").write_text("class GovernanceDecision: pass\n")
        files = ["pkg/__init__.py", "pkg/engine.py", "pkg/governor.py"]
        forward, _lookup = build_import_graph(str(tmp_path), files)
        assert "pkg/governor.py" in forward.get("pkg/engine.py", set())

    def test_missing_file_skipped(self, tmp_path):
        forward, _lookup = build_import_graph(str(tmp_path), ["nonexistent.py"])
        assert forward == {}

    def test_returns_module_lookup(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        _forward, lookup = build_import_graph(str(tmp_path), ["a.py"])
        assert lookup["a"] == "a.py"

    def test_lazy_imports_create_edges(self, tmp_path):
        """Imports inside methods (lazy imports) are still resolved."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "governor.py").write_text("class Gov: pass\n")
        (pkg / "decider.py").write_text(
            "from pkg.governor import Gov\n"
            "class Decider: pass\n"
        )
        engines = tmp_path / "engines"
        engines.mkdir()
        (engines / "engine.py").write_text(
            "class Engine:\n"
            "    def init(self):\n"
            "        from pkg.decider import Decider\n"
        )
        files = ["pkg/governor.py", "pkg/decider.py", "engines/engine.py"]
        forward, _lookup = build_import_graph(str(tmp_path), files)
        # engine.py has a lazy import of pkg.decider -> should create edge
        assert "pkg/decider.py" in forward.get("engines/engine.py", set())
        # decider.py has top-level import of pkg.governor -> should create edge
        assert "pkg/governor.py" in forward.get("pkg/decider.py", set())


# ---------------------------------------------------------------------------
# Interface signature extraction
# ---------------------------------------------------------------------------
class TestExtractSignaturesFromPython:
    def test_class_with_typed_methods(self):
        code = (
            "class ChatProvider:\n"
            "    def chat(self, prompt: str) -> str:\n"
            "        pass\n"
        )
        result = _extract_signatures_from_python(code)
        assert "class ChatProvider:" in result
        assert "chat(prompt: str) -> str" in result

    def test_class_with_bases(self):
        code = (
            "class OpenAIChat(ChatProvider):\n"
            "    def chat(self, prompt: str) -> str:\n"
            "        pass\n"
        )
        result = _extract_signatures_from_python(code)
        assert "class OpenAIChat(ChatProvider):" in result

    def test_async_method(self):
        code = (
            "class Engine:\n"
            "    async def run(self, query: str) -> dict:\n"
            "        pass\n"
        )
        result = _extract_signatures_from_python(code)
        assert "async run(query: str) -> dict" in result

    def test_no_annotations(self):
        code = (
            "class Foo:\n"
            "    def bar(self, x, y):\n"
            "        pass\n"
        )
        result = _extract_signatures_from_python(code)
        assert "bar(x, y)" in result
        assert "->" not in result

    def test_top_level_function(self):
        code = "def process(data: list[str]) -> bool:\n    pass\n"
        result = _extract_signatures_from_python(code)
        assert "process(data: list[str]) -> bool" in result

    def test_skips_self(self):
        code = (
            "class Foo:\n"
            "    def bar(self, x: int) -> None:\n"
            "        pass\n"
        )
        result = _extract_signatures_from_python(code)
        assert "self" not in result

    def test_syntax_error_returns_empty(self):
        result = _extract_signatures_from_python("def foo(:\n")
        assert result == ""

    def test_empty_class(self):
        code = "class Empty:\n    pass\n"
        result = _extract_signatures_from_python(code)
        assert "class Empty:" in result


class TestExtractInterfaceSignatures:
    def test_extracts_from_python_files(self, tmp_path):
        (tmp_path / "main.py").write_text(
            "class Engine:\n"
            "    def run(self, query: str) -> str:\n"
            "        pass\n"
        )
        result = extract_interface_signatures(str(tmp_path), ["main.py"])
        assert "## main.py" in result
        assert "run(query: str) -> str" in result

    def test_skips_non_python(self, tmp_path):
        (tmp_path / "config.yaml").write_text("key: value\n")
        result = extract_interface_signatures(str(tmp_path), ["config.yaml"])
        assert result == ""

    def test_budget_cap(self, tmp_path):
        # Create many files — should stop at budget
        for i in range(100):
            (tmp_path / f"mod{i}.py").write_text(
                f"class Class{i}:\n"
                f"    def method(self, x: int) -> str:\n"
                f"        pass\n"
            )
        files = [f"mod{i}.py" for i in range(100)]
        result = extract_interface_signatures(str(tmp_path), files)
        assert len(result) <= 8500  # ~8000 cap + some header tolerance

    def test_missing_file_skipped(self, tmp_path):
        result = extract_interface_signatures(str(tmp_path), ["nonexistent.py"])
        assert result == ""

    def test_multiple_files(self, tmp_path):
        (tmp_path / "a.py").write_text("def foo() -> int:\n    pass\n")
        (tmp_path / "b.py").write_text("def bar() -> str:\n    pass\n")
        result = extract_interface_signatures(str(tmp_path), ["a.py", "b.py"])
        assert "## a.py" in result
        assert "## b.py" in result
        assert "foo() -> int" in result
        assert "bar() -> str" in result


class TestExtractLibrarySignatures:
    """Tests for extract_library_signatures()."""

    def test_extracts_stdlib_package(self, tmp_path):
        """Should extract signatures from an importable stdlib package."""
        (tmp_path / "app.py").write_text("import json\nx = json.dumps({})\n")
        result = extract_library_signatures(
            str(tmp_path), ["app.py"], ["app.py"],
        )
        assert "## json" in result
        assert "dumps" in result

    def test_skips_intra_project_imports(self, tmp_path):
        """Imports that resolve to project files should be excluded."""
        (tmp_path / "mymod.py").write_text("def foo(): pass\n")
        (tmp_path / "app.py").write_text("import mymod\nmymod.foo()\n")
        result = extract_library_signatures(
            str(tmp_path), ["app.py"], ["app.py", "mymod.py"],
        )
        assert "mymod" not in result

    def test_skips_uninstalled_packages(self, tmp_path):
        """Uninstalled packages should be silently skipped."""
        (tmp_path / "app.py").write_text("import nonexistent_pkg_xyz\n")
        result = extract_library_signatures(
            str(tmp_path), ["app.py"], ["app.py"],
        )
        assert result == ""

    def test_empty_included_files(self, tmp_path):
        result = extract_library_signatures(str(tmp_path), [], [])
        assert result == ""

    def test_skips_typing(self, tmp_path):
        """typing is excluded since it's not useful as API reference."""
        (tmp_path / "app.py").write_text("from typing import Any, Dict\n")
        result = extract_library_signatures(
            str(tmp_path), ["app.py"], ["app.py"],
        )
        assert "typing" not in result

    def test_caps_output_length(self, tmp_path):
        """Output should be capped to avoid context bloat."""
        # Import many packages
        imports = "\n".join(f"import {pkg}" for pkg in [
            "json", "os", "sys", "re", "ast", "pathlib",
            "logging", "inspect", "importlib", "hashlib",
            "sqlite3", "email",
        ])
        (tmp_path / "app.py").write_text(imports + "\n")
        result = extract_library_signatures(
            str(tmp_path), ["app.py"], ["app.py"],
        )
        assert len(result) <= 5000  # ~4000 cap + header tolerance

    def test_non_python_files_skipped(self, tmp_path):
        (tmp_path / "config.yaml").write_text("key: value\n")
        result = extract_library_signatures(
            str(tmp_path), ["config.yaml"], ["config.yaml"],
        )
        assert result == ""


class TestParseClassHierarchies:
    """Tests for _parse_class_hierarchies()."""

    def test_single_hierarchy(self):
        sigs = "class OpenAIChat(ChatProvider):\nclass OllamaChat(ChatProvider):"
        result = _parse_class_hierarchies(sigs)
        assert result == {"ChatProvider": ["OpenAIChat", "OllamaChat"]}

    def test_no_hierarchies(self):
        sigs = "def foo() -> str:\ndef bar() -> int:"
        result = _parse_class_hierarchies(sigs)
        assert result == {}

    def test_skips_object_abc_basemodel(self):
        sigs = "class Foo(object):\nclass Bar(ABC):\nclass Baz(BaseModel):\nclass Qux(Protocol):"
        result = _parse_class_hierarchies(sigs)
        assert result == {}

    def test_multiple_bases(self):
        sigs = "class Impl(Base, Mixin):"
        result = _parse_class_hierarchies(sigs)
        assert "Base" in result
        assert "Mixin" in result

    def test_multiple_base_classes(self):
        sigs = (
            "class A(Base):\nclass B(Base):\n"
            "class X(Other):\nclass Y(Other):\nclass Z(Other):"
        )
        result = _parse_class_hierarchies(sigs)
        assert len(result["Base"]) == 2
        assert len(result["Other"]) == 3


class TestParseSimpleReturnMethods:
    """Tests for _parse_simple_return_methods()."""

    def test_finds_simple_return(self):
        sigs = "## src/provider.py\n  chat(prompt: str, model: str, config: dict) -> str"
        result = _parse_simple_return_methods(sigs)
        assert len(result) == 1
        assert result[0] == ("src/provider.py", "chat", "str", 3)

    def test_skips_complex_return(self):
        sigs = "## mod.py\n  process(a: int, b: int, c: int) -> dict"
        result = _parse_simple_return_methods(sigs)
        assert result == []

    def test_skips_private_methods(self):
        sigs = "## mod.py\n  _internal(a: int, b: int, c: int) -> str"
        result = _parse_simple_return_methods(sigs)
        assert result == []

    def test_skips_few_params(self):
        sigs = "## mod.py\n  get(key: str) -> str"
        result = _parse_simple_return_methods(sigs)
        assert result == []

    def test_bool_return(self):
        sigs = "## check.py\n  validate(data: dict, schema: dict, strict: bool) -> bool"
        result = _parse_simple_return_methods(sigs)
        assert len(result) == 1
        assert result[0][2] == "bool"

    def test_async_method(self):
        sigs = "## svc.py\n  async generate(prompt: str, model: str, temp: float) -> str"
        result = _parse_simple_return_methods(sigs)
        assert len(result) == 1
        assert result[0][1] == "generate"

    def test_tracks_current_file(self):
        sigs = (
            "## a.py\n  foo(x: int, y: int, z: int) -> str\n"
            "## b.py\n  bar(a: str, b: str, c: str) -> bool"
        )
        result = _parse_simple_return_methods(sigs)
        assert result[0][0] == "a.py"
        assert result[1][0] == "b.py"


class TestGenerateInvestigationQuestions:
    """Tests for generate_investigation_questions()."""

    def test_class_hierarchy_question(self):
        sigs = "class ImplA(Base):\nclass ImplB(Base):"
        result = generate_investigation_questions(sigs, {}, {})
        assert len(result) == 1
        assert "Base" in result[0]
        assert "ImplA" in result[0]

    def test_hub_file_question(self):
        result = generate_investigation_questions(
            "", {}, {"core/engine.py": 7},
        )
        assert len(result) == 1
        assert "core/engine.py" in result[0]
        assert "7" in result[0]

    def test_simple_return_question(self):
        sigs = "## provider.py\n  chat(prompt: str, model: str, config: dict) -> str"
        result = generate_investigation_questions(sigs, {}, {})
        assert len(result) == 1
        assert "chat" in result[0]
        assert "str" in result[0]

    def test_max_three_questions(self):
        # 2 hierarchy + 1 hub + 1 simple return = should cap at 3
        sigs = (
            "class A(X):\nclass B(X):\n"
            "class C(Y):\nclass D(Y):\n"
            "## mod.py\n  do(a: int, b: int, c: int) -> str"
        )
        result = generate_investigation_questions(
            sigs, {}, {"hub.py": 10},
        )
        assert len(result) <= 3

    def test_empty_inputs(self):
        result = generate_investigation_questions("", {}, {})
        assert result == []

    def test_hub_below_threshold_ignored(self):
        result = generate_investigation_questions(
            "", {}, {"small.py": 3},
        )
        assert result == []

    def test_combined_all_types(self):
        sigs = (
            "class ImplA(Base):\nclass ImplB(Base):\n"
            "## svc.py\n  process(a: int, b: int, c: int) -> bool"
        )
        result = generate_investigation_questions(
            sigs, {}, {"hub.py": 6},
        )
        # All 3 heuristics fire = exactly 3 (at cap)
        assert len(result) == 3
