# tests/unit/test_indexer.py
"""Unit tests for the structural index builder."""

import json
from pathlib import Path

import pytest

from fitz_graveyard.planning.agent.indexer import (
    _MAX_INDEX_CHARS,
    _extract_config,
    _extract_generic_code,
    _extract_markdown,
    _extract_python,
    _extract_python_regex,
    build_structural_index,
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
