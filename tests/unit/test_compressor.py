# tests/unit/test_compressor.py
"""Tests for AST-based code compression."""

import textwrap

import pytest

from fitz_graveyard.planning.agent.compressor import (
    compress_file,
    compress_python,
    _strip_comments_and_blanks,
)


class TestStripCommentsAndBlanks:

    def test_removes_comment_lines(self):
        src = "x = 1\n# this is a comment\ny = 2\n"
        result = _strip_comments_and_blanks(src)
        assert "# this is a comment" not in result
        assert "x = 1" in result
        assert "y = 2" in result

    def test_keeps_shebangs(self):
        src = "#!/usr/bin/env python\nx = 1\n"
        result = _strip_comments_and_blanks(src)
        assert "#!/usr/bin/env python" in result

    def test_keeps_type_ignore(self):
        src = "x = foo()  # type: ignore\n"
        result = _strip_comments_and_blanks(src)
        assert "# type: ignore" in result

    def test_collapses_blank_lines(self):
        src = "x = 1\n\n\n\n\ny = 2\n"
        result = _strip_comments_and_blanks(src)
        assert result.count("\n\n") <= 1


class TestCompressPython:

    def test_strips_module_docstring(self):
        src = '"""Module docstring."""\n\nimport os\n'
        result = compress_python(src)
        assert '"""Module docstring."""' not in result
        assert "import os" in result

    def test_strips_function_docstring(self):
        src = textwrap.dedent('''\
            def foo(x: int) -> int:
                """Return x squared."""
                return x * x
        ''')
        result = compress_python(src)
        assert '"""Return x squared."""' not in result
        assert "def foo(x: int) -> int:" in result

    def test_strips_class_docstring(self):
        src = textwrap.dedent('''\
            class Foo:
                """A foo class."""
                x: int = 0
        ''')
        result = compress_python(src)
        assert '"""A foo class."""' not in result
        assert "class Foo:" in result
        assert "x: int = 0" in result

    def test_keeps_short_function_body(self):
        src = textwrap.dedent('''\
            def add(a, b):
                return a + b
        ''')
        result = compress_python(src)
        assert "return a + b" in result

    def test_collapses_long_function_body(self):
        body_lines = "\n".join(f"    x{i} = {i}" for i in range(20))
        src = f"def long_func(a, b):\n{body_lines}\n    return a\n"
        result = compress_python(src)
        assert "def long_func(a, b):" in result
        assert "..." in result
        assert "x10 = 10" not in result

    def test_keeps_imports(self):
        src = "import os\nimport sys\nfrom pathlib import Path\n"
        result = compress_python(src)
        assert "import os" in result
        assert "import sys" in result
        assert "from pathlib import Path" in result

    def test_keeps_class_attributes(self):
        src = textwrap.dedent('''\
            class Config:
                MAX_SIZE: int = 100
                DEFAULT_NAME: str = "test"
        ''')
        result = compress_python(src)
        assert "MAX_SIZE: int = 100" in result
        assert 'DEFAULT_NAME: str = "test"' in result

    def test_keeps_decorators(self):
        src = textwrap.dedent('''\
            @property
            def name(self) -> str:
                return self._name
        ''')
        result = compress_python(src)
        assert "@property" in result
        assert "def name(self) -> str:" in result

    def test_handles_syntax_errors(self):
        src = "def broken(\n"
        result = compress_python(src)
        assert result == src  # Returns unchanged

    def test_handles_empty_source(self):
        result = compress_python("")
        assert result == ""

    def test_keeps_dataclass_fields(self):
        src = textwrap.dedent('''\
            from dataclasses import dataclass

            @dataclass
            class Point:
                x: float
                y: float
                z: float = 0.0
        ''')
        result = compress_python(src)
        assert "x: float" in result
        assert "y: float" in result
        assert "z: float = 0.0" in result

    def test_keeps_constants(self):
        src = textwrap.dedent('''\
            MAX_RETRIES = 3
            DEFAULT_TIMEOUT = 30
            _INTERNAL = "secret"
        ''')
        result = compress_python(src)
        assert "MAX_RETRIES = 3" in result
        assert "DEFAULT_TIMEOUT = 30" in result


class TestCompressFile:

    def test_python_files_get_ast_compression(self):
        src = textwrap.dedent('''\
            """Module doc."""

            def foo():
                """Func doc."""
                return 1
        ''')
        result = compress_file(src, "mymodule.py")
        assert '"""Module doc."""' not in result
        assert "def foo():" in result

    def test_non_python_gets_comment_stripping(self):
        src = "key: value\n# comment\nother: stuff\n"
        result = compress_file(src, "config.yaml")
        assert "# comment" not in result
        assert "key: value" in result

    def test_test_files_collapse_all_bodies(self):
        src = textwrap.dedent('''\
            def test_simple():
                x = 1
                assert x == 1

            def test_another():
                y = 2
                assert y == 2
        ''')
        result = compress_file(src, "tests/unit/test_foo.py")
        assert "def test_simple():" in result
        assert "def test_another():" in result
        assert "x = 1" not in result
        assert "y = 2" not in result

    def test_test_detection_by_path(self):
        src = "def test_x():\n    assert True\n"
        # Various test path patterns
        for path in ["tests/test_foo.py", "test_bar.py", "tests/unit/test_baz.py"]:
            result = compress_file(src, path)
            assert "..." in result or "assert" not in result

    def test_significant_reduction(self):
        # A realistic file with docstrings, comments, long methods
        src = textwrap.dedent('''\
            """Big module with lots of stuff."""

            import os
            import sys

            # Configuration constants
            MAX_SIZE = 100
            TIMEOUT = 30

            class MyService:
                """Service that does things."""

                def __init__(self, name: str, config: dict):
                    """Initialize service."""
                    self.name = name
                    self.config = config
                    self._cache = {}
                    self._running = False

                def process(self, data: list[str]) -> dict:
                    """Process data and return results."""
                    results = {}
                    for item in data:
                        key = item.strip()
                        value = self._transform(key)
                        if value is not None:
                            results[key] = value
                        else:
                            results[key] = self._default(key)
                    self._cache.update(results)
                    return results

                def _transform(self, key: str) -> str | None:
                    """Internal transform."""
                    if key in self.config:
                        return self.config[key]
                    return None

                def _default(self, key: str) -> str:
                    """Default value."""
                    return f"default_{key}"
        ''')
        result = compress_file(src, "myservice.py")
        reduction = 1 - len(result) / len(src)
        assert reduction > 0.4, f"Expected >40% reduction, got {reduction:.0%}"
        # Key structural elements preserved
        assert "class MyService:" in result
        assert "def __init__(self, name: str, config: dict):" in result
        assert "def process(self, data: list[str]) -> dict:" in result
        assert "MAX_SIZE = 100" in result
        assert "import os" in result
