# tests/unit/test_agent_tools.py
"""Unit tests for AgentContextGatherer filesystem tools."""

import pytest

from fitz_graveyard.planning.agent.tools import _make_tools


@pytest.fixture
def tools_and_root(tmp_path):
    """Create tool set bound to a temp directory with sample files."""
    # Create sample structure
    (tmp_path / "README.md").write_text("# Test Project\nA test project.")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "# src/main.py\nclass App:\n    def run(self):\n        pass\n"
    )
    (tmp_path / "src" / "config.py").write_text("# src/config.py\nDEBUG = True\n")
    (tmp_path / ".hidden").write_text("hidden file")

    tools = _make_tools(str(tmp_path))
    tool_map = {fn.__name__: fn for fn in tools}
    return tool_map, tmp_path


class TestListDirectory:
    def test_lists_root(self, tools_and_root):
        tool_map, tmp_path = tools_and_root
        result = tool_map["list_directory"](".")
        assert "README.md" in result
        assert "src/" in result

    def test_lists_subdir(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["list_directory"]("src")
        assert "main.py" in result
        assert "config.py" in result

    def test_traversal_blocked(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["list_directory"]("../..")
        assert result.startswith("Error:")

    def test_nonexistent_returns_error(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["list_directory"]("nonexistent_dir")
        assert result.startswith("Error:")

    def test_file_path_returns_error(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["list_directory"]("README.md")
        assert result.startswith("Error:") or "not a directory" in result


class TestReadFile:
    def test_reads_full_file(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["read_file"]("README.md")
        assert "Test Project" in result

    def test_reads_line_range(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["read_file"]("src/main.py", start_line=2, end_line=3)
        assert "class App" in result
        assert "# src/main.py" not in result  # line 1 excluded

    def test_traversal_blocked(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["read_file"]("../../etc/passwd")
        assert result.startswith("Error:")

    def test_nonexistent_returns_error(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["read_file"]("no_such_file.py")
        assert result.startswith("Error:")

    def test_truncates_large_file(self, tmp_path):
        large_file = tmp_path / "large.txt"
        large_file.write_bytes(b"x" * 100_000)
        tools = _make_tools(str(tmp_path), max_file_bytes=1000)
        tool_map = {fn.__name__: fn for fn in tools}
        result = tool_map["read_file"]("large.txt")
        assert "truncated" in result.lower()

    def test_directory_returns_error(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["read_file"]("src")
        assert result.startswith("Error:")


class TestSearchText:
    def test_finds_matches(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["search_text"]("class App")
        assert "main.py" in result
        assert "class App" in result

    def test_case_insensitive_by_default(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["search_text"]("CLASS APP")
        assert "main.py" in result

    def test_no_matches(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["search_text"]("ZZZNOMATCHZZZ")
        assert "(no matches found)" in result

    def test_invalid_regex(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["search_text"]("[invalid(")
        assert result.startswith("Error:") and "regex" in result

    def test_traversal_blocked(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["search_text"]("pattern", path="../..")
        assert result.startswith("Error:")

    def test_subdir_search(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["search_text"]("DEBUG", path="src")
        assert "config.py" in result


class TestFindFiles:
    def test_finds_python_files(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["find_files"]("**/*.py")
        assert "main.py" in result
        assert "config.py" in result

    def test_finds_markdown(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["find_files"]("*.md")
        assert "README.md" in result

    def test_no_matches(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["find_files"]("*.xyz_no_match")
        assert "(no files matched)" in result

    def test_traversal_blocked(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["find_files"]("*.py", directory="../../..")
        assert result.startswith("Error:")

    def test_nonexistent_dir(self, tools_and_root):
        tool_map, _ = tools_and_root
        result = tool_map["find_files"]("*.py", directory="no_such_dir")
        assert result.startswith("Error:")
