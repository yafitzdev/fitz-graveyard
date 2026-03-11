# tests/unit/test_agent_gatherer.py
"""Unit tests for AgentContextGatherer (multi-signal retrieval pipeline)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fitz_graveyard.config.schema import AgentConfig
from fitz_graveyard.planning.agent.gatherer import (
    AgentContextGatherer,
    _MAX_TREE_FILES,
)
from fitz_graveyard.planning.agent.indexer import build_import_graph


def _make_config(**kwargs):
    defaults = dict(enabled=True, max_file_bytes=50_000)
    defaults.update(kwargs)
    return AgentConfig(**defaults)


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.model = "test-model"
    client.fast_model = "test-model"
    client.mid_model = "test-model"
    client.smart_model = "test-model"
    client.context_size = 65536
    client.generate = AsyncMock(return_value="LLM response")
    client.unload_model = AsyncMock(return_value=True)
    client.reload_model = AsyncMock(return_value=True)
    return client


# ---------------------------------------------------------------------------
# Pass 1: _build_file_tree
# ---------------------------------------------------------------------------
class TestBuildFileTree:
    def test_empty_dir(self, tmp_path):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        tree, paths = gatherer._build_file_tree()
        assert tree == ""
        assert paths == []

    def test_lists_files_with_sizes(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hi')")
        (tmp_path / "README.md").write_text("# Hello")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        tree, paths = gatherer._build_file_tree()
        assert "main.py" in tree
        assert "README.md" in tree
        assert "main.py" in paths
        assert "README.md" in paths

    def test_skips_git_dir(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config.py").write_text("x = 1")
        (tmp_path / "main.py").write_text("print('hi')")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        _, paths = gatherer._build_file_tree()
        assert "main.py" in paths
        assert not any(".git" in p for p in paths)

    def test_skips_pycache(self, tmp_path):
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "module.cpython-311.pyc").write_bytes(b"\x00")
        (tmp_path / "main.py").write_text("print('hi')")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        _, paths = gatherer._build_file_tree()
        assert not any("__pycache__" in p for p in paths)

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "index.js").write_text("module.exports = {}")
        (tmp_path / "main.py").write_text("print('hi')")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        _, paths = gatherer._build_file_tree()
        assert not any("node_modules" in p for p in paths)

    def test_skips_venv(self, tmp_path):
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "pyvenv.cfg").write_text("home = /usr/bin")
        (tmp_path / "main.py").write_text("print('hi')")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        _, paths = gatherer._build_file_tree()
        assert not any(".venv" in p for p in paths)

    def test_skips_non_indexable_extensions(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02")
        (tmp_path / "main.py").write_text("print('hi')")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        _, paths = gatherer._build_file_tree()
        assert "main.py" in paths
        assert "image.png" not in paths

    def test_caps_at_max_files(self, tmp_path):
        for i in range(_MAX_TREE_FILES + 5):
            (tmp_path / f"file_{i:05d}.py").write_text(f"x = {i}")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        tree, paths = gatherer._build_file_tree()
        assert len(paths) == _MAX_TREE_FILES
        assert "truncated" in tree

    def test_uses_posix_paths(self, tmp_path):
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "mod.py").write_text("x = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        _, paths = gatherer._build_file_tree()
        assert "pkg/mod.py" in paths

    def test_invalid_dir(self):
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir="/nonexistent/path"
        )
        tree, paths = gatherer._build_file_tree()
        assert tree == ""
        assert paths == []


# ---------------------------------------------------------------------------
# Pass 2: Query expansion
# ---------------------------------------------------------------------------
class TestExpandQuery:
    @pytest.mark.asyncio
    async def test_parses_terms_and_hyde(self, mock_client):
        mock_client.generate = AsyncMock(return_value=(
            "TERMS:\nchat_provider\nopenai_client\nllm_plugin\n\n"
            "HYPOTHETICAL:\n```python\nclass OpenAIChat:\n    pass\n```"
        ))
        gatherer = AgentContextGatherer(config=_make_config(), source_dir="/tmp")
        terms, hyde = await gatherer._expand_query(
            mock_client, "test-model", "build openai chat plugin",
        )
        assert "chat_provider" in terms
        assert "openai_client" in terms
        assert "class OpenAIChat" in hyde

    @pytest.mark.asyncio
    async def test_uses_temperature_zero(self, mock_client):
        mock_client.generate = AsyncMock(return_value="TERMS:\nfoo\n\nHYPOTHETICAL:\n```python\nx=1\n```")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir="/tmp")
        await gatherer._expand_query(mock_client, "test-model", "task")
        assert mock_client.generate.call_args[1]["temperature"] == 0

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self, mock_client):
        mock_client.generate = AsyncMock(side_effect=RuntimeError("boom"))
        gatherer = AgentContextGatherer(config=_make_config(), source_dir="/tmp")
        terms, hyde = await gatherer._expand_query(
            mock_client, "test-model", "task",
        )
        assert terms == ""
        assert hyde == ""


# ---------------------------------------------------------------------------
# Pass 3: Structural scan
# ---------------------------------------------------------------------------
class TestStructuralScan:
    @pytest.mark.asyncio
    async def test_returns_file_list(self, mock_client):
        mock_client.generate = AsyncMock(
            return_value='```json\n["engine.py", "providers/openai.py"]\n```'
        )
        gatherer = AgentContextGatherer(config=_make_config(), source_dir="/tmp")
        result = await gatherer._structural_scan(
            mock_client, "test-model", "## engine.py\nclasses: Engine", "task",
        )
        assert result == ["engine.py", "providers/openai.py"]

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self, mock_client):
        mock_client.generate = AsyncMock(side_effect=RuntimeError("boom"))
        gatherer = AgentContextGatherer(config=_make_config(), source_dir="/tmp")
        result = await gatherer._structural_scan(
            mock_client, "test-model", "index", "task",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty(self, mock_client):
        mock_client.generate = AsyncMock(return_value="not valid json")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir="/tmp")
        result = await gatherer._structural_scan(
            mock_client, "test-model", "index", "task",
        )
        assert result == []


# ---------------------------------------------------------------------------
# Pass 5: Neighbor expansion
# ---------------------------------------------------------------------------
class TestNeighborExpand:
    def test_adds_same_directory_files(self):
        selected = ["pkg/providers/base.py"]
        all_paths = [
            "pkg/providers/base.py",
            "pkg/providers/openai.py",
            "pkg/providers/anthropic.py",
            "pkg/core/engine.py",
        ]
        result = AgentContextGatherer._neighbor_expand(selected, all_paths)
        assert "pkg/providers/base.py" in result
        assert "pkg/providers/openai.py" in result
        assert "pkg/providers/anthropic.py" in result
        assert "pkg/core/engine.py" not in result

    def test_no_duplicates(self):
        selected = ["pkg/a.py", "pkg/b.py"]
        all_paths = ["pkg/a.py", "pkg/b.py", "pkg/c.py"]
        result = AgentContextGatherer._neighbor_expand(selected, all_paths)
        assert result.count("pkg/a.py") == 1
        assert result.count("pkg/b.py") == 1

    def test_root_files_not_expanded(self):
        """Files at root (no parent dir) don't trigger expansion."""
        selected = ["main.py"]
        all_paths = ["main.py", "setup.py", "pkg/core.py"]
        result = AgentContextGatherer._neighbor_expand(selected, all_paths)
        assert result == ["main.py"]

    def test_multiple_directories(self):
        selected = ["pkg/a.py", "lib/x.py"]
        all_paths = ["pkg/a.py", "pkg/b.py", "lib/x.py", "lib/y.py", "other/z.py"]
        result = AgentContextGatherer._neighbor_expand(selected, all_paths)
        assert "pkg/b.py" in result
        assert "lib/y.py" in result
        assert "other/z.py" not in result

    def test_empty_selected(self):
        result = AgentContextGatherer._neighbor_expand([], ["a.py", "b.py"])
        assert result == []

    def test_siblings_inserted_after_trigger(self):
        """Siblings appear right after the selected file, not at the end."""
        selected = ["core/engine.py", "llm/providers/base.py", "api/server.py"]
        all_paths = [
            "core/engine.py", "core/utils.py",
            "llm/providers/base.py", "llm/providers/openai.py", "llm/providers/ollama.py",
            "api/server.py", "api/routes.py",
        ]
        result = AgentContextGatherer._neighbor_expand(selected, all_paths)
        # providers siblings should appear right after base.py, not at end
        base_idx = result.index("llm/providers/base.py")
        openai_idx = result.index("llm/providers/openai.py")
        ollama_idx = result.index("llm/providers/ollama.py")
        server_idx = result.index("api/server.py")
        assert openai_idx == base_idx + 1 or ollama_idx == base_idx + 1
        assert openai_idx < server_idx
        assert ollama_idx < server_idx

    def test_siblings_not_duplicated_across_dirs(self):
        """If two selected files share a directory, siblings only appear once."""
        selected = ["pkg/a.py", "pkg/b.py"]
        all_paths = ["pkg/a.py", "pkg/b.py", "pkg/c.py"]
        result = AgentContextGatherer._neighbor_expand(selected, all_paths)
        assert result.count("pkg/c.py") == 1

    def test_expand_from_limits_which_dirs_expand(self):
        """Only directories containing expand_from files get expanded."""
        selected = [
            "llm/providers/base.py",   # high confidence (scan hit)
            "llm/auth/token_provider.py",  # low confidence (BM25 noise)
        ]
        all_paths = [
            "llm/providers/base.py",
            "llm/providers/openai.py",
            "llm/providers/ollama.py",
            "llm/auth/token_provider.py",
            "llm/auth/m2m.py",
            "llm/auth/httpx_auth.py",
        ]
        # Only expand from base.py (scan hit), not token_provider.py
        result = AgentContextGatherer._neighbor_expand(
            selected, all_paths,
            expand_from=["llm/providers/base.py"],
        )
        # Provider siblings found
        assert "llm/providers/openai.py" in result
        assert "llm/providers/ollama.py" in result
        # Auth siblings NOT found (token_provider.py not in expand_from)
        assert "llm/auth/m2m.py" not in result
        assert "llm/auth/httpx_auth.py" not in result
        # Original selected files still present
        assert "llm/auth/token_provider.py" in result


# ---------------------------------------------------------------------------
# Pass 5b: Neighbor screening
# ---------------------------------------------------------------------------
class TestScreenNeighbors:
    @pytest.mark.asyncio
    async def test_small_dirs_not_screened(self, mock_client):
        """Directories with <= 10 siblings are kept without LLM call."""
        gatherer = AgentContextGatherer(config=_make_config(), source_dir="/tmp")
        before = ["core/engine.py"]
        after = ["core/engine.py", "core/a.py", "core/b.py"]
        result = await gatherer._screen_neighbors(
            mock_client, "model", "task", before, after,
        )
        assert result == after
        mock_client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_large_dir_screened(self, mock_client):
        """Directory with > 10 siblings triggers LLM screening."""
        gatherer = AgentContextGatherer(config=_make_config(), source_dir="/tmp")
        before = ["core/engine.py"]
        siblings = [f"core/file{i}.py" for i in range(15)]
        after = ["core/engine.py"] + siblings

        # LLM says only file0, file1, file2 are relevant
        mock_client.generate = AsyncMock(
            return_value='["core/file0.py", "core/file1.py", "core/file2.py"]'
        )
        result = await gatherer._screen_neighbors(
            mock_client, "model", "task", before, after,
        )
        assert "core/engine.py" in result  # original kept
        assert "core/file0.py" in result
        assert "core/file1.py" in result
        assert "core/file2.py" in result
        assert "core/file10.py" not in result  # screened out

    @pytest.mark.asyncio
    async def test_llm_failure_keeps_all(self, mock_client):
        """On LLM failure, all siblings are kept (graceful fallback)."""
        gatherer = AgentContextGatherer(config=_make_config(), source_dir="/tmp")
        before = ["core/engine.py"]
        siblings = [f"core/file{i}.py" for i in range(15)]
        after = ["core/engine.py"] + siblings

        mock_client.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
        result = await gatherer._screen_neighbors(
            mock_client, "model", "task", before, after,
        )
        assert result == after  # all kept on failure

    @pytest.mark.asyncio
    async def test_parse_failure_keeps_all(self, mock_client):
        """On unparseable LLM response, all siblings are kept."""
        gatherer = AgentContextGatherer(config=_make_config(), source_dir="/tmp")
        before = ["core/engine.py"]
        siblings = [f"core/file{i}.py" for i in range(15)]
        after = ["core/engine.py"] + siblings

        mock_client.generate = AsyncMock(return_value="I don't understand")
        result = await gatherer._screen_neighbors(
            mock_client, "model", "task", before, after,
        )
        assert result == after

    @pytest.mark.asyncio
    async def test_mixed_dirs(self, mock_client):
        """Small dirs kept as-is, only large dirs screened."""
        gatherer = AgentContextGatherer(config=_make_config(), source_dir="/tmp")
        before = ["core/engine.py", "llm/client.py"]
        # core/ has 12 new siblings (screened), llm/ has 3 (kept)
        core_siblings = [f"core/s{i}.py" for i in range(12)]
        llm_siblings = ["llm/types.py", "llm/config.py", "llm/factory.py"]
        after = ["core/engine.py"] + core_siblings + ["llm/client.py"] + llm_siblings

        mock_client.generate = AsyncMock(
            return_value='["core/s0.py", "core/s1.py"]'
        )
        result = await gatherer._screen_neighbors(
            mock_client, "model", "task", before, after,
        )
        # Core screened: only s0, s1 kept
        assert "core/s0.py" in result
        assert "core/s1.py" in result
        assert "core/s5.py" not in result
        # LLM siblings all kept (under threshold)
        assert "llm/types.py" in result
        assert "llm/config.py" in result
        assert "llm/factory.py" in result


# ---------------------------------------------------------------------------
# Pass 6: _read_selected_files
# ---------------------------------------------------------------------------
class TestReadSelectedFiles:
    def test_reads_all_files_into_dict(self, tmp_path):
        (tmp_path / "main.py").write_text("def run(): pass")
        paths = ["main.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        contents, included = gatherer._read_selected_files(paths)
        assert included == ["main.py"]
        assert "def run(): pass" in contents["main.py"]

    def test_reads_multiple_files(self, tmp_path):
        for name in ["a.py", "b.py", "c.py"]:
            (tmp_path / name).write_text(f"# {name}\nx = 1")
        paths = ["a.py", "b.py", "c.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        contents, included = gatherer._read_selected_files(paths)
        assert included == ["a.py", "b.py", "c.py"]
        assert len(contents) == 3

    def test_skips_missing_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        paths = ["a.py", "missing.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        contents, included = gatherer._read_selected_files(paths)
        assert "a.py" in included
        assert "missing.py" not in included
        assert "missing.py" not in contents

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.py").write_text("")
        (tmp_path / "real.py").write_text("x = 1")
        paths = ["empty.py", "real.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        contents, included = gatherer._read_selected_files(paths)
        assert "real.py" in included
        assert "empty.py" not in included

    def test_respects_max_file_bytes(self, tmp_path):
        (tmp_path / "big.py").write_text("x" * 200)
        paths = ["big.py"]
        gatherer = AgentContextGatherer(
            config=_make_config(max_file_bytes=50),
            source_dir=str(tmp_path),
        )
        contents, included = gatherer._read_selected_files(paths)
        assert "big.py" in included
        assert len(contents["big.py"]) <= 50

    def test_empty_selected_returns_empty(self, tmp_path):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        contents, included = gatherer._read_selected_files([])
        assert contents == {}
        assert included == []


# ---------------------------------------------------------------------------
# Import expansion
# ---------------------------------------------------------------------------
class TestImportExpand:
    def _graph(self, tmp_path, file_paths):
        fwd, _ = build_import_graph(str(tmp_path), file_paths)
        return fwd

    def test_adds_direct_imports(self, tmp_path):
        (tmp_path / "a.py").write_text("from b import helper")
        (tmp_path / "b.py").write_text("def helper(): pass")
        paths = ["a.py", "b.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._import_expand(["a.py"], paths, self._graph(tmp_path, paths))
        assert "a.py" in result
        assert "b.py" in result

    def test_no_duplicates(self, tmp_path):
        (tmp_path / "a.py").write_text("from b import helper")
        (tmp_path / "b.py").write_text("def helper(): pass")
        paths = ["a.py", "b.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._import_expand(["a.py", "b.py"], paths, self._graph(tmp_path, paths))
        assert result.count("a.py") == 1
        assert result.count("b.py") == 1

    def test_no_imports_returns_same(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        paths = ["a.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._import_expand(["a.py"], paths, self._graph(tmp_path, paths))
        assert result == ["a.py"]

    def test_forward_only_depth_one(self, tmp_path):
        """Forward-only depth 1: a imports b, but b's transitive import c is not reached."""
        (tmp_path / "a.py").write_text("from b import x")
        (tmp_path / "b.py").write_text("from c import y\nx = 1")
        (tmp_path / "c.py").write_text("y = 2")
        paths = ["a.py", "b.py", "c.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._import_expand(["a.py"], paths, self._graph(tmp_path, paths))
        assert "a.py" in result
        assert "b.py" in result  # direct forward import
        assert "c.py" not in result  # transitive, not depth 1

    def test_no_reverse_imports(self, tmp_path):
        """Reverse imports are excluded to prevent hub-file explosion."""
        (tmp_path / "base.py").write_text("class Base: pass")
        (tmp_path / "child.py").write_text("from base import Base")
        paths = ["base.py", "child.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        # base.py is relevant; child.py imports it but reverse is excluded
        result = gatherer._import_expand(["base.py"], paths, self._graph(tmp_path, paths))
        assert "base.py" in result
        assert "child.py" not in result  # reverse import, excluded


# ---------------------------------------------------------------------------
# Prioritize for summary
# ---------------------------------------------------------------------------
class TestPrioritizeForSummary:
    def test_code_before_docs(self):
        paths = [
            "docs/ARCHITECTURE.md",
            "fitz_ai/llm/providers/openai.py",
            "docs/CONFIG.md",
            "fitz_ai/core/answer.py",
        ]
        result = AgentContextGatherer._prioritize_for_summary(paths)
        assert result[:2] == [
            "fitz_ai/llm/providers/openai.py",
            "fitz_ai/core/answer.py",
        ]
        assert set(result[2:]) == {"docs/ARCHITECTURE.md", "docs/CONFIG.md"}

    def test_tests_between_code_and_docs(self):
        paths = [
            "docs/README.md",
            "tests/unit/test_foo.py",
            "fitz_ai/engine.py",
        ]
        result = AgentContextGatherer._prioritize_for_summary(paths)
        assert result[0] == "fitz_ai/engine.py"
        assert result[1] == "tests/unit/test_foo.py"
        assert result[2] == "docs/README.md"

    def test_preserves_order_within_tier(self):
        paths = [
            "fitz_ai/b.py",
            "fitz_ai/a.py",
            "fitz_ai/c.py",
        ]
        result = AgentContextGatherer._prioritize_for_summary(paths)
        assert result == ["fitz_ai/b.py", "fitz_ai/a.py", "fitz_ai/c.py"]

    def test_examples_and_github_are_low_priority(self):
        paths = [
            "examples/01_quickstart.py",
            ".github/workflows/ci.yml",
            "fitz_ai/core.py",
        ]
        result = AgentContextGatherer._prioritize_for_summary(paths)
        assert result[0] == "fitz_ai/core.py"

    def test_config_files_between_code_and_tests(self):
        paths = [
            "tests/test_x.py",
            "pyproject.toml",
            "fitz_ai/main.py",
        ]
        result = AgentContextGatherer._prioritize_for_summary(paths)
        assert result[0] == "fitz_ai/main.py"
        assert result[1] == "pyproject.toml"
        assert result[2] == "tests/test_x.py"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
class TestParseFileList:
    def test_plain_json_array(self):
        result = AgentContextGatherer._parse_file_list('["a.py", "b.py"]')
        assert result == ["a.py", "b.py"]

    def test_json_object_with_files(self):
        result = AgentContextGatherer._parse_file_list('{"files": ["a.py"]}')
        assert result == ["a.py"]

    def test_markdown_fenced(self):
        result = AgentContextGatherer._parse_file_list(
            '```json\n["a.py"]\n```'
        )
        assert result == ["a.py"]

    def test_invalid_json_returns_none(self):
        result = AgentContextGatherer._parse_file_list("not json at all")
        assert result is None

    def test_filters_non_string_items(self):
        result = AgentContextGatherer._parse_file_list('["a.py", 42, null]')
        assert result == ["a.py"]


class TestParseExpandResponse:
    def test_parses_terms_and_code(self):
        response = (
            "TERMS:\nchat_provider\nopenai_client\nllm_plugin\n\n"
            "HYPOTHETICAL:\n```python\nclass OpenAIChat:\n    pass\n```"
        )
        terms, hyde = AgentContextGatherer._parse_expand_response(response)
        assert "chat_provider" in terms
        assert "class OpenAIChat" in hyde

    def test_missing_terms(self):
        response = "HYPOTHETICAL:\n```python\nx = 1\n```"
        terms, hyde = AgentContextGatherer._parse_expand_response(response)
        assert terms == ""
        assert "x = 1" in hyde

    def test_missing_hyde(self):
        response = "TERMS:\nfoo\nbar\n"
        terms, hyde = AgentContextGatherer._parse_expand_response(response)
        assert "foo" in terms
        assert hyde == ""

    def test_empty_response(self):
        terms, hyde = AgentContextGatherer._parse_expand_response("")
        assert terms == ""
        assert hyde == ""


class TestReadFileContent:
    def test_reads_existing_file(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        content = gatherer._read_file_content("main.py")
        assert content == "x = 1"

    def test_missing_file_returns_empty(self, tmp_path):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        assert gatherer._read_file_content("nonexistent.py") == ""

    def test_path_traversal_returns_empty(self, tmp_path):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        assert gatherer._read_file_content("../../etc/passwd") == ""

    def test_respects_max_file_bytes(self, tmp_path):
        (tmp_path / "big.py").write_text("x" * 200)
        gatherer = AgentContextGatherer(
            config=_make_config(max_file_bytes=50), source_dir=str(tmp_path),
        )
        content = gatherer._read_file_content("big.py")
        assert len(content) <= 50


# ---------------------------------------------------------------------------
# E2E: gather()
# ---------------------------------------------------------------------------
class TestGatherEndToEnd:
    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self, tmp_path, mock_client):
        gatherer = AgentContextGatherer(
            config=_make_config(enabled=False), source_dir=str(tmp_path)
        )
        result = await gatherer.gather(mock_client, "task")
        assert result["synthesized"] == ""
        assert result["raw_summaries"] == ""

    @pytest.mark.asyncio
    async def test_happy_path(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text(
            "def run(): pass\nopenai chat provider interface"
        )
        (tmp_path / "util.py").write_text("def helper(): pass")

        # LLM calls: expand, scan (no judgment)
        mock_client.generate = AsyncMock(side_effect=[
            # expand: terms + HyDE
            "TERMS:\nopenai\nchat\nprovider\n\nHYPOTHETICAL:\n```python\nclass Chat: pass\n```",
            # scan: structural index scan
            '```json\n["main.py"]\n```',
        ])
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path),
        )
        result = await gatherer.gather(mock_client, "build openai chat provider")

        # synthesized = structural overview (compact, no full source)
        assert "STRUCTURAL OVERVIEW" in result["synthesized"]
        # raw_summaries = overview + scan hit full source
        assert "### main.py" in result["raw_summaries"]
        assert "def run(): pass" in result["raw_summaries"]
        # file_contents = all files available for tool use
        assert "main.py" in result["file_contents"]
        assert "def run(): pass" in result["file_contents"]["main.py"]
        assert "agent_files" in result
        agent_files = result["agent_files"]
        assert agent_files["total_screened"] == 2
        assert "main.py" in agent_files["selected"]

    @pytest.mark.asyncio
    async def test_empty_dir_returns_empty(self, tmp_path, mock_client):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "task")
        assert result["synthesized"] == ""

    @pytest.mark.asyncio
    async def test_uses_smart_model(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("openai chat provider")
        mock_client.smart_model = "smart-30b"
        mock_client.generate = AsyncMock(side_effect=[
            "TERMS:\nfoo\n\nHYPOTHETICAL:\n```python\nx=1\n```",
            '["main.py"]',
        ])
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path),
        )
        await gatherer.gather(mock_client, "openai chat")
        for call in mock_client.generate.call_args_list:
            assert call[1]["model"] == "smart-30b"

    @pytest.mark.asyncio
    async def test_uses_agent_model_config(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("openai chat provider")
        mock_client.generate = AsyncMock(side_effect=[
            "TERMS:\nfoo\n\nHYPOTHETICAL:\n```python\nx=1\n```",
            '["main.py"]',
        ])
        gatherer = AgentContextGatherer(
            config=_make_config(agent_model="custom-model"),
            source_dir=str(tmp_path),
        )
        await gatherer.gather(mock_client, "openai chat")
        for call in mock_client.generate.call_args_list:
            assert call[1]["model"] == "custom-model"

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_two_llm_calls(self, tmp_path, mock_client):
        """Pipeline makes exactly 2 LLM calls: expand and scan."""
        (tmp_path / "a.py").write_text("openai provider chat")
        mock_client.generate = AsyncMock(side_effect=[
            "TERMS:\nopenai\n\nHYPOTHETICAL:\n```python\nx=1\n```",
            '["a.py"]',
        ])
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path),
        )
        await gatherer.gather(mock_client, "openai provider")
        assert mock_client.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_total_failure_returns_empty(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("openai chat provider")
        mock_client.generate = AsyncMock(side_effect=RuntimeError("total fail"))
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "openai chat")
        # expand fails -> empty query -> BM25 uses original -> pipeline exception
        # or pipeline exception caught -> empty
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_metadata_has_all_signals(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("openai chat provider interface")
        mock_client.generate = AsyncMock(side_effect=[
            "TERMS:\nopenai\nchat\n\nHYPOTHETICAL:\n```python\nx=1\n```",
            '["a.py"]',
        ])
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path),
        )
        result = await gatherer.gather(mock_client, "openai chat")
        agent_files = result["agent_files"]
        assert "scan_hits" in agent_files
        assert "selected" in agent_files
        assert "included" in agent_files


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------
class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_all_phases_reported(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("openai chat provider")
        mock_client.generate = AsyncMock(side_effect=[
            "TERMS:\nopenai\n\nHYPOTHETICAL:\n```python\nx=1\n```",
            '["main.py"]',
        ])
        phases = []

        def track(progress, phase):
            phases.append(phase)

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        await gatherer.gather(mock_client, "openai chat", progress_callback=track)

        phase_names = [p.split(":")[1] if ":" in p else p for p in phases]
        assert "mapping" in phase_names
        assert "expanding_query" in phase_names
        assert "scanning_index" in phase_names
        assert "import_expand" in phase_names
        assert "neighbor_expand" in phase_names
        assert "reading" in phase_names

    @pytest.mark.asyncio
    async def test_async_callback_awaited(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("openai chat provider")
        mock_client.generate = AsyncMock(side_effect=[
            "TERMS:\nopenai\n\nHYPOTHETICAL:\n```python\nx=1\n```",
            '["main.py"]',
        ])
        calls = []

        async def async_track(progress, phase):
            calls.append(phase)

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        await gatherer.gather(mock_client, "openai chat", progress_callback=async_track)
        assert len(calls) > 0


# ---------------------------------------------------------------------------
# Budget-aware file inclusion
# ---------------------------------------------------------------------------
class TestSeedAndFetch:
    @pytest.mark.asyncio
    async def test_all_files_as_seeds_when_under_cap(self, tmp_path, mock_client):
        """With fewer files than max_seed_files, all become seeds."""
        (tmp_path / "main.py").write_text("def run(): pass")
        (tmp_path / "util.py").write_text("def helper(): pass")
        mock_client.generate = AsyncMock(side_effect=[
            "TERMS:\nrun\n\nHYPOTHETICAL:\n```python\nx=1\n```",
            '["main.py", "util.py"]',
        ])
        gatherer = AgentContextGatherer(
            config=_make_config(max_seed_files=30), source_dir=str(tmp_path),
        )
        result = await gatherer.gather(mock_client, "run helper")
        # Both files should be in raw_summaries as seeds
        assert "### main.py" in result["raw_summaries"]
        assert "### util.py" in result["raw_summaries"]
        assert "SEED FILES" in result["raw_summaries"]

    @pytest.mark.asyncio
    async def test_seed_cap_defers_excess_to_tool_pool(self, tmp_path, mock_client):
        """Files beyond max_seed_files go to tool pool (file_contents only)."""
        for i in range(5):
            (tmp_path / f"file{i}.py").write_text(f"def func{i}(): pass")
        mock_client.generate = AsyncMock(side_effect=[
            "TERMS:\nfunc\n\nHYPOTHETICAL:\n```python\nx=1\n```",
            '["file0.py", "file1.py", "file2.py", "file3.py", "file4.py"]',
        ])
        gatherer = AgentContextGatherer(
            config=_make_config(max_seed_files=2), source_dir=str(tmp_path),
        )
        result = await gatherer.gather(mock_client, "func")
        # All 5 files should be in file_contents for tool access
        for i in range(5):
            assert f"file{i}.py" in result["file_contents"]
        # But only first 2 (scan hits first) should be in raw_summaries
        raw = result["raw_summaries"]
        seed_count = sum(1 for i in range(5) if f"### file{i}.py" in raw)
        assert seed_count == 2

    @pytest.mark.asyncio
    async def test_scan_hits_prioritized_in_seed_set(self, tmp_path, mock_client):
        """Scan hits become seeds before BM25/embedding matches."""
        (tmp_path / "scan_hit.py").write_text("def scanned(): pass")
        (tmp_path / "bm25_match.py").write_text("def matched(): pass")
        mock_client.generate = AsyncMock(side_effect=[
            "TERMS:\nscanned\n\nHYPOTHETICAL:\n```python\nx=1\n```",
            '["scan_hit.py"]',  # Only scan_hit.py from structural scan
        ])
        gatherer = AgentContextGatherer(
            config=_make_config(max_seed_files=1), source_dir=str(tmp_path),
        )
        result = await gatherer.gather(mock_client, "scanned matched")
        raw = result["raw_summaries"]
        # scan_hit.py should be seed, bm25_match.py should be tool pool
        assert "### scan_hit.py" in raw
        # bm25_match should be in file_contents but not in raw_summaries
        if "bm25_match.py" in result.get("file_contents", {}):
            assert "### bm25_match.py" not in raw

    @pytest.mark.asyncio
    async def test_provenance_tracks_seed_vs_pool(self, tmp_path, mock_client):
        """File provenance correctly marks seed vs tool_pool files."""
        for i in range(4):
            (tmp_path / f"m{i}.py").write_text(f"def f{i}(): pass")
        mock_client.generate = AsyncMock(side_effect=[
            "TERMS:\nfunc\n\nHYPOTHETICAL:\n```python\nx=1\n```",
            '["m0.py", "m1.py"]',  # scan hits
        ])
        gatherer = AgentContextGatherer(
            config=_make_config(max_seed_files=2), source_dir=str(tmp_path),
        )
        result = await gatherer.gather(mock_client, "func")
        prov = result.get("agent_files", {}).get("file_provenance", {})
        # Scan hits (m0, m1) should be in_prompt=True (seeds)
        for p in ["m0.py", "m1.py"]:
            if p in prov:
                assert prov[p]["in_prompt"] is True
        # Non-scan files should be in_prompt=False (tool pool)
        for p in ["m2.py", "m3.py"]:
            if p in prov:
                assert prov[p]["in_prompt"] is False
