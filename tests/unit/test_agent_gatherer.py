# tests/unit/test_agent_gatherer.py
"""Unit tests for AgentContextGatherer (BM25 + per-file LLM confirm)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from fitz_graveyard.config.schema import AgentConfig
from fitz_graveyard.planning.agent.gatherer import (
    AgentContextGatherer,
    _MAX_TREE_FILES,
    _tokenize,
)


def _make_config(**kwargs):
    defaults = dict(enabled=True, max_summary_files=15, max_file_bytes=50_000)
    defaults.update(kwargs)
    return AgentConfig(**defaults)


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.model = "test-model"
    client.fast_model = "test-model"
    client.mid_model = "test-model"
    client.smart_model = "test-model"
    client.generate = AsyncMock(return_value="LLM response")
    return client


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class TestTokenize:
    def test_basic_tokenization(self):
        result = _tokenize("build an openai chat plugin")
        assert "build" in result
        assert "openai" in result
        assert "chat" in result
        assert "plugin" in result
        assert "an" not in result

    def test_removes_stopwords(self):
        result = _tokenize("the quick and simple test")
        assert "the" not in result
        assert "and" not in result
        assert "quick" in result
        assert "simple" in result
        assert "test" in result

    def test_removes_python_keywords(self):
        result = _tokenize("import os from pathlib def main class Foo")
        assert "import" not in result
        assert "from" not in result
        assert "def" not in result
        assert "class" not in result
        assert "pathlib" in result
        assert "main" in result
        assert "foo" in result

    def test_lowercases(self):
        result = _tokenize("OpenAI ChatPlugin")
        assert "openai" in result
        assert "chatplugin" in result

    def test_splits_on_non_alphanumeric(self):
        result = _tokenize("foo/bar.py")
        assert "foo" in result
        assert "bar" in result
        assert "py" in result

    def test_empty_input(self):
        assert _tokenize("") == []

    def test_removes_single_char_tokens(self):
        result = _tokenize("a b c dd ee")
        assert "dd" in result
        assert "ee" in result
        assert len([t for t in result if len(t) == 1]) == 0


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
# Pass 2: BM25 screening
# ---------------------------------------------------------------------------
class TestBm25Screen:
    def test_scores_relevant_files_higher(self, tmp_path):
        (tmp_path / "openai_provider.py").write_text(
            "class OpenAIProvider:\n    def chat(self, prompt): pass"
        )
        (tmp_path / "utils.py").write_text(
            "def format_string(s): return s.strip()"
        )
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        paths, scores = gatherer._bm25_screen(
            ["openai_provider.py", "utils.py"],
            "build an openai chat plugin",
            top_k=10,
        )
        assert len(paths) > 0
        assert paths[0] == "openai_provider.py"

    def test_returns_top_k(self, tmp_path):
        for i in range(10):
            (tmp_path / f"file_{i}.py").write_text(f"keyword_{i} = {i}")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        all_paths = [f"file_{i}.py" for i in range(10)]
        paths, scores = gatherer._bm25_screen(all_paths, "keyword_0 keyword_1", top_k=3)
        assert len(paths) <= 3

    def test_empty_query_returns_empty(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        paths, scores = gatherer._bm25_screen(["main.py"], "the and or", top_k=10)
        assert paths == []
        assert scores == []

    def test_no_files_returns_empty(self, tmp_path):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        paths, scores = gatherer._bm25_screen([], "some query", top_k=10)
        assert paths == []

    def test_filters_zero_score_files(self, tmp_path):
        (tmp_path / "relevant.py").write_text("openai chat provider interface")
        (tmp_path / "unrelated.py").write_text("zzzzz qqqqq wwwww")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        paths, scores = gatherer._bm25_screen(
            ["relevant.py", "unrelated.py"],
            "openai chat",
            top_k=10,
        )
        assert "relevant.py" in paths
        for s in scores:
            assert s > 0

    def test_path_bonus_boosts_matching_paths(self, tmp_path):
        (tmp_path / "openai.py").write_text("provider = 1")
        (tmp_path / "other.py").write_text("provider = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        paths, scores = gatherer._bm25_screen(
            ["openai.py", "other.py"],
            "openai provider",
            top_k=10,
        )
        assert paths[0] == "openai.py"

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.py").write_text("")
        (tmp_path / "real.py").write_text("openai chat provider")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        paths, scores = gatherer._bm25_screen(
            ["empty.py", "real.py"], "openai", top_k=10,
        )
        assert "empty.py" not in paths
        assert "real.py" in paths

    def test_skips_markdown_files(self, tmp_path):
        (tmp_path / "docs.md").write_text("openai chat provider plugin interface")
        (tmp_path / "code.py").write_text("openai chat provider")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        paths, scores = gatherer._bm25_screen(
            ["docs.md", "code.py"], "openai chat", top_k=10,
        )
        assert "code.py" in paths
        assert "docs.md" not in paths

    def test_scores_are_descending(self, tmp_path):
        for i in range(5):
            content = "openai " * (i + 1) + "filler " * 20
            (tmp_path / f"f{i}.py").write_text(content)
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        all_paths = [f"f{i}.py" for i in range(5)]
        paths, scores = gatherer._bm25_screen(all_paths, "openai", top_k=10)
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Pass 3: Per-file LLM confirm
# ---------------------------------------------------------------------------
class TestScreenFile:
    @pytest.mark.asyncio
    async def test_screens_relevant(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("def engine(): pass")
        mock_client.generate = AsyncMock(return_value="YES")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._screen_file(
            mock_client, "test-model", "a.py", "build engine",
        )
        assert result == ("a.py", True)
        mock_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_screens_irrelevant(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("def utils(): pass")
        mock_client.generate = AsyncMock(return_value="NO")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._screen_file(
            mock_client, "test-model", "a.py", "build engine",
        )
        assert result == ("a.py", False)

    @pytest.mark.asyncio
    async def test_empty_file_no_llm_call(self, tmp_path, mock_client):
        (tmp_path / "empty.py").write_text("")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._screen_file(
            mock_client, "test-model", "empty.py", "task",
        )
        mock_client.generate.assert_not_called()
        assert result == ("empty.py", False)

    @pytest.mark.asyncio
    async def test_llm_failure(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("x = 1")
        mock_client.generate = AsyncMock(side_effect=RuntimeError("boom"))
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._screen_file(
            mock_client, "test-model", "a.py", "task",
        )
        assert result == ("a.py", False)

    @pytest.mark.asyncio
    async def test_uses_screen_prompt(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("x = 1")
        mock_client.generate = AsyncMock(return_value="YES")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        await gatherer._screen_file(
            mock_client, "test-model", "a.py", "build engine",
        )
        prompt = mock_client.generate.call_args[1]["messages"][0]["content"]
        assert "build engine" in prompt
        assert "FILE: a.py" in prompt
        assert "x = 1" in prompt

    @pytest.mark.asyncio
    async def test_uses_temperature_zero(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("x = 1")
        mock_client.generate = AsyncMock(return_value="YES")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        await gatherer._screen_file(
            mock_client, "test-model", "a.py", "task",
        )
        assert mock_client.generate.call_args[1]["temperature"] == 0

    @pytest.mark.asyncio
    async def test_yes_with_explanation(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("x = 1")
        mock_client.generate = AsyncMock(return_value="YES - this is the main engine")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._screen_file(
            mock_client, "test-model", "a.py", "task",
        )
        assert result == ("a.py", True)


class TestScreenAll:
    @pytest.mark.asyncio
    async def test_returns_relevant_files(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("def a(): pass")
        (tmp_path / "b.py").write_text("def b(): pass")
        (tmp_path / "c.py").write_text("def c(): pass")
        mock_client.generate = AsyncMock(
            side_effect=["YES", "NO", "YES"]
        )
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._screen_all(
            mock_client, "test-model",
            ["a.py", "b.py", "c.py"],
            "some task",
        )
        assert "a.py" in result
        assert "b.py" not in result
        assert "c.py" in result

    @pytest.mark.asyncio
    async def test_one_call_per_file(self, tmp_path, mock_client):
        for name in ["a.py", "b.py", "c.py", "d.py"]:
            (tmp_path / name).write_text(f"content of {name}")
        mock_client.generate = AsyncMock(
            side_effect=["YES", "YES", "YES", "YES"]
        )
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        await gatherer._screen_all(
            mock_client, "test-model",
            ["a.py", "b.py", "c.py", "d.py"],
            "task",
        )
        assert mock_client.generate.call_count == 4

    @pytest.mark.asyncio
    async def test_all_irrelevant(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("x = 1")
        mock_client.generate = AsyncMock(return_value="NO")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._screen_all(
            mock_client, "test-model", ["a.py"], "task",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_handles_llm_failures_gracefully(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("y = 2")
        (tmp_path / "c.py").write_text("z = 3")
        mock_client.generate = AsyncMock(
            side_effect=[RuntimeError("boom"), "NO", "YES"]
        )
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._screen_all(
            mock_client, "test-model", ["a.py", "b.py", "c.py"], "task",
        )
        assert "c.py" in result
        assert "a.py" not in result


# ---------------------------------------------------------------------------
# Pass 4: _read_raw_source
# ---------------------------------------------------------------------------
class TestReadRawSource:
    def test_reads_files_as_fenced_blocks(self, tmp_path):
        (tmp_path / "main.py").write_text("def run(): pass")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result, included, _, _ = gatherer._read_raw_source(["main.py"], ["main.py"])
        assert "### main.py" in result
        assert "def run(): pass" in result
        assert "```" in result
        assert included == ["main.py"]

    def test_prepends_interface_signatures(self, tmp_path):
        (tmp_path / "api.py").write_text(
            "class ChatProvider:\n"
            "    def chat(self, prompt: str) -> str:\n"
            "        pass\n"
        )
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result, included, _, _ = gatherer._read_raw_source(["api.py"], ["api.py"])
        assert "INTERFACE SIGNATURES" in result
        assert "chat(prompt: str) -> str" in result
        sig_pos = result.index("INTERFACE SIGNATURES")
        source_pos = result.index("### api.py")
        assert sig_pos < source_pos

    def test_budget_truncation(self, tmp_path):
        for name in ["a.py", "b.py", "c.py"]:
            (tmp_path / name).write_text(f"# {name}\n" + "x = 1\n" * 800)
        gatherer = AgentContextGatherer(
            config=_make_config(max_context_chars=10_000),
            source_dir=str(tmp_path),
        )
        result, included, _, _ = gatherer._read_raw_source(
            ["a.py", "b.py", "c.py"], ["a.py", "b.py", "c.py"]
        )
        assert len(included) < 3
        assert len(included) >= 1
        assert len(result) <= 10_000

    def test_connectivity_ordering(self, tmp_path):
        (tmp_path / "a.py").write_text("from b import x\ny = 1")
        (tmp_path / "b.py").write_text("x = 42")
        (tmp_path / "c.py").write_text("z = 99")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result, included, _, _ = gatherer._read_raw_source(
            ["a.py", "b.py", "c.py"], ["a.py", "b.py", "c.py"]
        )
        b_pos = result.index("### b.py")
        c_pos = result.index("### c.py")
        assert b_pos < c_pos

    def test_skips_missing_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result, included, _, _ = gatherer._read_raw_source(
            ["a.py", "missing.py"], ["a.py", "missing.py"]
        )
        assert "a.py" in included
        assert "missing.py" not in included

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.py").write_text("")
        (tmp_path / "real.py").write_text("x = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result, included, _, _ = gatherer._read_raw_source(
            ["empty.py", "real.py"], ["empty.py", "real.py"]
        )
        assert "real.py" in included
        assert "empty.py" not in included

    def test_respects_max_file_bytes(self, tmp_path):
        (tmp_path / "big.py").write_text("x" * 200)
        gatherer = AgentContextGatherer(
            config=_make_config(max_file_bytes=50),
            source_dir=str(tmp_path),
        )
        result, included, _, _ = gatherer._read_raw_source(["big.py"], ["big.py"])
        assert "big.py" in included
        content_lines = result.split("```")[1]
        assert len(content_lines.strip()) <= 50

    def test_empty_selected_returns_empty(self, tmp_path):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result, included, _, _ = gatherer._read_raw_source([], [])
        assert result == ""
        assert included == []


# ---------------------------------------------------------------------------
# Pass 4: _import_expand
# ---------------------------------------------------------------------------
class TestImportExpand:
    def test_adds_direct_imports(self, tmp_path):
        (tmp_path / "a.py").write_text("from b import helper")
        (tmp_path / "b.py").write_text("def helper(): pass")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._import_expand(["a.py"], ["a.py", "b.py"])
        assert "a.py" in result
        assert "b.py" in result

    def test_no_duplicates(self, tmp_path):
        (tmp_path / "a.py").write_text("from b import helper")
        (tmp_path / "b.py").write_text("def helper(): pass")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._import_expand(["a.py", "b.py"], ["a.py", "b.py"])
        assert result.count("a.py") == 1
        assert result.count("b.py") == 1

    def test_no_imports_returns_same(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._import_expand(["a.py"], ["a.py"])
        assert result == ["a.py"]

    def test_depth_two_forward(self, tmp_path):
        """Depth 2 BFS catches transitive imports (a→b→c)."""
        (tmp_path / "a.py").write_text("from b import x")
        (tmp_path / "b.py").write_text("from c import y\nx = 1")
        (tmp_path / "c.py").write_text("y = 2")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._import_expand(["a.py"], ["a.py", "b.py", "c.py"])
        assert "a.py" in result
        assert "b.py" in result
        assert "c.py" in result  # depth 2 catches this

    def test_depth_three_not_reached(self, tmp_path):
        """Depth 2 BFS stops — d.py at depth 3 is not reached."""
        (tmp_path / "a.py").write_text("from b import x")
        (tmp_path / "b.py").write_text("from c import y\nx = 1")
        (tmp_path / "c.py").write_text("from d import z\ny = 2")
        (tmp_path / "d.py").write_text("z = 3")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._import_expand(["a.py"], ["a.py", "b.py", "c.py", "d.py"])
        assert "a.py" in result
        assert "b.py" in result
        assert "c.py" in result
        assert "d.py" not in result  # depth 3, beyond BFS limit

    def test_reverse_imports(self, tmp_path):
        """Reverse direction: files that import a relevant file are discovered."""
        (tmp_path / "base.py").write_text("class Base: pass")
        (tmp_path / "child.py").write_text("from base import Base")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        # base.py is relevant; child.py imports it → discovered via reverse
        result = gatherer._import_expand(["base.py"], ["base.py", "child.py"])
        assert "base.py" in result
        assert "child.py" in result


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

        # BM25 shortlists both. Batched screen: main.py=YES, util.py=NO
        mock_client.generate = AsyncMock(
            return_value="main.py: YES\nutil.py: NO"
        )
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path),
        )
        result = await gatherer.gather(mock_client, "build openai chat provider")

        assert "def run(): pass" in result["synthesized"]
        assert "### main.py" in result["raw_summaries"]
        assert "agent_files" in result
        agent_files = result["agent_files"]
        assert agent_files["total_screened"] == 2
        assert "main.py" in agent_files["relevant"]
        assert "util.py" not in agent_files["relevant"]

    @pytest.mark.asyncio
    async def test_all_screened_out_falls_back_to_bm25(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("openai provider chat interface")
        mock_client.generate = AsyncMock(return_value="main.py: NO")
        gatherer = AgentContextGatherer(
            config=_make_config(max_summary_files=5), source_dir=str(tmp_path),
        )
        result = await gatherer.gather(mock_client, "openai chat")
        assert result["synthesized"] != ""
        assert "main.py" in result["agent_files"]["relevant"]

    @pytest.mark.asyncio
    async def test_empty_dir_returns_empty(self, tmp_path, mock_client):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "task")
        assert result["synthesized"] == ""

    @pytest.mark.asyncio
    async def test_uses_agent_model_config(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("openai chat provider")
        mock_client.generate = AsyncMock(return_value="main.py: YES")
        gatherer = AgentContextGatherer(
            config=_make_config(agent_model="custom-model"),
            source_dir=str(tmp_path),
        )
        await gatherer.gather(mock_client, "openai chat")
        for call in mock_client.generate.call_args_list:
            assert call[1]["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_falls_back_to_mid_model(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("openai chat provider")
        mock_client.mid_model = "mid-30b"
        mock_client.generate = AsyncMock(return_value="main.py: YES")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path),
        )
        await gatherer.gather(mock_client, "openai chat")
        for call in mock_client.generate.call_args_list:
            assert call[1]["model"] == "mid-30b"

    @pytest.mark.asyncio
    async def test_one_llm_call_per_file(self, tmp_path, mock_client):
        """4 BM25 candidates -> 4 individual LLM calls."""
        for name in ["a.py", "b.py", "c.py", "d.py"]:
            (tmp_path / name).write_text(f"openai provider {name}")
        mock_client.generate = AsyncMock(
            side_effect=["YES", "NO", "YES", "NO"]
        )
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path),
        )
        await gatherer.gather(mock_client, "openai provider")
        assert mock_client.generate.call_count == 4

    @pytest.mark.asyncio
    async def test_total_failure_returns_empty(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("openai chat provider")
        mock_client.generate = AsyncMock(side_effect=RuntimeError("total fail"))
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "openai chat")
        # All screening fails -> fallback to BM25 -> still works
        assert result["synthesized"] != ""

    @pytest.mark.asyncio
    async def test_caps_at_max_summary_files(self, tmp_path, mock_client):
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text(f"target keyword relevant {i}")

        mock_client.generate = AsyncMock(return_value="YES")
        gatherer = AgentContextGatherer(
            config=_make_config(max_summary_files=2),
            source_dir=str(tmp_path),
        )
        result = await gatherer.gather(mock_client, "target keyword relevant")
        agent_files = result["agent_files"]
        assert len(agent_files["selected"]) == 2

    @pytest.mark.asyncio
    async def test_bm25_candidates_in_metadata(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("openai chat provider interface")
        mock_client.generate = AsyncMock(return_value="a.py: YES")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path),
        )
        result = await gatherer.gather(mock_client, "openai chat")
        assert "bm25_candidates" in result["agent_files"]
        assert isinstance(result["agent_files"]["bm25_candidates"], list)


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------
class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_all_phases_reported(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("openai chat provider")
        mock_client.generate = AsyncMock(return_value="main.py: YES")
        phases = []

        def track(progress, phase):
            phases.append(phase)

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        await gatherer.gather(mock_client, "openai chat", progress_callback=track)

        phase_names = [p.split(":")[1] if ":" in p else p for p in phases]
        assert "mapping" in phase_names
        assert "screening" in phase_names
        assert "confirming" in phase_names
        assert "reading" in phase_names

    @pytest.mark.asyncio
    async def test_async_callback_awaited(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("openai chat provider")
        mock_client.generate = AsyncMock(return_value="main.py: YES")
        calls = []

        async def async_track(progress, phase):
            calls.append(phase)

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        await gatherer.gather(mock_client, "openai chat", progress_callback=async_track)
        assert len(calls) > 0


# ---------------------------------------------------------------------------
# Helpers
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
