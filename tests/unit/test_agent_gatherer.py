# tests/unit/test_agent_gatherer.py
"""Unit tests for AgentContextGatherer (brute-force parallel screening)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from fitz_graveyard.config.schema import AgentConfig
from fitz_graveyard.planning.agent.gatherer import (
    AgentContextGatherer,
    _MAX_TREE_FILES,
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
# Pass 2: _screen_file / _screen_all / _parse_yes_no
# ---------------------------------------------------------------------------
class TestParseYesNo:
    def test_yes(self):
        assert AgentContextGatherer._parse_yes_no("YES") is True

    def test_no(self):
        assert AgentContextGatherer._parse_yes_no("NO") is False

    def test_yes_with_explanation(self):
        assert AgentContextGatherer._parse_yes_no("YES this file is relevant") is True

    def test_no_with_explanation(self):
        assert AgentContextGatherer._parse_yes_no("NO this is unrelated") is False

    def test_lowercase_yes(self):
        assert AgentContextGatherer._parse_yes_no("yes") is True

    def test_lowercase_no(self):
        assert AgentContextGatherer._parse_yes_no("no") is False

    def test_whitespace(self):
        assert AgentContextGatherer._parse_yes_no("  YES  ") is True

    def test_empty(self):
        assert AgentContextGatherer._parse_yes_no("") is False

    def test_garbage(self):
        assert AgentContextGatherer._parse_yes_no("maybe") is False


class TestScreenFile:
    @pytest.mark.asyncio
    async def test_relevant_file(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("def run(): pass")
        mock_client.generate = AsyncMock(return_value="YES")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._screen_file(
            mock_client, "test-model", "main.py", "add logging"
        )
        assert result is True
        mock_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_irrelevant_file(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("def run(): pass")
        mock_client.generate = AsyncMock(return_value="NO")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._screen_file(
            mock_client, "test-model", "main.py", "add logging"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_missing_file(self, tmp_path, mock_client):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._screen_file(
            mock_client, "test-model", "nonexistent.py", "add logging"
        )
        assert result is False
        mock_client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_file(self, tmp_path, mock_client):
        (tmp_path / "empty.py").write_text("")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._screen_file(
            mock_client, "test-model", "empty.py", "add logging"
        )
        assert result is False
        mock_client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_failure(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("def run(): pass")
        mock_client.generate = AsyncMock(side_effect=RuntimeError("boom"))
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._screen_file(
            mock_client, "test-model", "main.py", "add logging"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_respects_max_file_bytes(self, tmp_path, mock_client):
        (tmp_path / "big.py").write_text("x" * 200)
        mock_client.generate = AsyncMock(return_value="YES")
        gatherer = AgentContextGatherer(
            config=_make_config(max_file_bytes=50), source_dir=str(tmp_path)
        )
        await gatherer._screen_file(
            mock_client, "test-model", "big.py", "task"
        )
        prompt = mock_client.generate.call_args[1]["messages"][0]["content"]
        # Content should be truncated
        assert len(prompt) < 200 + 200  # prompt template + content

    @pytest.mark.asyncio
    async def test_rejects_path_traversal(self, tmp_path, mock_client):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._screen_file(
            mock_client, "test-model", "../../etc/passwd", "task"
        )
        assert result is False
        mock_client.generate.assert_not_called()


class TestScreenAll:
    @pytest.mark.asyncio
    async def test_returns_relevant_files(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("def a(): pass")
        (tmp_path / "b.py").write_text("def b(): pass")
        (tmp_path / "c.py").write_text("def c(): pass")
        # a=YES, b=NO, c=YES
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
        mock_client.generate = AsyncMock(
            side_effect=[RuntimeError("boom"), "YES"]
        )
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._screen_all(
            mock_client, "test-model", ["a.py", "b.py"], "task",
        )
        # a fails (treated as not relevant), b succeeds
        assert "b.py" in result
        assert "a.py" not in result


# ---------------------------------------------------------------------------
# Pass 4: _read_raw_source
# ---------------------------------------------------------------------------
class TestReadRawSource:
    def test_reads_files_as_fenced_blocks(self, tmp_path):
        (tmp_path / "main.py").write_text("def run(): pass")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result, included = gatherer._read_raw_source(["main.py"], ["main.py"])
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
        result, included = gatherer._read_raw_source(["api.py"], ["api.py"])
        assert "INTERFACE SIGNATURES" in result
        assert "chat(prompt: str) -> str" in result
        # Signatures should come before the raw source
        sig_pos = result.index("INTERFACE SIGNATURES")
        source_pos = result.index("### api.py")
        assert sig_pos < source_pos

    def test_budget_truncation(self, tmp_path):
        # Create 3 files, each ~5000 chars, with budget of 10000
        for name in ["a.py", "b.py", "c.py"]:
            (tmp_path / name).write_text(f"# {name}\n" + "x = 1\n" * 800)
        gatherer = AgentContextGatherer(
            config=_make_config(max_context_chars=10_000),
            source_dir=str(tmp_path),
        )
        result, included = gatherer._read_raw_source(
            ["a.py", "b.py", "c.py"], ["a.py", "b.py", "c.py"]
        )
        # Should include some but not all files
        assert len(included) < 3
        assert len(included) >= 1
        assert len(result) <= 10_000

    def test_connectivity_ordering(self, tmp_path):
        # a.py imports b.py — b has higher connectivity, should come first
        (tmp_path / "a.py").write_text("from b import x\ny = 1")
        (tmp_path / "b.py").write_text("x = 42")
        (tmp_path / "c.py").write_text("z = 99")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result, included = gatherer._read_raw_source(
            ["a.py", "b.py", "c.py"], ["a.py", "b.py", "c.py"]
        )
        # b.py should appear before c.py (b has connections, c has none)
        b_pos = result.index("### b.py")
        c_pos = result.index("### c.py")
        assert b_pos < c_pos

    def test_skips_missing_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result, included = gatherer._read_raw_source(
            ["a.py", "missing.py"], ["a.py", "missing.py"]
        )
        assert "a.py" in included
        assert "missing.py" not in included

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.py").write_text("")
        (tmp_path / "real.py").write_text("x = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result, included = gatherer._read_raw_source(
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
        result, included = gatherer._read_raw_source(["big.py"], ["big.py"])
        assert "big.py" in included
        # Content should be truncated at 50 bytes
        content_lines = result.split("```")[1]
        assert len(content_lines.strip()) <= 50

    def test_empty_selected_returns_empty(self, tmp_path):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result, included = gatherer._read_raw_source([], [])
        assert result == ""
        assert included == []


# ---------------------------------------------------------------------------
# Pass 3: _import_expand
# ---------------------------------------------------------------------------
class TestImportExpand:
    def test_adds_direct_imports(self, tmp_path):
        """If a.py imports b.py, and a.py is relevant, b.py should be added."""
        (tmp_path / "a.py").write_text("from b import helper")
        (tmp_path / "b.py").write_text("def helper(): pass")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._import_expand(["a.py"], ["a.py", "b.py"])
        assert "a.py" in result
        assert "b.py" in result

    def test_no_duplicates(self, tmp_path):
        """If b.py is already relevant, import expand shouldn't duplicate it."""
        (tmp_path / "a.py").write_text("from b import helper")
        (tmp_path / "b.py").write_text("def helper(): pass")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._import_expand(["a.py", "b.py"], ["a.py", "b.py"])
        assert result.count("a.py") == 1
        assert result.count("b.py") == 1

    def test_no_imports_returns_same(self, tmp_path):
        """Files with no imports should return the same list."""
        (tmp_path / "a.py").write_text("x = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._import_expand(["a.py"], ["a.py"])
        assert result == ["a.py"]

    def test_depth_one_only(self, tmp_path):
        """Import expansion should be depth 1 only (a->b, not a->b->c)."""
        (tmp_path / "a.py").write_text("from b import x")
        (tmp_path / "b.py").write_text("from c import y\nx = 1")
        (tmp_path / "c.py").write_text("y = 2")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._import_expand(["a.py"], ["a.py", "b.py", "c.py"])
        assert "a.py" in result
        assert "b.py" in result
        # c.py should NOT be added (depth 2)
        assert "c.py" not in result


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
        (tmp_path / "main.py").write_text("def run(): pass")
        (tmp_path / "util.py").write_text("def helper(): pass")

        # Calls: broad_screen(main.py)=YES, broad_screen(util.py)=NO,
        #        refine_screen(main.py)=YES
        # No summarize/synthesize — raw source is read directly
        mock_client.generate = AsyncMock(
            side_effect=["YES", "NO", "YES"]
        )
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "add logging")

        # Raw source should contain actual file content
        assert "def run(): pass" in result["synthesized"]
        assert "def run(): pass" in result["raw_summaries"]
        assert "### main.py" in result["raw_summaries"]
        assert "agent_files" in result
        agent_files = result["agent_files"]
        assert agent_files["total_screened"] == 2
        assert "main.py" in agent_files["relevant"]
        assert "util.py" not in agent_files["relevant"]
        assert "included" in agent_files

    @pytest.mark.asyncio
    async def test_all_screened_out_returns_empty(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("x = 1")
        mock_client.generate = AsyncMock(return_value="NO")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "task")
        assert result["synthesized"] == ""

    @pytest.mark.asyncio
    async def test_empty_dir_returns_empty(self, tmp_path, mock_client):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "task")
        assert result["synthesized"] == ""

    @pytest.mark.asyncio
    async def test_uses_agent_model_config(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("x = 1")
        # broad=YES, refine=YES (no summarize/synthesize)
        mock_client.generate = AsyncMock(
            side_effect=["YES", "YES"]
        )
        gatherer = AgentContextGatherer(
            config=_make_config(agent_model="custom-model"),
            source_dir=str(tmp_path),
        )
        await gatherer.gather(mock_client, "task")
        for call in mock_client.generate.call_args_list:
            assert call[1]["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_falls_back_to_client_model(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("x = 1")
        # broad=YES, refine=YES
        mock_client.generate = AsyncMock(
            side_effect=["YES", "YES"]
        )
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        await gatherer.gather(mock_client, "task")
        for call in mock_client.generate.call_args_list:
            assert call[1]["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_uses_correct_tier_per_pass(self, tmp_path):
        """Broad screen=fast, refine=mid. No summarize/synthesize LLM calls."""
        client = MagicMock()
        client.fast_model = "fast-3b"
        client.mid_model = "mid-30b"
        client.smart_model = "smart-27b"
        (tmp_path / "main.py").write_text("x = 1")
        # broad=YES, refine=YES
        client.generate = AsyncMock(
            side_effect=["YES", "YES"]
        )
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        await gatherer.gather(client, "task")

        calls = client.generate.call_args_list
        assert len(calls) == 2  # Only screening calls, no summarize/synthesize
        # First call is broad screening -> fast model
        assert calls[0][1]["model"] == "fast-3b"
        # Second call is refine screening -> mid model
        assert calls[1][1]["model"] == "mid-30b"

    @pytest.mark.asyncio
    async def test_agent_model_overrides_tiers(self, tmp_path):
        """When agent_model is set, it overrides both fast and smart tiers."""
        client = MagicMock()
        client.fast_model = "fast-3b"
        client.smart_model = "smart-30b"
        (tmp_path / "main.py").write_text("x = 1")
        # broad=YES, refine=YES
        client.generate = AsyncMock(
            side_effect=["YES", "YES"]
        )
        gatherer = AgentContextGatherer(
            config=_make_config(agent_model="override-model"),
            source_dir=str(tmp_path),
        )
        await gatherer.gather(client, "task")
        for call in client.generate.call_args_list:
            assert call[1]["model"] == "override-model"

    @pytest.mark.asyncio
    async def test_total_failure_returns_empty(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("x = 1")
        mock_client.generate = AsyncMock(side_effect=RuntimeError("total fail"))
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "task")
        assert result["synthesized"] == ""
        assert result["raw_summaries"] == ""

    @pytest.mark.asyncio
    async def test_caps_at_max_summary_files(self, tmp_path, mock_client):
        # Create 5 files, max_summary_files=2
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text(f"x = {i}")

        # broad=YES*5, refine=YES*5 (no summarize/synthesize)
        responses = ["YES"] * 5 + ["YES"] * 5
        mock_client.generate = AsyncMock(side_effect=responses)
        gatherer = AgentContextGatherer(
            config=_make_config(max_summary_files=2),
            source_dir=str(tmp_path),
        )
        result = await gatherer.gather(mock_client, "task")
        agent_files = result["agent_files"]
        assert len(agent_files["relevant"]) == 5
        assert len(agent_files["selected"]) == 2


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------
class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_all_phases_reported(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("x = 1")
        # broad=YES, refine=YES
        mock_client.generate = AsyncMock(
            side_effect=["YES", "YES"]
        )
        phases = []

        def track(progress, phase):
            phases.append(phase)

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        await gatherer.gather(mock_client, "task", progress_callback=track)

        phase_prefixes = [p.split(":")[1] if ":" in p else p for p in phases]
        assert "mapping" in phase_prefixes
        assert "screening" in phase_prefixes or any("screening" in p for p in phases)
        assert any("refining" in p for p in phases)
        assert any("reading" in p for p in phases)

    @pytest.mark.asyncio
    async def test_async_callback_awaited(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("x = 1")
        # broad=YES, refine=YES
        mock_client.generate = AsyncMock(
            side_effect=["YES", "YES"]
        )
        calls = []

        async def async_track(progress, phase):
            calls.append(phase)

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        await gatherer.gather(mock_client, "task", progress_callback=async_track)
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
