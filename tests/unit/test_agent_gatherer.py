# tests/unit/test_agent_gatherer.py
"""Unit tests for AgentContextGatherer (multi-pass pipeline)."""

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
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("gitconfig")
        (tmp_path / "main.py").write_text("code")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        tree, paths = gatherer._build_file_tree()
        assert ".git" not in tree
        assert "main.py" in tree

    def test_skips_pycache(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "mod.cpython-311.pyc").write_bytes(b"\x00")
        (tmp_path / "app.py").write_text("code")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        tree, _ = gatherer._build_file_tree()
        assert "__pycache__" not in tree

    def test_skips_node_modules(self, tmp_path):
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.js").write_text("js")
        (tmp_path / "index.js").write_text("code")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        tree, _ = gatherer._build_file_tree()
        assert "node_modules" not in tree

    def test_skips_venv(self, tmp_path):
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "pip.py").write_text("pip")
        (tmp_path / "app.py").write_text("code")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        tree, _ = gatherer._build_file_tree()
        assert ".venv" not in tree

    def test_skips_non_indexable_extensions(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "lib.so").write_bytes(b"\x00")
        (tmp_path / "data.bin").write_bytes(b"\x00")
        (tmp_path / "no_extension").write_text("junk")
        (tmp_path / "app.py").write_text("code")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        tree, _ = gatherer._build_file_tree()
        assert "image.png" not in tree
        assert "lib.so" not in tree
        assert "data.bin" not in tree
        assert "no_extension" not in tree
        assert "app.py" in tree

    def test_caps_at_max_files(self, tmp_path):
        for i in range(_MAX_TREE_FILES + 10):
            (tmp_path / f"file_{i:04d}.py").write_text(f"# {i}")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        tree, paths = gatherer._build_file_tree()
        lines = tree.strip().splitlines()
        # _MAX_TREE_FILES entries + 1 truncation line
        assert len(lines) == _MAX_TREE_FILES + 1
        assert "truncated" in lines[-1]
        assert len(paths) == _MAX_TREE_FILES

    def test_uses_posix_paths(self, tmp_path):
        sub = tmp_path / "src" / "pkg"
        sub.mkdir(parents=True)
        (sub / "mod.py").write_text("code")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        tree, paths = gatherer._build_file_tree()
        assert "src/pkg/mod.py" in tree
        assert "\\" not in tree
        assert "src/pkg/mod.py" in paths

    def test_invalid_dir(self):
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir="/nonexistent/dir"
        )
        tree, paths = gatherer._build_file_tree()
        assert tree == ""
        assert paths == []

    def test_skips_non_indexable_in_subdirs(self, tmp_path):
        """Files in arbitrary dirs are skipped if extension isn't indexable."""
        junk = tmp_path / "some_cache"
        junk.mkdir()
        (junk / "blob_abc123").write_text("cached data")
        (tmp_path / "app.py").write_text("code")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        tree, _ = gatherer._build_file_tree()
        assert "blob_abc123" not in tree
        assert "app.py" in tree


# ---------------------------------------------------------------------------
# Pass 2: _build_index
# ---------------------------------------------------------------------------
class TestBuildIndex:
    def test_returns_structural_index(self, tmp_path):
        (tmp_path / "app.py").write_text("class MyApp:\n    def run(self): pass")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = gatherer._build_index(["app.py"])
        assert "## app.py" in index
        assert "MyApp" in index

    def test_empty_file_list(self, tmp_path):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = gatherer._build_index([])
        assert index == ""


# ---------------------------------------------------------------------------
# Pass 3: _navigate_files
# ---------------------------------------------------------------------------
class TestNavigateFiles:
    @pytest.mark.asyncio
    async def test_parses_json_array(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("code")
        (tmp_path / "utils.py").write_text("code")
        mock_client.generate.return_value = '["main.py", "utils.py"]'

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = "## main.py\nfunctions: main()\n\n## utils.py\nfunctions: helper()"
        selected = await gatherer._navigate_files(mock_client, "m", index, "task")
        assert "main.py" in selected
        assert "utils.py" in selected

    @pytest.mark.asyncio
    async def test_parses_json_object_with_files_key(self, tmp_path, mock_client):
        (tmp_path / "app.py").write_text("code")
        mock_client.generate.return_value = '{"files": ["app.py"]}'

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        selected = await gatherer._navigate_files(mock_client, "m", "## app.py\n", "task")
        assert selected == ["app.py"]

    @pytest.mark.asyncio
    async def test_caps_at_max_summary_files(self, tmp_path, mock_client):
        import json as _json

        for i in range(20):
            (tmp_path / f"f{i}.py").write_text("code")
        paths = [f"f{i}.py" for i in range(20)]
        mock_client.generate.return_value = _json.dumps(paths)

        gatherer = AgentContextGatherer(
            config=_make_config(max_summary_files=5), source_dir=str(tmp_path)
        )
        selected = await gatherer._navigate_files(mock_client, "m", "index", "task")
        assert len(selected) == 5

    @pytest.mark.asyncio
    async def test_heuristic_fallback_on_invalid_json(self, tmp_path, mock_client):
        (tmp_path / "README.md").write_text("# Hi")
        (tmp_path / "main.py").write_text("code")
        mock_client.generate.return_value = "not valid json at all"

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = "## README.md\nheadings: # Hi\n\n## main.py\nfunctions: main()"
        selected = await gatherer._navigate_files(mock_client, "m", index, "task")
        assert len(selected) > 0
        assert any("README" in p for p in selected)

    @pytest.mark.asyncio
    async def test_heuristic_fallback_on_llm_exception(self, tmp_path, mock_client):
        (tmp_path / "app.py").write_text("code")
        mock_client.generate.side_effect = RuntimeError("connection refused")

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = "## app.py\nfunctions: run()"
        selected = await gatherer._navigate_files(mock_client, "m", index, "task")
        assert len(selected) > 0

    @pytest.mark.asyncio
    async def test_filters_nonexistent_paths(self, tmp_path, mock_client):
        (tmp_path / "real.py").write_text("code")
        mock_client.generate.return_value = '["real.py", "fake.py"]'

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        selected = await gatherer._navigate_files(mock_client, "m", "index", "task")
        assert selected == ["real.py"]


# ---------------------------------------------------------------------------
# Heuristic select
# ---------------------------------------------------------------------------
class TestHeuristicSelect:
    def test_readme_and_pyproject_first_from_index(self):
        index = "## src/main.py\nfunctions: main()\n\n## README.md\nheadings: # Intro\n\n## pyproject.toml\nkeys: project"
        result = AgentContextGatherer._heuristic_select(index, "task")
        assert result[0] == "README.md"
        assert result[1] == "pyproject.toml"

    def test_shallow_init_prioritized(self):
        index = "## pkg/__init__.py\nexports: App\n\n## pkg/sub/deep/__init__.py\n\n## pkg/core.py\nclasses: Core"
        result = AgentContextGatherer._heuristic_select(index, "task")
        assert "pkg/__init__.py" in result[:2]

    def test_rest_sorted_by_depth(self):
        index = "## a/b/c/deep.py\n\n## top.py\n\n## a/mid.py\n"
        result = AgentContextGatherer._heuristic_select(index, "task")
        rest_idx = {p: i for i, p in enumerate(result)}
        assert rest_idx["top.py"] < rest_idx["a/mid.py"] < rest_idx["a/b/c/deep.py"]

    def test_falls_back_to_tree_format(self):
        tree = "file.py (1KB)\n... (truncated at 500 files)"
        result = AgentContextGatherer._heuristic_select(tree, "task")
        assert len(result) == 1
        assert result[0] == "file.py"


# ---------------------------------------------------------------------------
# Pass 4: _summarize_file
# ---------------------------------------------------------------------------
class TestSummarizeFile:
    @pytest.mark.asyncio
    async def test_reads_file_and_calls_generate(self, tmp_path, mock_client):
        (tmp_path / "app.py").write_text("def main(): pass")
        mock_client.generate.return_value = "**Purpose:** Entry point."

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._summarize_file(mock_client, "m", "app.py", "task")
        assert result == "**Purpose:** Entry point."
        mock_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_on_missing_file(self, tmp_path, mock_client):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._summarize_file(mock_client, "m", "nope.py", "task")
        assert result is None
        mock_client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_file(self, tmp_path, mock_client):
        (tmp_path / "empty.py").write_text("")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._summarize_file(mock_client, "m", "empty.py", "task")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_failure(self, tmp_path, mock_client):
        (tmp_path / "app.py").write_text("code")
        mock_client.generate.side_effect = RuntimeError("timeout")

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._summarize_file(mock_client, "m", "app.py", "task")
        assert result is None

    @pytest.mark.asyncio
    async def test_respects_max_file_bytes(self, tmp_path, mock_client):
        (tmp_path / "big.py").write_text("x" * 10_000)
        mock_client.generate.return_value = "summary"

        gatherer = AgentContextGatherer(
            config=_make_config(max_file_bytes=100), source_dir=str(tmp_path)
        )
        await gatherer._summarize_file(mock_client, "m", "big.py", "task")
        call_args = mock_client.generate.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "x" * 200 not in prompt_content

    @pytest.mark.asyncio
    async def test_rejects_path_traversal(self, tmp_path, mock_client):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._summarize_file(
            mock_client, "m", "../../etc/passwd", "task"
        )
        assert result is None


# ---------------------------------------------------------------------------
# Pass 5: _synthesize
# ---------------------------------------------------------------------------
class TestSynthesize:
    @pytest.mark.asyncio
    async def test_passes_summaries_to_generate(self, mock_client):
        mock_client.generate.return_value = "## Project Overview\nSynthesized."
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir="/tmp"
        )
        summaries = ["### main.py\nEntry point", "### utils.py\nHelpers"]
        result = await gatherer._synthesize(mock_client, "m", summaries, "task")
        assert "Synthesized" in result

    @pytest.mark.asyncio
    async def test_concatenation_fallback_on_failure(self, mock_client):
        mock_client.generate.side_effect = RuntimeError("timeout")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir="/tmp"
        )
        summaries = ["### main.py\nEntry point", "### utils.py\nHelpers"]
        result = await gatherer._synthesize(mock_client, "m", summaries, "task")
        assert "### main.py" in result
        assert "### utils.py" in result
        assert "File Summaries" in result


# ---------------------------------------------------------------------------
# End-to-end gather()
# ---------------------------------------------------------------------------
class TestGatherEndToEnd:
    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self, tmp_path, mock_client):
        gatherer = AgentContextGatherer(
            config=_make_config(enabled=False), source_dir=str(tmp_path)
        )
        result = await gatherer.gather(mock_client, "task")
        assert result == {"synthesized": "", "raw_summaries": ""}
        mock_client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_happy_path(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "utils.py").write_text("def helper(): pass")

        mock_client.generate.side_effect = [
            '["main.py", "utils.py"]',        # navigate
            "**Purpose:** Entry point.",       # summarize main.py
            "**Purpose:** Helpers.",            # summarize utils.py
            "## Project Overview\nFull doc.",   # synthesize
        ]

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "build a REST API")
        assert "## Project Overview" in result["synthesized"]
        assert "Entry point" in result["raw_summaries"]
        assert mock_client.generate.call_count == 4

    @pytest.mark.asyncio
    async def test_empty_dir_returns_empty(self, tmp_path, mock_client):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "task")
        assert result == {"synthesized": "", "raw_summaries": ""}

    @pytest.mark.asyncio
    async def test_all_summaries_fail_returns_empty(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("code")

        mock_client.generate.side_effect = [
            '["main.py"]',                     # navigate
            RuntimeError("LLM down"),          # summarize fails
        ]

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "task")
        assert result == {"synthesized": "", "raw_summaries": ""}

    @pytest.mark.asyncio
    async def test_uses_agent_model_config(self, tmp_path, mock_client):
        (tmp_path / "app.py").write_text("code")

        mock_client.generate.side_effect = [
            '["app.py"]',
            "summary",
            "## Overview\nDone.",
        ]

        gatherer = AgentContextGatherer(
            config=_make_config(agent_model="special-model"),
            source_dir=str(tmp_path),
        )
        await gatherer.gather(mock_client, "task")
        for call in mock_client.generate.call_args_list:
            assert call[1]["model"] == "special-model"

    @pytest.mark.asyncio
    async def test_falls_back_to_client_model(self, tmp_path, mock_client):
        (tmp_path / "app.py").write_text("code")

        mock_client.generate.side_effect = [
            '["app.py"]',
            "summary",
            "## Overview\nDone.",
        ]

        gatherer = AgentContextGatherer(
            config=_make_config(agent_model=None),
            source_dir=str(tmp_path),
        )
        await gatherer.gather(mock_client, "task")
        for call in mock_client.generate.call_args_list:
            assert call[1]["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_total_failure_returns_empty(self, tmp_path, mock_client):
        """Exception in pipeline returns empty dict (graceful degradation)."""
        (tmp_path / "app.py").write_text("code")
        mock_client.generate.side_effect = RuntimeError("total failure")

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "task")
        assert result == {"synthesized": "", "raw_summaries": ""}


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------
class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_all_phases_reported(self, tmp_path, mock_client):
        (tmp_path / "app.py").write_text("code")
        mock_client.generate.side_effect = [
            '["app.py"]',
            "summary",
            "## Overview\nDone.",
        ]

        phases = []

        def callback(progress, phase):
            phases.append(phase)

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        await gatherer.gather(mock_client, "task", progress_callback=callback)

        assert "agent:mapping" in phases
        assert "agent:indexing" in phases
        assert "agent:navigating" in phases
        assert any(p.startswith("agent:summarizing:") for p in phases)
        assert "agent:synthesizing" in phases

    @pytest.mark.asyncio
    async def test_async_callback_awaited(self, tmp_path, mock_client):
        (tmp_path / "app.py").write_text("code")
        mock_client.generate.side_effect = [
            '["app.py"]',
            "summary",
            "## Overview\nDone.",
        ]

        phases = []

        async def async_callback(progress, phase):
            phases.append(phase)

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        await gatherer.gather(mock_client, "task", progress_callback=async_callback)
        assert len(phases) >= 5  # mapping, indexing, navigating, summarizing, synthesizing


# ---------------------------------------------------------------------------
# _parse_file_list (static method)
# ---------------------------------------------------------------------------
class TestParseFileList:
    def test_plain_json_array(self):
        result = AgentContextGatherer._parse_file_list('["a.py", "b.py"]')
        assert result == ["a.py", "b.py"]

    def test_json_object_with_files(self):
        result = AgentContextGatherer._parse_file_list('{"files": ["a.py"]}')
        assert result == ["a.py"]

    def test_markdown_fenced(self):
        result = AgentContextGatherer._parse_file_list('```json\n["a.py"]\n```')
        assert result == ["a.py"]

    def test_invalid_json_returns_none(self):
        result = AgentContextGatherer._parse_file_list("not json")
        assert result is None

    def test_filters_non_string_items(self):
        result = AgentContextGatherer._parse_file_list('[1, "a.py", null]')
        assert result == ["a.py"]
