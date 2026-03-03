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
from fitz_graveyard.planning.agent.indexer import _CLUSTERING_THRESHOLD


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
            '[]',                              # discover
            "Refined: build REST API with main and utils",  # rewrite query
            '[]',                              # re-navigate
            "## Project Overview\nFull doc.",   # synthesize
        ]

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "build a REST API")
        assert "## Project Overview" in result["synthesized"]
        assert "Entry point" in result["raw_summaries"]
        assert mock_client.generate.call_count == 7

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
            '[]',                              # discover
            "Refined task for special model",   # rewrite query
            '[]',                              # re-navigate
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
            '[]',                              # discover
            "Refined task description here",    # rewrite query
            '[]',                              # re-navigate
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
            '[]',                              # discover
            "Refined task description here",    # rewrite query
            '[]',                              # re-navigate
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
        assert "agent:discovering" in phases
        assert "agent:rewriting_query" in phases
        assert "agent:re_navigating" in phases
        assert "agent:synthesizing" in phases

    @pytest.mark.asyncio
    async def test_async_callback_awaited(self, tmp_path, mock_client):
        (tmp_path / "app.py").write_text("code")
        mock_client.generate.side_effect = [
            '["app.py"]',
            "summary",
            '[]',                              # discover
            "Refined task description here",    # rewrite query
            '[]',                              # re-navigate
            "## Overview\nDone.",
        ]

        phases = []

        async def async_callback(progress, phase):
            phases.append(phase)

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        await gatherer.gather(mock_client, "task", progress_callback=async_callback)
        assert len(phases) >= 8  # mapping, indexing, navigating, summarizing, discovering, rewriting, re-navigating, synthesizing


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


# ---------------------------------------------------------------------------
# Two-tier directory selection
# ---------------------------------------------------------------------------
class TestTwoTierSelection:
    @pytest.mark.asyncio
    async def test_small_codebase_skips_clustering(self, tmp_path, mock_client):
        """Under threshold: single-pass, no dir selection call."""
        for i in range(5):
            (tmp_path / f"mod{i}.py").write_text(f"def fn{i}(): pass")

        mock_client.generate.side_effect = [
            '["mod0.py", "mod1.py"]',           # navigate
            "summary 0",                         # summarize mod0
            "summary 1",                         # summarize mod1
            '[]',                                # discover
            "Refined task description here",     # rewrite query
            '[]',                                # re-navigate
            "## Overview\nSynthesized.",          # synthesize
        ]

        phases = []
        def callback(progress, phase):
            phases.append(phase)

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "task", progress_callback=callback)
        assert result["synthesized"] != ""
        assert "agent:selecting_dirs" not in phases
        # 7 LLM calls: navigate + 2 summarize + discover + rewrite + re-nav + synthesize
        assert mock_client.generate.call_count == 7

    @pytest.mark.asyncio
    async def test_large_codebase_triggers_clustering(self, tmp_path, mock_client):
        """At/above threshold: two-tier path with dir selection."""
        # Create enough files to trigger clustering
        for i in range(_CLUSTERING_THRESHOLD + 10):
            d = tmp_path / f"pkg{i % 5}"
            d.mkdir(exist_ok=True)
            (d / f"mod{i}.py").write_text(f"def fn{i}(): pass")

        mock_client.generate.side_effect = [
            '["pkg0/"]',                          # select dirs
            '["pkg0/mod0.py"]',                   # navigate
            "summary 0",                          # summarize
            '[]',                                 # discover
            "Refined task description here",      # rewrite query
            '[]',                                 # re-navigate
            "## Overview\nSynthesized.",           # synthesize
        ]

        phases = []
        def callback(progress, phase):
            phases.append(phase)

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "task", progress_callback=callback)
        assert result["synthesized"] != ""
        assert "agent:selecting_dirs" in phases

    @pytest.mark.asyncio
    async def test_dir_selection_failure_falls_back(self, tmp_path, mock_client):
        """If dir selection LLM fails, fall back to all dirs."""
        for i in range(_CLUSTERING_THRESHOLD + 10):
            d = tmp_path / f"pkg{i % 3}"
            d.mkdir(exist_ok=True)
            (d / f"mod{i}.py").write_text(f"def fn{i}(): pass")

        mock_client.generate.side_effect = [
            "not valid json",                     # dir selection fails
            '["pkg0/mod0.py"]',                   # navigate (with all dirs)
            "summary",                            # summarize
            '[]',                                 # discover
            "Refined task description here",      # rewrite query
            '[]',                                 # re-navigate
            "## Overview\nDone.",                  # synthesize
        ]

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "task")
        assert result["synthesized"] != ""

    @pytest.mark.asyncio
    async def test_caller_expansion_in_gather(self, tmp_path, mock_client):
        """Caller expansion adds files that import from selected files."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "governor.py").write_text("class GovernanceDecision: pass\n")
        (pkg / "engine.py").write_text(
            "from pkg.governor import GovernanceDecision\ndef run(): pass\n"
        )

        mock_client.generate.side_effect = [
            '["pkg/governor.py"]',              # navigate — only picks governor
            "**Purpose:** Governance decisions.", # summarize governor
            "**Purpose:** Engine.",              # summarize engine (added by expansion)
            '[]',                               # discover
            "Refined task description here",    # rewrite query
            '[]',                               # re-navigate
            "## Overview\nDone.",                # synthesize
        ]

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "add webhook to governance")
        assert result["synthesized"] != ""
        # Should have 7 calls: navigate + 2 summarize + discover + rewrite + re-nav + synthesize
        assert mock_client.generate.call_count == 7

    @pytest.mark.asyncio
    async def test_dir_selection_validates_names(self, tmp_path, mock_client):
        """Only dirs that actually exist in groups are accepted."""
        for i in range(_CLUSTERING_THRESHOLD + 10):
            d = tmp_path / f"pkg{i % 3}"
            d.mkdir(exist_ok=True)
            (d / f"mod{i}.py").write_text(f"def fn{i}(): pass")

        mock_client.generate.side_effect = [
            '["pkg0/", "nonexistent/"]',          # dir selection — one valid, one not
            '["pkg0/mod0.py"]',                   # navigate
            "summary",                            # summarize
            '[]',                                 # discover
            "Refined task description here",      # rewrite query
            '[]',                                 # re-navigate
            "## Overview\nDone.",                  # synthesize
        ]

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer.gather(mock_client, "task")
        assert result["synthesized"] != ""


# ---------------------------------------------------------------------------
# _expand_with_callers
# ---------------------------------------------------------------------------
class TestExtractTaskKeywords:
    def test_extracts_meaningful_words(self):
        keywords = AgentContextGatherer._extract_task_keywords(
            "add token usage tracking to LLM calls"
        )
        assert "token" in keywords
        assert "usage" in keywords
        assert "tracking" in keywords
        assert "llm" in keywords
        assert "calls" in keywords

    def test_filters_stopwords(self):
        keywords = AgentContextGatherer._extract_task_keywords(
            "build a REST API for the users"
        )
        assert "rest" in keywords
        assert "api" in keywords
        assert "users" in keywords
        # Stopwords excluded
        assert "build" not in keywords
        assert "the" not in keywords
        assert "for" not in keywords

    def test_filters_short_words(self):
        keywords = AgentContextGatherer._extract_task_keywords("do it on db")
        assert "do" not in keywords
        assert "it" not in keywords
        assert "on" not in keywords
        assert "db" not in keywords

    def test_handles_underscores(self):
        keywords = AgentContextGatherer._extract_task_keywords(
            "track token_usage across queries"
        )
        assert "token_usage" in keywords

    def test_empty_description(self):
        assert AgentContextGatherer._extract_task_keywords("") == set()


class TestScoreByKeywords:
    def test_path_component_match(self):
        score = AgentContextGatherer._score_by_keywords(
            "llm/providers/ollama.py", {"llm", "ollama"}
        )
        assert score == 2

    def test_no_match_scores_zero(self):
        score = AgentContextGatherer._score_by_keywords(
            "ingestion/loader.py", {"token", "llm"}
        )
        assert score == 0

    def test_stem_match(self):
        score = AgentContextGatherer._score_by_keywords(
            "metrics/token_tracker.py", {"token"}
        )
        assert score == 1

    def test_empty_keywords(self):
        score = AgentContextGatherer._score_by_keywords("any/path.py", set())
        assert score == 0

    def test_substring_match_in_component(self):
        """Keyword 'governance' matches directory 'governance'."""
        score = AgentContextGatherer._score_by_keywords(
            "fitz_ai/governance/conflict_aware.py", {"governance"}
        )
        assert score == 1

# ---------------------------------------------------------------------------
# Discovery pass: _discover_additional
# ---------------------------------------------------------------------------
class TestDiscoverAdditional:
    @pytest.mark.asyncio
    async def test_discovers_new_files(self, tmp_path, mock_client):
        """Returns validated paths not already in selected."""
        (tmp_path / "selected.py").write_text("x = 1\n")
        (tmp_path / "discovered.py").write_text("y = 2\n")

        mock_client.generate.return_value = '["discovered.py"]'
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._discover_additional(
            mock_client, "m",
            selected=["selected.py"],
            summaries=["### selected.py\nSome summary"],
            structural_index="## selected.py\n## discovered.py\n",
            job_description="task",
        )
        assert result == ["discovered.py"]

    @pytest.mark.asyncio
    async def test_skips_already_selected(self, tmp_path, mock_client):
        """Paths already in selected are filtered out."""
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")

        mock_client.generate.return_value = '["a.py", "b.py"]'
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._discover_additional(
            mock_client, "m",
            selected=["a.py"],
            summaries=["### a.py\nSummary"],
            structural_index="## a.py\n## b.py\n",
            job_description="task",
        )
        assert result == ["b.py"]

    @pytest.mark.asyncio
    async def test_respects_max_discover(self, tmp_path, mock_client):
        """Caps at max_discover even if LLM returns more."""
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f"x = {i}\n")

        paths = [f"file{i}.py" for i in range(10)]
        mock_client.generate.return_value = str(paths).replace("'", '"')
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._discover_additional(
            mock_client, "m",
            selected=[],
            summaries=[],
            structural_index="index",
            job_description="task",
            max_discover=3,
        )
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self, tmp_path, mock_client):
        """LLM exception returns empty list."""
        mock_client.generate.side_effect = RuntimeError("timeout")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._discover_additional(
            mock_client, "m",
            selected=[],
            summaries=[],
            structural_index="index",
            job_description="task",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_validates_paths_exist(self, tmp_path, mock_client):
        """Non-existent paths filtered out."""
        (tmp_path / "real.py").write_text("x = 1\n")

        mock_client.generate.return_value = '["real.py", "fake.py"]'
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._discover_additional(
            mock_client, "m",
            selected=[],
            summaries=[],
            structural_index="index",
            job_description="task",
        )
        assert result == ["real.py"]

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty(self, tmp_path, mock_client):
        """Invalid JSON from LLM returns empty list."""
        mock_client.generate.return_value = "not json at all"
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._discover_additional(
            mock_client, "m",
            selected=[],
            summaries=[],
            structural_index="index",
            job_description="task",
        )
        assert result == []


# ---------------------------------------------------------------------------
# Re-query pass: _rewrite_query + _re_navigate
# ---------------------------------------------------------------------------
class TestRewriteQuery:
    @pytest.mark.asyncio
    async def test_returns_rewritten_text(self, mock_client):
        rewritten = "Track token usage in ChatProvider.chat() across ThreadPoolExecutor workers"
        mock_client.generate.return_value = rewritten

        gatherer = AgentContextGatherer(config=_make_config(), source_dir="/tmp")
        result = await gatherer._rewrite_query(
            mock_client, "m", "track LLM tokens",
            summaries=["### base.py\nDefines ChatProvider protocol"],
        )
        assert result == rewritten
        mock_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, mock_client):
        mock_client.generate.side_effect = RuntimeError("timeout")

        gatherer = AgentContextGatherer(config=_make_config(), source_dir="/tmp")
        result = await gatherer._rewrite_query(
            mock_client, "m", "original task", summaries=[],
        )
        assert result == "original task"

    @pytest.mark.asyncio
    async def test_fallback_on_short_response(self, mock_client):
        mock_client.generate.return_value = "too short"

        gatherer = AgentContextGatherer(config=_make_config(), source_dir="/tmp")
        result = await gatherer._rewrite_query(
            mock_client, "m", "original task", summaries=[],
        )
        assert result == "original task"


class TestReNavigate:
    @pytest.mark.asyncio
    async def test_finds_new_files(self, tmp_path, mock_client):
        (tmp_path / "already.py").write_text("x = 1\n")
        (tmp_path / "new_file.py").write_text("y = 2\n")

        mock_client.generate.return_value = '["new_file.py"]'
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._re_navigate(
            mock_client, "m",
            structural_index="## already.py\n## new_file.py\n",
            rewritten_query="refined task",
            already_selected=["already.py"],
        )
        assert result == ["new_file.py"]

    @pytest.mark.asyncio
    async def test_skips_already_selected(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")

        mock_client.generate.return_value = '["a.py", "b.py"]'
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._re_navigate(
            mock_client, "m",
            structural_index="## a.py\n## b.py\n",
            rewritten_query="refined task",
            already_selected=["a.py"],
        )
        assert result == ["b.py"]

    @pytest.mark.asyncio
    async def test_annotates_index(self, tmp_path, mock_client):
        """Verify structural index headers get [ALREADY ANALYZED] markers."""
        (tmp_path / "new.py").write_text("x = 1\n")

        mock_client.generate.return_value = '["new.py"]'
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        await gatherer._re_navigate(
            mock_client, "m",
            structural_index="## old.py\nfunctions: foo()\n\n## new.py\nfunctions: bar()\n",
            rewritten_query="task",
            already_selected=["old.py"],
        )
        # Check the prompt sent to the LLM contains the annotation
        call_args = mock_client.generate.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "[ALREADY ANALYZED]" in prompt
        assert "## old.py  [ALREADY ANALYZED]" in prompt

    @pytest.mark.asyncio
    async def test_respects_max_files(self, tmp_path, mock_client):
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f"x = {i}\n")

        paths = [f"file{i}.py" for i in range(10)]
        mock_client.generate.return_value = str(paths).replace("'", '"')
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._re_navigate(
            mock_client, "m",
            structural_index="index",
            rewritten_query="task",
            already_selected=[],
            max_files=3,
        )
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, tmp_path, mock_client):
        mock_client.generate.side_effect = RuntimeError("timeout")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._re_navigate(
            mock_client, "m",
            structural_index="index",
            rewritten_query="task",
            already_selected=[],
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_validates_paths_exist(self, tmp_path, mock_client):
        (tmp_path / "real.py").write_text("x = 1\n")

        mock_client.generate.return_value = '["real.py", "fake.py"]'
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = await gatherer._re_navigate(
            mock_client, "m",
            structural_index="index",
            rewritten_query="task",
            already_selected=[],
        )
        assert result == ["real.py"]


class TestExpandWithCallers:
    def test_adds_importers(self, tmp_path):
        """Selected=[governor.py], engine imports governor -> engine added."""
        (tmp_path / "governor.py").write_text("class Gov: pass\n")
        (tmp_path / "engine.py").write_text("from governor import Gov\n")
        (tmp_path / "unrelated.py").write_text("import os\n")

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._expand_with_callers(
            ["governor.py"], ["governor.py", "engine.py", "unrelated.py"], 15,
            "add governance webhook",
        )
        assert "governor.py" in result
        assert "engine.py" in result
        assert "unrelated.py" not in result

    def test_respects_max_files_for_irrelevant(self, tmp_path):
        """Irrelevant callers don't exceed max_files total."""
        (tmp_path / "base.py").write_text("x = 1\n")
        for i in range(10):
            (tmp_path / f"caller{i}.py").write_text("from base import x\n")

        files = ["base.py"] + [f"caller{i}.py" for i in range(10)]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        # No keyword matches, so only base_room (5-1=4) callers added
        result = gatherer._expand_with_callers(
            ["base.py"], files, 5, "unrelated task description",
        )
        assert len(result) <= 5
        assert result[0] == "base.py"

    def test_no_duplicates(self, tmp_path):
        """Caller already in selected -> not duplicated."""
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("from a import x\n")

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._expand_with_callers(
            ["a.py", "b.py"], ["a.py", "b.py"], 15, "some task",
        )
        assert result == ["a.py", "b.py"]

    def test_no_callers_unchanged(self, tmp_path):
        """No file imports selected -> unchanged."""
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._expand_with_callers(
            ["a.py"], ["a.py", "b.py"], 15, "some task",
        )
        assert result == ["a.py"]

    def test_preserves_llm_order(self, tmp_path):
        """LLM-selected files stay in original order, callers appended."""
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        (tmp_path / "c.py").write_text("from a import x\nfrom b import y\n")

        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._expand_with_callers(
            ["b.py", "a.py"], ["a.py", "b.py", "c.py"], 15, "some task",
        )
        assert result[0] == "b.py"
        assert result[1] == "a.py"
        assert "c.py" in result

    def test_multi_hop_transitive(self, tmp_path):
        """governor -> decider -> engine: selecting governor finds engine at 2 hops."""
        (tmp_path / "governor.py").write_text("class Gov: pass\n")
        (tmp_path / "decider.py").write_text("from governor import Gov\n")
        (tmp_path / "engine.py").write_text("from decider import something\n")

        files = ["governor.py", "decider.py", "engine.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._expand_with_callers(
            ["governor.py"], files, 15, "add governance webhook",
        )
        assert "decider.py" in result
        assert "engine.py" in result

    def test_multi_hop_three_levels(self, tmp_path):
        """a -> b -> c -> d: selecting a finds d at 3 hops."""
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("from a import x\n")
        (tmp_path / "c.py").write_text("from b import x\n")
        (tmp_path / "d.py").write_text("from c import x\n")

        files = ["a.py", "b.py", "c.py", "d.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._expand_with_callers(
            ["a.py"], files, 15, "some task",
        )
        assert result[0] == "a.py"
        assert set(result) == {"a.py", "b.py", "c.py", "d.py"}

    def test_multi_hop_stops_at_budget(self, tmp_path):
        """Irrelevant callers capped by base_room budget."""
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("from a import x\n")
        (tmp_path / "c.py").write_text("from b import x\n")
        (tmp_path / "d.py").write_text("from c import x\n")

        files = ["a.py", "b.py", "c.py", "d.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        # max_files=3, selected=1 → base_room=2, bonus_room=1
        # No keyword matches → only 2 irrelevant callers added
        result = gatherer._expand_with_callers(
            ["a.py"], files, 3, "unrelated task",
        )
        assert len(result) <= 3
        assert result[0] == "a.py"

    def test_multi_hop_lazy_imports(self, tmp_path):
        """Lazy imports inside methods are followed transitively."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "governor.py").write_text("class Gov: pass\n")
        (pkg / "decider.py").write_text("from pkg.governor import Gov\n")
        engines = tmp_path / "engines"
        engines.mkdir()
        (engines / "engine.py").write_text(
            "class Engine:\n"
            "    def init(self):\n"
            "        from pkg.decider import Decider\n"
        )

        files = ["pkg/governor.py", "pkg/decider.py", "engines/engine.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._expand_with_callers(
            ["pkg/governor.py"], files, 15, "governance webhook",
        )
        assert "pkg/decider.py" in result
        assert "engines/engine.py" in result

    def test_multi_hop_no_cycles(self, tmp_path):
        """Circular imports don't cause infinite loop."""
        (tmp_path / "a.py").write_text("from b import y\nx = 1\n")
        (tmp_path / "b.py").write_text("from a import x\ny = 2\n")

        files = ["a.py", "b.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._expand_with_callers(
            ["a.py"], files, 15, "some task",
        )
        assert "b.py" in result
        # No duplicates
        assert len(result) == len(set(result))

    def test_keyword_callers_ranked_first(self, tmp_path):
        """Keyword-matching callers come before non-matching ones."""
        (tmp_path / "base.py").write_text("x = 1\n")
        # alpha_first.py would be first alphabetically
        (tmp_path / "alpha_first.py").write_text("from base import x\n")
        # llm_consumer.py matches keyword "llm"
        llm_dir = tmp_path / "llm"
        llm_dir.mkdir()
        (llm_dir / "consumer.py").write_text("from base import x\n")

        files = ["base.py", "alpha_first.py", "llm/consumer.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._expand_with_callers(
            ["base.py"], files, 15, "track LLM token usage",
        )
        assert result[0] == "base.py"
        # llm/consumer.py should come before alpha_first.py (keyword match)
        llm_idx = result.index("llm/consumer.py")
        alpha_idx = result.index("alpha_first.py")
        assert llm_idx < alpha_idx

    def test_keyword_callers_get_bonus_slots(self, tmp_path):
        """Keyword-matching callers can exceed base max_files."""
        (tmp_path / "base.py").write_text("x = 1\n")
        # 3 keyword-matching callers in llm/ directory
        llm = tmp_path / "llm"
        llm.mkdir()
        for name in ["provider.py", "client.py", "tracker.py"]:
            (llm / name).write_text("from base import x\n")
        # 1 non-matching caller
        (tmp_path / "unrelated.py").write_text("from base import x\n")

        files = ["base.py", "llm/provider.py", "llm/client.py",
                 "llm/tracker.py", "unrelated.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        # max_files=2, selected=1 → base_room=1, bonus_room=1
        # 3 keyword-matching callers, cap at base_room+bonus_room=2
        result = gatherer._expand_with_callers(
            ["base.py"], files, 2, "track LLM usage",
        )
        # Should include base.py + 2 keyword-matching callers (exceeding max_files=2)
        llm_callers = [r for r in result if r.startswith("llm/")]
        assert len(llm_callers) == 2
        assert len(result) == 3  # 1 selected + 2 keyword callers

    def test_no_keyword_matches_uses_alphabetical(self, tmp_path):
        """When no callers match keywords, fall back to alphabetical."""
        (tmp_path / "base.py").write_text("x = 1\n")
        (tmp_path / "zebra.py").write_text("from base import x\n")
        (tmp_path / "alpha.py").write_text("from base import x\n")

        files = ["base.py", "zebra.py", "alpha.py"]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._expand_with_callers(
            ["base.py"], files, 15, "completely unrelated task xyz",
        )
        callers = result[1:]  # skip base.py
        assert callers == sorted(callers)

    def test_dynamic_budget_proportional(self, tmp_path):
        """Bonus room scales with len(selected)."""
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        llm = tmp_path / "llm"
        llm.mkdir()
        for i in range(10):
            (llm / f"caller{i}.py").write_text("from a import x\nfrom b import y\n")

        files = ["a.py", "b.py"] + [f"llm/caller{i}.py" for i in range(10)]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        # max_files=3, selected=2 → base_room=1, bonus_room=2
        # 10 keyword-matching callers, capped at base_room+bonus_room=3
        result = gatherer._expand_with_callers(
            ["a.py", "b.py"], files, 3, "track LLM usage",
        )
        # 2 selected + 3 keyword callers = 5 total
        assert len(result) == 5
        assert result[0] == "a.py"
        assert result[1] == "b.py"
