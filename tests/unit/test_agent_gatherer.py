# tests/unit/test_agent_gatherer.py
"""Unit tests for AgentContextGatherer (fitz-ai powered retrieval)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fitz_graveyard.config.schema import AgentConfig
from fitz_graveyard.planning.agent.gatherer import (
    AgentContextGatherer,
    _make_chat_factory,
)


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
    return client


def _make_read_result(file_path, content, origin="selected"):
    """Build a mock ReadResult matching fitz-ai's ReadResult shape."""
    address = MagicMock()
    address.metadata = {"origin": origin}
    address.score = {"selected": 1.0, "import": 0.9, "neighbor": 0.8}.get(origin, 0.8)
    result = MagicMock()
    result.file_path = file_path
    result.content = content
    result.address = address
    return result


# ---------------------------------------------------------------------------
# ChatFactory bridge
# ---------------------------------------------------------------------------
class TestChatFactoryBridge:
    @pytest.mark.asyncio
    async def test_bridge_maps_tiers(self, mock_client):
        mock_client.fast_model = "fast-4b"
        mock_client.mid_model = "balanced-9b"
        mock_client.smart_model = "smart-30b"

        loop = MagicMock()
        factory = _make_chat_factory(mock_client, loop)

        fast_chat = factory("fast")
        balanced_chat = factory("balanced")
        smart_chat = factory("smart")

        assert fast_chat._model == "fast-4b"
        assert balanced_chat._model == "balanced-9b"
        assert smart_chat._model == "smart-30b"


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
# E2E: gather() — mocks CodeRetriever
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

        mock_results = [
            _make_read_result("main.py", "def run(): pass", "selected"),
            _make_read_result("util.py", "def helper(): pass", "neighbor"),
        ]

        with patch(
            "fitz_graveyard.planning.agent.gatherer.CodeRetriever"
        ) as MockRetriever:
            instance = MockRetriever.return_value
            instance.retrieve.return_value = mock_results
            instance.get_file_paths.return_value = ["main.py", "util.py"]

            gatherer = AgentContextGatherer(
                config=_make_config(), source_dir=str(tmp_path),
            )
            result = await gatherer.gather(mock_client, "how does run work?")

        assert "STRUCTURAL OVERVIEW" in result["synthesized"]
        assert "FILE MANIFEST" in result["raw_summaries"]
        assert "main.py" in result["raw_summaries"]
        assert "main.py" in result["file_contents"]
        assert "file_index_entries" in result
        assert "agent_files" in result
        agent_files = result["agent_files"]
        assert agent_files["total_screened"] == 2
        assert "main.py" in agent_files["scan_hits"]

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty(self, tmp_path, mock_client):
        with patch(
            "fitz_graveyard.planning.agent.gatherer.CodeRetriever"
        ) as MockRetriever:
            instance = MockRetriever.return_value
            instance.retrieve.return_value = []
            instance.get_file_paths.return_value = ["main.py"]

            gatherer = AgentContextGatherer(
                config=_make_config(), source_dir=str(tmp_path),
            )
            result = await gatherer.gather(mock_client, "task")

        assert result["synthesized"] == ""
        assert result["raw_summaries"] == ""

    @pytest.mark.asyncio
    async def test_provenance_tracks_origins(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("class A: pass")
        (tmp_path / "b.py").write_text("class B: pass")
        (tmp_path / "c.py").write_text("class C: pass")

        mock_results = [
            _make_read_result("a.py", "class A: pass", "selected"),
            _make_read_result("b.py", "class B: pass", "import"),
            _make_read_result("c.py", "class C: pass", "neighbor"),
        ]

        with patch(
            "fitz_graveyard.planning.agent.gatherer.CodeRetriever"
        ) as MockRetriever:
            instance = MockRetriever.return_value
            instance.retrieve.return_value = mock_results
            instance.get_file_paths.return_value = ["a.py", "b.py", "c.py"]

            gatherer = AgentContextGatherer(
                config=_make_config(), source_dir=str(tmp_path),
            )
            result = await gatherer.gather(mock_client, "task")

        prov = result["agent_files"]["file_provenance"]
        assert prov["a.py"]["signals"] == ["scan"]
        assert prov["b.py"]["signals"] == ["import"]
        assert prov["c.py"]["signals"] == ["neighbor"]

    @pytest.mark.asyncio
    async def test_total_failure_returns_empty(self, tmp_path, mock_client):
        with patch(
            "fitz_graveyard.planning.agent.gatherer.CodeRetriever"
        ) as MockRetriever:
            instance = MockRetriever.return_value
            instance.retrieve.side_effect = RuntimeError("total fail")

            gatherer = AgentContextGatherer(
                config=_make_config(), source_dir=str(tmp_path),
            )
            result = await gatherer.gather(mock_client, "task")

        assert isinstance(result, dict)
        assert result["synthesized"] == ""

    @pytest.mark.asyncio
    async def test_retriever_receives_correct_config(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("x = 1")

        with patch(
            "fitz_graveyard.planning.agent.gatherer.CodeRetriever"
        ) as MockRetriever:
            instance = MockRetriever.return_value
            instance.retrieve.return_value = [
                _make_read_result("a.py", "x = 1", "selected"),
            ]
            instance.get_file_paths.return_value = ["a.py"]

            gatherer = AgentContextGatherer(
                config=_make_config(max_file_bytes=25_000),
                source_dir=str(tmp_path),
            )
            await gatherer.gather(mock_client, "task")

        # Verify CodeRetriever was constructed with correct params
        MockRetriever.assert_called_once()
        call_kwargs = MockRetriever.call_args[1]
        assert call_kwargs["max_file_bytes"] == 25_000
        assert str(call_kwargs["source_dir"]) == str(tmp_path)


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------
class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_all_phases_reported(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("def run(): pass")

        with patch(
            "fitz_graveyard.planning.agent.gatherer.CodeRetriever"
        ) as MockRetriever:
            instance = MockRetriever.return_value
            instance.retrieve.return_value = [
                _make_read_result("main.py", "def run(): pass", "selected"),
            ]
            instance.get_file_paths.return_value = ["main.py"]

            phases = []

            def track(progress, phase):
                phases.append(phase)

            gatherer = AgentContextGatherer(
                config=_make_config(), source_dir=str(tmp_path),
            )
            await gatherer.gather(mock_client, "task", progress_callback=track)

        phase_names = [p.split(":")[1] if ":" in p else p for p in phases]
        assert "mapping" in phase_names
        assert "scanning_index" in phase_names
        assert "reading" in phase_names

    @pytest.mark.asyncio
    async def test_async_callback_awaited(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("def run(): pass")

        with patch(
            "fitz_graveyard.planning.agent.gatherer.CodeRetriever"
        ) as MockRetriever:
            instance = MockRetriever.return_value
            instance.retrieve.return_value = [
                _make_read_result("main.py", "def run(): pass", "selected"),
            ]
            instance.get_file_paths.return_value = ["main.py"]

            calls = []

            async def async_track(progress, phase):
                calls.append(phase)

            gatherer = AgentContextGatherer(
                config=_make_config(), source_dir=str(tmp_path),
            )
            await gatherer.gather(
                mock_client, "task", progress_callback=async_track,
            )

        assert len(calls) > 0


# ---------------------------------------------------------------------------
# Seed-and-fetch
# ---------------------------------------------------------------------------
class TestSeedAndFetch:
    @pytest.mark.asyncio
    async def test_all_files_as_seeds_when_under_cap(self, tmp_path, mock_client):
        (tmp_path / "main.py").write_text("def run(): pass")
        (tmp_path / "util.py").write_text("def helper(): pass")

        mock_results = [
            _make_read_result("main.py", "def run(): pass", "selected"),
            _make_read_result("util.py", "def helper(): pass", "selected"),
        ]

        with patch(
            "fitz_graveyard.planning.agent.gatherer.CodeRetriever"
        ) as MockRetriever:
            instance = MockRetriever.return_value
            instance.retrieve.return_value = mock_results
            instance.get_file_paths.return_value = ["main.py", "util.py"]

            gatherer = AgentContextGatherer(
                config=_make_config(max_seed_files=30),
                source_dir=str(tmp_path),
            )
            result = await gatherer.gather(mock_client, "run helper")

        assert "main.py" in result["raw_summaries"]
        assert "util.py" in result["raw_summaries"]
        assert "FILE MANIFEST" in result["raw_summaries"]

    @pytest.mark.asyncio
    async def test_seed_cap_defers_excess_to_tool_pool(self, tmp_path, mock_client):
        for i in range(5):
            (tmp_path / f"file{i}.py").write_text(f"def func{i}(): pass")

        mock_results = [
            _make_read_result(f"file{i}.py", f"def func{i}(): pass", "selected")
            for i in range(5)
        ]

        with patch(
            "fitz_graveyard.planning.agent.gatherer.CodeRetriever"
        ) as MockRetriever:
            instance = MockRetriever.return_value
            instance.retrieve.return_value = mock_results
            instance.get_file_paths.return_value = [
                f"file{i}.py" for i in range(5)
            ]

            gatherer = AgentContextGatherer(
                config=_make_config(max_seed_files=2),
                source_dir=str(tmp_path),
            )
            result = await gatherer.gather(mock_client, "func")

        # All 5 in file_contents for tool access
        for i in range(5):
            assert f"file{i}.py" in result["file_contents"]
        # All 5 in manifest (no seed/pool split — all files in manifest)
        raw = result["raw_summaries"]
        for i in range(5):
            assert f"file{i}.py" in raw

    @pytest.mark.asyncio
    async def test_scan_hits_prioritized_in_seed_set(self, tmp_path, mock_client):
        (tmp_path / "scan_hit.py").write_text("def scanned(): pass")
        (tmp_path / "neighbor.py").write_text("def matched(): pass")

        mock_results = [
            _make_read_result("scan_hit.py", "def scanned(): pass", "selected"),
            _make_read_result("neighbor.py", "def matched(): pass", "neighbor"),
        ]

        with patch(
            "fitz_graveyard.planning.agent.gatherer.CodeRetriever"
        ) as MockRetriever:
            instance = MockRetriever.return_value
            instance.retrieve.return_value = mock_results
            instance.get_file_paths.return_value = [
                "scan_hit.py", "neighbor.py",
            ]

            gatherer = AgentContextGatherer(
                config=_make_config(max_seed_files=1),
                source_dir=str(tmp_path),
            )
            result = await gatherer.gather(mock_client, "scanned")

        raw = result["raw_summaries"]
        # Both files in manifest (no seed cap — all files listed)
        assert "scan_hit.py" in raw
        assert "neighbor.py" in raw

    @pytest.mark.asyncio
    async def test_provenance_tracks_seed_vs_pool(self, tmp_path, mock_client):
        for i in range(4):
            (tmp_path / f"m{i}.py").write_text(f"def f{i}(): pass")

        mock_results = [
            _make_read_result(f"m{i}.py", f"def f{i}(): pass", "selected")
            for i in range(4)
        ]

        with patch(
            "fitz_graveyard.planning.agent.gatherer.CodeRetriever"
        ) as MockRetriever:
            instance = MockRetriever.return_value
            instance.retrieve.return_value = mock_results
            instance.get_file_paths.return_value = [
                f"m{i}.py" for i in range(4)
            ]

            gatherer = AgentContextGatherer(
                config=_make_config(max_seed_files=2),
                source_dir=str(tmp_path),
            )
            result = await gatherer.gather(mock_client, "func")

        prov = result["agent_files"]["file_provenance"]
        # No files inlined in prompt (manifest-only approach)
        in_prompt_count = sum(1 for p in prov.values() if p["in_prompt"])
        assert in_prompt_count == 0
        # All files should have provenance
        assert len(prov) == 4
