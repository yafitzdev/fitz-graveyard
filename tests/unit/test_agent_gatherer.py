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
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config.py").write_text("x = 1")
        (tmp_path / "main.py").write_text("x = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        _, paths = gatherer._build_file_tree()
        assert "main.py" in paths
        assert not any(".git" in p for p in paths)

    def test_skips_pycache(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "mod.py").write_text("x = 1")
        (tmp_path / "app.py").write_text("x = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        _, paths = gatherer._build_file_tree()
        assert not any("__pycache__" in p for p in paths)

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "index.js").write_text("x")
        (tmp_path / "app.py").write_text("x = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        _, paths = gatherer._build_file_tree()
        assert not any("node_modules" in p for p in paths)

    def test_skips_venv(self, tmp_path):
        venv = tmp_path / "venv"
        venv.mkdir()
        (venv / "pip.py").write_text("x")
        (tmp_path / "app.py").write_text("x = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        _, paths = gatherer._build_file_tree()
        assert not any("venv" in p for p in paths)

    def test_skips_non_indexable_extensions(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "model.pkl").write_bytes(b"pickle")
        (tmp_path / "main.py").write_text("x = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        _, paths = gatherer._build_file_tree()
        assert "main.py" in paths
        assert len(paths) == 1

    def test_caps_at_max_files(self, tmp_path):
        for i in range(_MAX_TREE_FILES + 50):
            (tmp_path / f"mod{i}.py").write_text(f"x = {i}")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        tree, paths = gatherer._build_file_tree()
        assert len(paths) == _MAX_TREE_FILES
        assert "truncated" in tree

    def test_uses_posix_paths(self, tmp_path):
        sub = tmp_path / "pkg" / "sub"
        sub.mkdir(parents=True)
        (sub / "mod.py").write_text("x")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        _, paths = gatherer._build_file_tree()
        assert "pkg/sub/mod.py" in paths

    def test_invalid_dir(self):
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir="/nonexistent/path"
        )
        tree, paths = gatherer._build_file_tree()
        assert tree == ""
        assert paths == []


# ---------------------------------------------------------------------------
# Pass 2: _build_index
# ---------------------------------------------------------------------------
class TestBuildIndex:
    def test_returns_structural_index(self, tmp_path):
        (tmp_path / "app.py").write_text("class Foo: pass")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = gatherer._build_index(["app.py"])
        assert "## app.py" in index
        assert "Foo" in index

    def test_empty_file_list(self, tmp_path):
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = gatherer._build_index([])
        assert index == ""


# ---------------------------------------------------------------------------
# Pass 3: _keyword_match (deterministic seed selection)
# ---------------------------------------------------------------------------
class TestExtractKeywords:
    def test_extracts_meaningful_words(self):
        kw = AgentContextGatherer._extract_keywords(
            "add token usage tracking to LLM calls"
        )
        assert "token" in kw
        assert "usage" in kw
        assert "tracking" in kw
        assert "llm" in kw

    def test_filters_stopwords(self):
        kw = AgentContextGatherer._extract_keywords("build a REST API for users")
        assert "rest" in kw
        assert "api" in kw
        assert "users" in kw
        assert "build" not in kw
        assert "the" not in kw

    def test_filters_short_words(self):
        kw = AgentContextGatherer._extract_keywords("do it on db")
        assert "do" not in kw
        assert "db" not in kw

    def test_empty(self):
        assert AgentContextGatherer._extract_keywords("") == set()


class TestKeywordMatch:
    def test_matches_by_path(self, tmp_path):
        """File path components count as matches."""
        (tmp_path / "token_tracker.py").write_text("x = 1")
        (tmp_path / "unrelated.py").write_text("x = 1")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = gatherer._build_index(["token_tracker.py", "unrelated.py"])
        result = gatherer._keyword_match(index, "token tracking", max_seeds=5)
        assert "token_tracker.py" in result

    def test_matches_by_class_name(self, tmp_path):
        """Class names in structural index are searchable."""
        (tmp_path / "core.py").write_text("class TokenTracker: pass")
        (tmp_path / "other.py").write_text("class Unrelated: pass")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = gatherer._build_index(["core.py", "other.py"])
        result = gatherer._keyword_match(index, "token tracking", max_seeds=5)
        assert "core.py" in result

    def test_matches_by_function_name(self, tmp_path):
        """Function names in structural index are searchable."""
        (tmp_path / "utils.py").write_text("def track_tokens(): pass")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = gatherer._build_index(["utils.py"])
        result = gatherer._keyword_match(index, "token tracking", max_seeds=5)
        assert "utils.py" in result

    def test_matches_by_imports(self, tmp_path):
        """Import paths in structural index are searchable."""
        (tmp_path / "consumer.py").write_text(
            "from token_module import Tracker\nclass X: pass"
        )
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = gatherer._build_index(["consumer.py"])
        result = gatherer._keyword_match(index, "token tracking", max_seeds=5)
        assert "consumer.py" in result

    def test_higher_score_first(self, tmp_path):
        """Files with more keyword matches rank higher."""
        (tmp_path / "best.py").write_text(
            "class TokenUsageTracker:\n"
            "    def track_usage(self): pass"
        )
        (tmp_path / "ok.py").write_text("class TokenCounter: pass")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = gatherer._build_index(["best.py", "ok.py"])
        result = gatherer._keyword_match(
            index, "token usage tracking", max_seeds=5,
        )
        assert result[0] == "best.py"

    def test_respects_max_seeds(self, tmp_path):
        for i in range(10):
            (tmp_path / f"token{i}.py").write_text(
                f"class Token{i}: pass"
            )
        files = [f"token{i}.py" for i in range(10)]
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = gatherer._build_index(files)
        result = gatherer._keyword_match(index, "token usage", max_seeds=3)
        assert len(result) == 3

    def test_excludes_test_files(self, tmp_path):
        """Test files excluded unless task mentions testing."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_token.py").write_text("class TestToken: pass")
        (tmp_path / "token.py").write_text("class Token: pass")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = gatherer._build_index(["tests/test_token.py", "token.py"])
        result = gatherer._keyword_match(index, "token usage", max_seeds=5)
        assert "token.py" in result
        assert "tests/test_token.py" not in result

    def test_includes_test_files_when_task_about_testing(self, tmp_path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_token.py").write_text("class TestToken: pass")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = gatherer._build_index(["tests/test_token.py"])
        result = gatherer._keyword_match(
            index, "fix test for token handling", max_seeds=5,
        )
        assert "tests/test_token.py" in result

    def test_no_matches_returns_empty(self, tmp_path):
        (tmp_path / "app.py").write_text("class Unrelated: pass")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = gatherer._build_index(["app.py"])
        result = gatherer._keyword_match(
            index, "quantum entanglement", max_seeds=5,
        )
        assert result == []

    def test_validates_paths_exist(self, tmp_path):
        """Only returns files that actually exist on disk."""
        (tmp_path / "real.py").write_text("class Token: pass")
        # Build index with a path that doesn't exist on disk
        index = "## real.py\nclasses: Token\n\n## ghost.py\nclasses: Token"
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        result = gatherer._keyword_match(index, "token", max_seeds=5)
        assert "real.py" in result
        assert "ghost.py" not in result

    def test_skips_non_source_files(self, tmp_path):
        """Docs, configs, and markdown files are never seeds."""
        (tmp_path / "app.py").write_text("class Token: pass")
        (tmp_path / "readme.md").write_text("# Token docs")
        (tmp_path / "config.yml").write_text("token: true")
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "guide.py").write_text("class Token: pass")
        gatherer = AgentContextGatherer(config=_make_config(), source_dir=str(tmp_path))
        index = (
            "## app.py\nclasses: Token\n\n"
            "## readme.md\nheadings: Token docs\n\n"
            "## config.yml\nkeys: token\n\n"
            "## docs/guide.py\nclasses: Token"
        )
        result = gatherer._keyword_match(index, "token", max_seeds=10)
        assert "app.py" in result
        assert "readme.md" not in result
        assert "config.yml" not in result
        assert "docs/guide.py" not in result


# ---------------------------------------------------------------------------
# _parse_structural_index
# ---------------------------------------------------------------------------
class TestParseStructuralIndex:
    def test_parses_file_entries(self):
        index = "## a.py\nclasses: Foo\nfunctions: bar()\n\n## b.py\nimports: os"
        lookup = AgentContextGatherer._parse_structural_index(index)
        assert "a.py" in lookup
        assert "classes: Foo" in lookup["a.py"]
        assert "b.py" in lookup
        assert "imports: os" in lookup["b.py"]

    def test_empty_index(self):
        assert AgentContextGatherer._parse_structural_index("") == {}

    def test_file_with_no_info(self):
        index = "## empty.py\n\n## next.py\nclasses: X"
        lookup = AgentContextGatherer._parse_structural_index(index)
        assert lookup["empty.py"] == ""
        assert "classes: X" in lookup["next.py"]


# ---------------------------------------------------------------------------
# Pass 4: _expand_graph
# ---------------------------------------------------------------------------
class TestExpandGraph:
    def test_finds_direct_importers(self, tmp_path):
        (tmp_path / "base.py").write_text("class Base: pass")
        (tmp_path / "impl.py").write_text(
            "from base import Base\nclass Impl(Base): pass"
        )
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        candidates = gatherer._expand_graph(
            seeds=["base.py"], file_paths=["base.py", "impl.py"],
        )
        paths = [p for p, _ in candidates]
        assert "impl.py" in paths

    def test_finds_dependencies(self, tmp_path):
        (tmp_path / "utils.py").write_text("def helper(): pass")
        (tmp_path / "main.py").write_text(
            "from utils import helper\ndef run(): pass"
        )
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        candidates = gatherer._expand_graph(
            seeds=["main.py"], file_paths=["main.py", "utils.py"],
        )
        paths = [p for p, _ in candidates]
        assert "utils.py" in paths

    def test_excludes_seeds(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("from a import x")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        candidates = gatherer._expand_graph(
            seeds=["a.py", "b.py"], file_paths=["a.py", "b.py"],
        )
        assert [p for p, _ in candidates] == []

    def test_transitive_depth_2(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("from a import x")
        (tmp_path / "c.py").write_text("from b import x")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        candidates = gatherer._expand_graph(
            seeds=["a.py"], file_paths=["a.py", "b.py", "c.py"],
        )
        paths = [p for p, _ in candidates]
        assert "b.py" in paths
        assert "c.py" in paths

    def test_respects_max_depth(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("from a import x")
        (tmp_path / "c.py").write_text("from b import x")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        candidates = gatherer._expand_graph(
            seeds=["a.py"], file_paths=["a.py", "b.py", "c.py"],
            max_depth=1,
        )
        paths = [p for p, _ in candidates]
        assert "b.py" in paths
        assert "c.py" not in paths

    def test_connection_descriptions(self, tmp_path):
        (tmp_path / "base.py").write_text("x = 1")
        (tmp_path / "caller.py").write_text("from base import x")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        candidates = gatherer._expand_graph(
            seeds=["base.py"], file_paths=["base.py", "caller.py"],
        )
        path, conns = candidates[0]
        assert path == "caller.py"
        assert any("imports base.py" in c for c in conns)

    def test_bidirectional(self, tmp_path):
        (tmp_path / "dep.py").write_text("x = 1")
        (tmp_path / "seed.py").write_text("from dep import x\ny = 1")
        (tmp_path / "caller.py").write_text("from seed import y")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        candidates = gatherer._expand_graph(
            seeds=["seed.py"],
            file_paths=["dep.py", "seed.py", "caller.py"],
        )
        paths = [p for p, _ in candidates]
        assert "dep.py" in paths
        assert "caller.py" in paths

    def test_no_connections_returns_empty(self, tmp_path):
        (tmp_path / "isolated.py").write_text("x = 1")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        candidates = gatherer._expand_graph(
            seeds=["isolated.py"], file_paths=["isolated.py"],
        )
        assert candidates == []

    def test_no_cycles(self, tmp_path):
        (tmp_path / "a.py").write_text("from b import y\nx = 1")
        (tmp_path / "b.py").write_text("from a import x\ny = 2")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        candidates = gatherer._expand_graph(
            seeds=["a.py"], file_paths=["a.py", "b.py"],
        )
        paths = [p for p, _ in candidates]
        assert "b.py" in paths
        assert len(paths) == 1

    def test_depth_1_before_depth_2(self, tmp_path):
        (tmp_path / "seed.py").write_text("x = 1")
        (tmp_path / "direct.py").write_text("from seed import x\ny = 1")
        (tmp_path / "transitive.py").write_text("from direct import y")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        candidates = gatherer._expand_graph(
            seeds=["seed.py"],
            file_paths=["seed.py", "direct.py", "transitive.py"],
        )
        paths = [p for p, _ in candidates]
        assert paths.index("direct.py") < paths.index("transitive.py")

    def test_packages_with_dotted_imports(self, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "base.py").write_text("class Base: pass")
        (pkg / "impl.py").write_text(
            "from pkg.base import Base\nclass Impl(Base): pass"
        )
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        candidates = gatherer._expand_graph(
            seeds=["pkg/base.py"],
            file_paths=["pkg/__init__.py", "pkg/base.py", "pkg/impl.py"],
        )
        paths = [p for p, _ in candidates]
        assert "pkg/impl.py" in paths


# ---------------------------------------------------------------------------
# Pass 5: _filter_candidates
# ---------------------------------------------------------------------------
class TestFilterCandidates:
    @pytest.mark.asyncio
    async def test_returns_llm_selected_files(self, tmp_path, mock_client):
        (tmp_path / "candidate.py").write_text("code")
        mock_client.generate.return_value = '["candidate.py"]'
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer._filter_candidates(
            mock_client, "m",
            seeds=["seed.py"],
            candidates=[("candidate.py", ["imports seed.py"])],

            job_description="task",
        )
        assert result == ["candidate.py"]

    @pytest.mark.asyncio
    async def test_excludes_seeds(self, tmp_path, mock_client):
        (tmp_path / "seed.py").write_text("code")
        (tmp_path / "other.py").write_text("code")
        mock_client.generate.return_value = '["seed.py", "other.py"]'
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer._filter_candidates(
            mock_client, "m",
            seeds=["seed.py"],
            candidates=[
                ("seed.py", ["self-ref"]),
                ("other.py", ["imports seed.py"]),
            ],

            job_description="task",
        )
        assert "seed.py" not in result
        assert "other.py" in result

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("code")
        (tmp_path / "b.py").write_text("code")
        mock_client.generate.side_effect = RuntimeError("down")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer._filter_candidates(
            mock_client, "m",
            seeds=["seed.py"],
            candidates=[
                ("a.py", ["imports seed.py"]),
                ("b.py", ["imports seed.py"]),
            ],

            job_description="task",
        )
        assert "a.py" in result
        assert "b.py" in result

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_json(self, tmp_path, mock_client):
        (tmp_path / "c.py").write_text("code")
        mock_client.generate.return_value = "not json"
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer._filter_candidates(
            mock_client, "m",
            seeds=["seed.py"],
            candidates=[("c.py", ["imports seed.py"])],

            job_description="task",
        )
        assert "c.py" in result

    @pytest.mark.asyncio
    async def test_validates_paths_exist(self, tmp_path, mock_client):
        (tmp_path / "real.py").write_text("code")
        mock_client.generate.return_value = '["real.py", "ghost.py"]'
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer._filter_candidates(
            mock_client, "m",
            seeds=["seed.py"],
            candidates=[
                ("real.py", ["imports seed.py"]),
                ("ghost.py", ["imports seed.py"]),
            ],

            job_description="task",
        )
        assert result == ["real.py"]



# ---------------------------------------------------------------------------
# Pass 5b: _scan_index
# ---------------------------------------------------------------------------
class TestScanIndex:
    @pytest.mark.asyncio
    async def test_finds_unreachable_files(self, tmp_path, mock_client):
        (tmp_path / "answer.py").write_text("class Answer: pass")
        mock_client.generate.return_value = '["answer.py"]'
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer._scan_index(
            mock_client, "m",
            structural_index="## answer.py\nclasses: Answer",
            job_description="add token usage to answer metadata",
            already_found={"provider.py"},
        )
        assert result == ["answer.py"]

    @pytest.mark.asyncio
    async def test_excludes_already_found(self, tmp_path, mock_client):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("x")
        mock_client.generate.return_value = '["a.py", "b.py"]'
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer._scan_index(
            mock_client, "m",
            structural_index="## a.py\nclasses: A\n\n## b.py\nclasses: B",
            job_description="task",
            already_found={"a.py"},
        )
        assert "a.py" not in result
        assert "b.py" in result

    @pytest.mark.asyncio
    async def test_strips_already_found_from_index(self, tmp_path, mock_client):
        """The structural index sent to LLM should NOT contain already-found files."""
        (tmp_path / "new.py").write_text("x")
        mock_client.generate.return_value = '["new.py"]'
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        await gatherer._scan_index(
            mock_client, "m",
            structural_index="## found.py\nclasses: F\n\n## new.py\nclasses: N",
            job_description="task",
            already_found={"found.py"},
        )
        prompt_sent = mock_client.generate.call_args[1]["messages"][0]["content"]
        assert "## found.py" not in prompt_sent
        assert "## new.py" in prompt_sent

    @pytest.mark.asyncio
    async def test_returns_empty_on_llm_failure(self, tmp_path, mock_client):
        mock_client.generate.side_effect = RuntimeError("down")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer._scan_index(
            mock_client, "m",
            structural_index="## x.py\nclasses: X",
            job_description="task",
            already_found=set(),
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_invalid_json(self, tmp_path, mock_client):
        mock_client.generate.return_value = "not json"
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer._scan_index(
            mock_client, "m",
            structural_index="## x.py\nclasses: X",
            job_description="task",
            already_found=set(),
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_non_source_files(self, tmp_path, mock_client):
        (tmp_path / "code.py").write_text("x")
        (tmp_path / "doc.md").write_text("x")
        mock_client.generate.return_value = '["code.py", "doc.md"]'
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer._scan_index(
            mock_client, "m",
            structural_index="## code.py\nclasses: C\n\n## doc.md\nheadings: D",
            job_description="task",
            already_found=set(),
        )
        assert "code.py" in result
        assert "doc.md" not in result



# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------
class TestSummarizeFile:
    @pytest.mark.asyncio
    async def test_reads_file_and_calls_generate(self, tmp_path, mock_client):
        (tmp_path / "app.py").write_text("def main(): pass")
        mock_client.generate.return_value = "summary"
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer._summarize_file(
            mock_client, "m", "app.py", "task"
        )
        assert result == "summary"

    @pytest.mark.asyncio
    async def test_returns_none_on_missing_file(self, tmp_path, mock_client):
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer._summarize_file(
            mock_client, "m", "ghost.py", "task"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_file(self, tmp_path, mock_client):
        (tmp_path / "blank.py").write_text("")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer._summarize_file(
            mock_client, "m", "blank.py", "task"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_failure(self, tmp_path, mock_client):
        (tmp_path / "app.py").write_text("code")
        mock_client.generate.side_effect = RuntimeError("fail")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer._summarize_file(
            mock_client, "m", "app.py", "task"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_respects_max_file_bytes(self, tmp_path, mock_client):
        big = "x" * 100
        (tmp_path / "big.py").write_text(big)
        gatherer = AgentContextGatherer(
            config=_make_config(max_file_bytes=10), source_dir=str(tmp_path)
        )
        mock_client.generate.return_value = "summary"
        await gatherer._summarize_file(mock_client, "m", "big.py", "task")
        prompt = mock_client.generate.call_args[1]["messages"][0]["content"]
        assert big not in prompt

    @pytest.mark.asyncio
    async def test_rejects_path_traversal(self, tmp_path, mock_client):
        secret = tmp_path.parent / "secret.py"
        secret.write_text("secret")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer._summarize_file(
            mock_client, "m", "../secret.py", "task"
        )
        assert result is None


# ---------------------------------------------------------------------------
# Synthesize
# ---------------------------------------------------------------------------
class TestSynthesize:
    @pytest.mark.asyncio
    async def test_passes_summaries_to_generate(self, mock_client):
        mock_client.generate.return_value = "## Overview\nDone."
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir="/fake"
        )
        result = await gatherer._synthesize(
            mock_client, "m", ["### a.py\nSummary A"], "task"
        )
        assert "## Overview" in result

    @pytest.mark.asyncio
    async def test_concatenation_fallback_on_failure(self, mock_client):
        mock_client.generate.side_effect = RuntimeError("fail")
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir="/fake"
        )
        result = await gatherer._synthesize(
            mock_client, "m", ["### a.py\nSummary A"], "task"
        )
        assert "Summary A" in result
        assert "## File Summaries" in result


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
        """Keyword match finds files, no graph neighbors, scan+summarize+synth."""
        (tmp_path / "token_tracker.py").write_text(
            "class TokenTracker:\n    def track(self): pass"
        )
        (tmp_path / "utils.py").write_text("def helper(): pass")

        mock_client.generate.side_effect = [
            # no filter call — isolated files have no graph neighbors
            '[]',                               # scan — finds nothing extra
            "**Purpose:** Tracks tokens.",      # summarize token_tracker
            "## Overview\nFull doc.",            # synthesize
        ]

        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer.gather(mock_client, "token tracking")
        assert "## Overview" in result["synthesized"]
        assert "Tracks tokens" in result["raw_summaries"]
        # 3 LLM calls: scan + summarize + synthesize
        assert mock_client.generate.call_count == 3
        # agent_files metadata present
        af = result["agent_files"]
        assert "token_tracker.py" in af["seeds"]
        assert af["selected"] == af["seeds"] + af["filtered"] + af["scanned"]

    @pytest.mark.asyncio
    async def test_graph_expansion_adds_callers(self, tmp_path, mock_client):
        """Graph expansion finds importers and filter includes them."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        # seed: matches keyword "widget" via class name (path is generic)
        (pkg / "core.py").write_text("class WidgetFactory: pass")
        # NOT a seed (no keyword match), but imports the seed
        (pkg / "runner.py").write_text(
            "from pkg.core import WidgetFactory\n"
            "class Runner:\n"
            "    def execute(self): pass"
        )

        mock_client.generate.side_effect = [
            '["pkg/runner.py"]',                # filter — selects runner
            # scan skipped — all indexed files already found
            "**Purpose:** Widget core.",         # summarize core
            "**Purpose:** Runner.",              # summarize runner
            "## Overview\nDone.",                # synthesize
        ]

        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer.gather(mock_client, "build widget system")
        assert result["synthesized"] != ""
        # 4 calls: filter + 2 summarize + synthesize (scan skipped — empty remaining index)
        assert mock_client.generate.call_count == 4
        # agent_files tracks all passes
        af = result["agent_files"]
        assert "pkg/core.py" in af["seeds"]
        assert "pkg/runner.py" in af["graph_candidates"]
        assert "pkg/runner.py" in af["filtered"]
        assert set(af["selected"]) == {"pkg/core.py", "pkg/runner.py"}

    @pytest.mark.asyncio
    async def test_empty_dir_returns_empty(self, tmp_path, mock_client):
        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer.gather(mock_client, "task")
        assert result == {"synthesized": "", "raw_summaries": ""}

    @pytest.mark.asyncio
    async def test_all_summaries_fail_returns_empty(self, tmp_path, mock_client):
        (tmp_path / "token.py").write_text("class Token: pass")

        mock_client.generate.side_effect = [
            # scan skipped — only file is already in seeds, remaining index empty
            RuntimeError("LLM down"),          # summarize fails
        ]

        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer.gather(mock_client, "token tracking")
        assert result == {"synthesized": "", "raw_summaries": ""}

    @pytest.mark.asyncio
    async def test_uses_agent_model_config(self, tmp_path, mock_client):
        (tmp_path / "token.py").write_text("class Token: pass")

        mock_client.generate.side_effect = [
            '[]',                              # scan
            "summary",                         # summarize
            "## Overview\nDone.",              # synthesize
        ]

        gatherer = AgentContextGatherer(
            config=_make_config(agent_model="special-model"),
            source_dir=str(tmp_path),
        )
        await gatherer.gather(mock_client, "token tracking")
        for call in mock_client.generate.call_args_list:
            assert call[1]["model"] == "special-model"

    @pytest.mark.asyncio
    async def test_falls_back_to_client_model(self, tmp_path, mock_client):
        (tmp_path / "token.py").write_text("class Token: pass")

        mock_client.generate.side_effect = [
            '[]',                              # scan
            "summary",                         # summarize
            "## Overview\nDone.",              # synthesize
        ]

        gatherer = AgentContextGatherer(
            config=_make_config(agent_model=None),
            source_dir=str(tmp_path),
        )
        await gatherer.gather(mock_client, "token tracking")
        for call in mock_client.generate.call_args_list:
            assert call[1]["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_total_failure_returns_empty(self, tmp_path, mock_client):
        (tmp_path / "token.py").write_text("class Token: pass")
        mock_client.generate.side_effect = RuntimeError("total failure")

        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer.gather(mock_client, "token tracking")
        assert result == {"synthesized": "", "raw_summaries": ""}

    @pytest.mark.asyncio
    async def test_no_keyword_matches_returns_empty(self, tmp_path, mock_client):
        """If no files match any keywords, returns empty."""
        (tmp_path / "app.py").write_text("class Unrelated: pass")

        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        result = await gatherer.gather(
            mock_client, "quantum entanglement simulation"
        )
        assert result == {"synthesized": "", "raw_summaries": ""}
        mock_client.generate.assert_not_called()


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------
class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_all_phases_reported(self, tmp_path, mock_client):
        """With graph neighbors, all phases including filter are reported."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "core.py").write_text("class WidgetFactory: pass")
        (pkg / "runner.py").write_text(
            "from pkg.core import WidgetFactory\n"
            "class Runner:\n"
            "    def execute(self): pass"
        )

        # Add an extra file so the scan has something to work with
        (pkg / "data.py").write_text("class Data: pass")

        mock_client.generate.side_effect = [
            '["pkg/runner.py"]',               # filter candidates
            '[]',                              # scan — nothing extra
            "summary core",                    # summarize core
            "summary runner",                  # summarize runner
            "## Overview\nDone.",              # synthesize
        ]

        phases = []

        def callback(progress, phase):
            phases.append(phase)

        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        await gatherer.gather(
            mock_client, "build widget system", progress_callback=callback,
        )

        assert "agent:mapping" in phases
        assert "agent:indexing" in phases
        assert "agent:matching" in phases
        assert "agent:expanding_graph" in phases
        assert "agent:filtering" in phases
        assert "agent:scanning" in phases
        assert any(p.startswith("agent:summarizing:") for p in phases)
        assert "agent:synthesizing" in phases

    @pytest.mark.asyncio
    async def test_async_callback_awaited(self, tmp_path, mock_client):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "core.py").write_text("class WidgetFactory: pass")
        (pkg / "runner.py").write_text(
            "from pkg.core import WidgetFactory\n"
            "class Runner:\n"
            "    def execute(self): pass"
        )

        # Add an extra file so the scan has something to work with
        (pkg / "data.py").write_text("class Data: pass")

        mock_client.generate.side_effect = [
            '["pkg/runner.py"]',               # filter
            '[]',                              # scan
            "summary core",                    # summarize core
            "summary runner",                  # summarize runner
            "## Overview\nDone.",              # synthesize
        ]

        phases = []

        async def async_callback(progress, phase):
            phases.append(phase)

        gatherer = AgentContextGatherer(
            config=_make_config(), source_dir=str(tmp_path)
        )
        await gatherer.gather(
            mock_client, "build widget system",
            progress_callback=async_callback,
        )
        # mapping, indexing, matching, expanding, filtering, scanning, 2x summarizing, synthesizing
        assert len(phases) >= 8


# ---------------------------------------------------------------------------
# _parse_file_list
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
        result = AgentContextGatherer._parse_file_list("just some text")
        assert result is None

    def test_filters_non_string_items(self):
        result = AgentContextGatherer._parse_file_list('["a.py", 123, null]')
        assert result == ["a.py"]
