# tests/unit/test_call_graph.py
"""Tests for call graph extraction."""
import pytest
from fitz_graveyard.planning.pipeline.call_graph import (
    extract_call_graph,
    _extract_task_keywords,
    _match_keywords_to_files,
    CallGraph,
    CallGraphNode,
)


class TestKeywordExtraction:
    def test_basic_keywords(self):
        kws = _extract_task_keywords("Add streaming support for queries")
        assert "streaming" in kws
        assert "support" in kws
        assert "queries" in kws

    def test_stop_words_removed(self):
        kws = _extract_task_keywords("Add the streaming to a query")
        assert "the" not in kws
        assert "add" not in kws
        assert "streaming" in kws

    def test_camelcase_split(self):
        kws = _extract_task_keywords("Fix StreamingAnswer class")
        assert "streaming" in kws
        assert "answer" in kws

    def test_snake_case_split(self):
        kws = _extract_task_keywords("Fix token_tracking module")
        assert "token" in kws
        assert "tracking" in kws

    def test_quoted_strings(self):
        kws = _extract_task_keywords('Fix "synthesizer.py" issue')
        assert "synthesizer" in kws

    def test_short_tokens_removed(self):
        kws = _extract_task_keywords("Go to py")
        assert "go" not in kws
        assert "py" not in kws

    def test_returns_sorted_unique(self):
        kws = _extract_task_keywords("streaming streaming streaming query query")
        assert kws == sorted(set(kws))


class TestKeywordMatching:
    def test_matches_class_names(self):
        index = "## engine.py\nclasses: FitzKragEngine\nfunctions: answer\n"
        matches = _match_keywords_to_files(["engine", "answer"], index)
        assert "engine.py" in matches

    def test_path_match_stronger(self):
        index = (
            "## streaming.py\nfunctions: process\n\n"
            "## other.py\nfunctions: streaming_helper\n"
        )
        matches = _match_keywords_to_files(["streaming"], index)
        assert matches[0] == "streaming.py"

    def test_no_matches(self):
        index = "## unrelated.py\nclasses: Foo\n"
        matches = _match_keywords_to_files(["streaming"], index)
        assert matches == []

    def test_multiple_keywords_score_higher(self):
        index = (
            "## both.py\nclasses: Streamer\nfunctions: query\n\n"
            "## one.py\nclasses: Streamer\n"
        )
        matches = _match_keywords_to_files(["streamer", "query"], index)
        assert matches[0] == "both.py"


class TestCallGraphExtraction:
    def test_basic_graph(self):
        index = "## a.py\nclasses: Engine\n\n## b.py\nfunctions: generate\n"
        forward = {"a.py": {"b.py"}}
        entries = {"a.py": "classes: Engine", "b.py": "functions: generate"}
        graph = extract_call_graph("engine", index, forward, entries)
        assert len(graph.nodes) >= 1
        assert "a.py" in graph.entry_points

    def test_empty_on_no_match(self):
        graph = extract_call_graph("xyz", "", {}, {})
        assert graph.nodes == []
        assert graph.edges == []

    def test_bidirectional_traversal(self):
        index = "## a.py\nclasses: Streamer\n\n## b.py\nfunctions: call\n\n## c.py\nfunctions: use\n"
        forward = {"b.py": {"a.py"}, "c.py": {"b.py"}}
        entries = {
            "a.py": "classes: Streamer",
            "b.py": "functions: call",
            "c.py": "functions: use",
        }
        graph = extract_call_graph("streamer", index, forward, entries)
        paths = [n.file_path for n in graph.nodes]
        assert "a.py" in paths
        assert "b.py" in paths  # reverse edge from a.py

    def test_max_depth_limits(self):
        index = "## a.py\nclasses: Alpha\n\n## b.py\n\n## c.py\n\n## d.py\n\n## e.py\n"
        forward = {"a.py": {"b.py"}, "b.py": {"c.py"}, "c.py": {"d.py"}, "d.py": {"e.py"}}
        entries = {f: "" for f in ["a.py", "b.py", "c.py", "d.py", "e.py"]}
        graph = extract_call_graph("alpha", index, forward, entries, max_depth=2)
        paths = [n.file_path for n in graph.nodes]
        assert "a.py" in paths  # depth 0
        assert "b.py" in paths  # depth 1
        assert "c.py" in paths  # depth 2
        assert "d.py" not in paths  # depth 3 = beyond limit

    def test_format_for_prompt(self):
        graph = CallGraph(
            nodes=[
                CallGraphNode("a.py", ["X"], "doc a", 0),
                CallGraphNode("b.py", ["Y"], "doc b", 1),
            ],
            edges=[("a.py", "b.py")],
            entry_points=["a.py"],
            max_depth=1,
        )
        text = graph.format_for_prompt()
        assert "a.py" in text
        assert "b.py" in text
        assert "a.py -> b.py" in text

    def test_segment_for_files(self):
        graph = CallGraph(
            nodes=[
                CallGraphNode("a.py", ["X"], "doc", 0),
                CallGraphNode("b.py", ["Y"], "doc", 1),
                CallGraphNode("c.py", ["Z"], "doc", 2),
            ],
            edges=[("a.py", "b.py"), ("b.py", "c.py")],
            entry_points=["a.py"],
            max_depth=2,
        )
        segment = graph.segment_for_files(["a.py", "b.py"])
        assert len(segment.nodes) == 2
        assert len(segment.edges) == 1

    def test_circular_imports_handled(self):
        index = "## a.py\nclasses: Foo\n\n## b.py\nclasses: Bar\n"
        forward = {"a.py": {"b.py"}, "b.py": {"a.py"}}
        entries = {"a.py": "classes: Foo", "b.py": "classes: Bar"}
        graph = extract_call_graph("foo", index, forward, entries)
        paths = [n.file_path for n in graph.nodes]
        assert "a.py" in paths
        assert "b.py" in paths
