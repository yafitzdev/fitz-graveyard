# tests/unit/test_decision_resolution.py
"""Tests for decision resolution topological sorting."""
from fitz_graveyard.planning.pipeline.stages.decision_resolution import (
    _topological_sort,
)


def test_simple_chain():
    decisions = [
        {"id": "d1", "depends_on": []},
        {"id": "d2", "depends_on": ["d1"]},
        {"id": "d3", "depends_on": ["d2"]},
    ]
    result = _topological_sort(decisions)
    ids = [d["id"] for d in result]
    assert ids == ["d1", "d2", "d3"]


def test_diamond():
    decisions = [
        {"id": "d1", "depends_on": []},
        {"id": "d2", "depends_on": ["d1"]},
        {"id": "d3", "depends_on": ["d1"]},
        {"id": "d4", "depends_on": ["d2", "d3"]},
    ]
    result = _topological_sort(decisions)
    ids = [d["id"] for d in result]
    assert ids.index("d1") < ids.index("d2")
    assert ids.index("d1") < ids.index("d3")
    assert ids.index("d2") < ids.index("d4")
    assert ids.index("d3") < ids.index("d4")


def test_cycle_handled():
    decisions = [
        {"id": "d1", "depends_on": ["d2"]},
        {"id": "d2", "depends_on": ["d1"]},
    ]
    result = _topological_sort(decisions)
    assert len(result) == 2  # Both present, cycle broken


def test_no_dependencies():
    decisions = [
        {"id": "d1", "depends_on": []},
        {"id": "d2", "depends_on": []},
    ]
    result = _topological_sort(decisions)
    assert len(result) == 2


def test_missing_dependency_ignored():
    decisions = [
        {"id": "d1", "depends_on": ["nonexistent"]},
        {"id": "d2", "depends_on": []},
    ]
    result = _topological_sort(decisions)
    assert len(result) == 2


def test_single_decision():
    decisions = [{"id": "d1", "depends_on": []}]
    result = _topological_sort(decisions)
    assert len(result) == 1
    assert result[0]["id"] == "d1"


def test_complex_dag():
    decisions = [
        {"id": "d1", "depends_on": []},
        {"id": "d2", "depends_on": ["d1"]},
        {"id": "d3", "depends_on": ["d1"]},
        {"id": "d4", "depends_on": ["d2"]},
        {"id": "d5", "depends_on": ["d3", "d4"]},
    ]
    result = _topological_sort(decisions)
    ids = [d["id"] for d in result]
    assert ids.index("d1") < ids.index("d2")
    assert ids.index("d1") < ids.index("d3")
    assert ids.index("d2") < ids.index("d4")
    assert ids.index("d4") < ids.index("d5")
    assert ids.index("d3") < ids.index("d5")
