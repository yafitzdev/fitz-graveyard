# tests/unit/test_decision_schemas.py
"""Tests for decision schemas."""
from fitz_graveyard.planning.schemas.decisions import (
    AtomicDecision,
    DecisionResolution,
    DecisionDecompositionOutput,
    DecisionResolutionOutput,
)


def test_atomic_decision_defaults():
    d = AtomicDecision(id="d1", question="What?")
    assert d.relevant_files == []
    assert d.depends_on == []
    assert d.category == "technical"


def test_atomic_decision_full():
    d = AtomicDecision(
        id="d1",
        question="What pattern?",
        relevant_files=["a.py"],
        depends_on=["d0"],
        category="pattern",
    )
    assert d.id == "d1"
    assert d.relevant_files == ["a.py"]
    assert d.depends_on == ["d0"]
    assert d.category == "pattern"


def test_atomic_decision_extra_ignored():
    d = AtomicDecision(id="d1", question="What?", unknown_field="ignored")
    assert d.id == "d1"


def test_decision_resolution_round_trip():
    r = DecisionResolution(
        decision_id="d1",
        decision="Use X",
        reasoning="Because Y",
        evidence=["file.py: method()"],
        constraints_for_downstream=["Must use X"],
    )
    dumped = r.model_dump()
    restored = DecisionResolution(**dumped)
    assert restored.decision_id == "d1"
    assert restored.constraints_for_downstream == ["Must use X"]


def test_decision_resolution_defaults():
    r = DecisionResolution(
        decision_id="d1",
        decision="Use X",
        reasoning="Because Y",
    )
    assert r.evidence == []
    assert r.constraints_for_downstream == []


def test_decomposition_output_parses():
    data = {
        "decisions": [
            {"id": "d1", "question": "What pattern?", "relevant_files": ["a.py"]},
            {"id": "d2", "question": "What interface?", "depends_on": ["d1"]},
        ]
    }
    output = DecisionDecompositionOutput(**data)
    assert len(output.decisions) == 2
    assert output.decisions[1].depends_on == ["d1"]


def test_decomposition_output_empty():
    output = DecisionDecompositionOutput()
    assert output.decisions == []


def test_resolution_output_parses():
    data = {
        "resolutions": [
            {
                "decision_id": "d1",
                "decision": "Use X",
                "reasoning": "Because",
            },
        ]
    }
    output = DecisionResolutionOutput(**data)
    assert len(output.resolutions) == 1
    assert output.resolutions[0].decision_id == "d1"
