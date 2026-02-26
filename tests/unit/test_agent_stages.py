# tests/unit/test_agent_stages.py
"""Tests for pipeline stages using gathered context instead of KRAG."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from fitz_graveyard.planning.pipeline.stages import (
    ArchitectureDesignStage,
    ContextStage,
    RoadmapRiskStage,
    create_stages,
)


GATHERED_CONTEXT = "## Architecture\nKey components: JobStore, BackgroundWorker."


class TestCreateStages:
    def test_create_stages_no_args(self):
        stages = create_stages()
        assert len(stages) == 3

    def test_stage_names(self):
        stages = create_stages()
        names = [s.name for s in stages]
        assert names == ["context", "architecture_design", "roadmap_risk"]

    def test_progress_ranges_sequential(self):
        stages = create_stages()
        for stage in stages:
            start, end = stage.progress_range
            assert start < end
            assert 0.0 <= start <= 1.0
            assert 0.0 <= end <= 1.0

    def test_context_stage_no_init_params(self):
        # ContextStage should take no args now
        stage = ContextStage()
        assert stage.name == "context"


class TestContextStageGatheredContext:
    def test_empty_prior_outputs_no_crash(self):
        stage = ContextStage()
        # Should not raise even with empty prior_outputs
        result = stage._get_gathered_context({})
        assert result == ""

    def test_uses_gathered_context(self):
        stage = ContextStage()
        result = stage._get_gathered_context(
            {"_gathered_context": GATHERED_CONTEXT}
        )
        assert result == GATHERED_CONTEXT

    def test_prompt_contains_gathered_context(self):
        stage = ContextStage()
        prior = {"_gathered_context": GATHERED_CONTEXT}
        messages = stage.build_prompt("add a feature", prior)
        assert len(messages) == 2
        # User message is at index 1 (index 0 is system prompt)
        assert GATHERED_CONTEXT in messages[1]["content"]

    def test_prompt_works_without_gathered_context(self):
        stage = ContextStage()
        messages = stage.build_prompt("add a feature", {})
        assert len(messages) == 2
        # Should not contain a literal {krag_context} placeholder
        assert "{krag_context}" not in messages[1]["content"]


class TestArchitectureDesignStageGatheredContext:
    def test_uses_gathered_context(self):
        stage = ArchitectureDesignStage()
        prior = {"_gathered_context": GATHERED_CONTEXT}
        messages = stage.build_prompt("add a feature", prior)
        assert len(messages) == 2
        # User message is at index 1 (index 0 is system prompt)
        assert GATHERED_CONTEXT in messages[1]["content"]

    def test_no_context_no_crash(self):
        stage = ArchitectureDesignStage()
        messages = stage.build_prompt("add a feature", {})
        assert len(messages) == 2
        assert "{krag_context}" not in messages[1]["content"]


class TestRoadmapRiskStageGatheredContext:
    def test_uses_gathered_context(self):
        stage = RoadmapRiskStage()
        prior = {"_gathered_context": GATHERED_CONTEXT}
        messages = stage.build_prompt("add a feature", prior)
        # User message is at index 1 (index 0 is system prompt)
        assert GATHERED_CONTEXT in messages[1]["content"]

    def test_no_context_no_crash(self):
        stage = RoadmapRiskStage()
        messages = stage.build_prompt("add a feature", {})
        assert "{krag_context}" not in messages[1]["content"]


class TestNoKragImports:
    def test_context_stage_has_no_krag_dependency(self):
        """ContextStage should not reference KragClient."""
        import inspect
        import fitz_graveyard.planning.pipeline.stages.context as mod
        source = inspect.getsource(mod)
        assert "KragClient" not in source
        assert "_get_krag_context" not in source

    def test_base_has_no_krag_dependency(self):
        import inspect
        import fitz_graveyard.planning.pipeline.stages.base as mod
        source = inspect.getsource(mod)
        assert "KragClient" not in source
        assert "_get_krag_context" not in source
