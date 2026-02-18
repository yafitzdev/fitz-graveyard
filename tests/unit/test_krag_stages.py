# tests/unit/test_krag_stages.py
"""Integration tests for KRAG context in all 5 pipeline stages."""

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fitz_planner_mcp.config.schema import FitzPlannerConfig, KragConfig
from fitz_planner_mcp.planning.pipeline.stages import (
    ArchitectureStage,
    ContextStage,
    DesignStage,
    RiskStage,
    RoadmapStage,
    create_stages,
)


@dataclass
class MockLLMResponse:
    """Mock LLM response for testing."""

    content: str


class TestContextStageKragIntegration:
    """Test ContextStage KRAG client creation and context injection."""

    def test_context_stage_no_config_no_krag(self):
        """ContextStage without config works normally (no _krag_client in prior_outputs)."""
        stage = ContextStage()
        prior_outputs = {}

        messages = stage.build_prompt("Build a blog platform", prior_outputs)

        # Should work without KRAG
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Build a blog platform" in messages[0]["content"]
        # No KragClient created
        assert "_krag_client" not in prior_outputs

    def test_context_stage_krag_disabled(self):
        """ContextStage with config.krag.enabled=False creates no client."""
        config = FitzPlannerConfig()
        config.krag.enabled = False
        stage = ContextStage(config=config, source_dir="./")
        prior_outputs = {}

        with patch("fitz_planner_mcp.planning.pipeline.stages.context.KragClient") as MockKrag:
            messages = stage.build_prompt("Build a blog platform", prior_outputs)

            # KragClient.from_config should be called
            MockKrag.from_config.assert_called_once()

            # But client should return empty context (enabled=False)
            # prior_outputs should have the client
            assert "_krag_client" in prior_outputs

    def test_context_stage_creates_krag_client(self):
        """ContextStage with config creates KragClient in prior_outputs['_krag_client']."""
        config = FitzPlannerConfig()
        config.krag.enabled = True
        stage = ContextStage(config=config, source_dir="./test")
        prior_outputs = {}

        with patch("fitz_planner_mcp.planning.pipeline.stages.context.KragClient") as MockKrag:
            # Mock the from_config class method
            mock_client = MagicMock()
            mock_client.multi_query.return_value = "## Codebase Context\n\nTest context"
            MockKrag.from_config.return_value = mock_client

            messages = stage.build_prompt("Build a blog platform", prior_outputs)

            # Verify KragClient was created
            MockKrag.from_config.assert_called_once_with(config.krag, "./test")

            # Verify client was stored in prior_outputs
            assert "_krag_client" in prior_outputs
            assert prior_outputs["_krag_client"] == mock_client

            # Verify multi_query was called
            mock_client.multi_query.assert_called_once()
            queries = mock_client.multi_query.call_args[0][0]
            assert len(queries) == 2
            assert "architecture overview" in queries[0]

    def test_context_stage_injects_krag_context(self):
        """With mocked KragClient, verify {krag_context} replaced in prompt."""
        config = FitzPlannerConfig()
        config.krag.enabled = True
        stage = ContextStage(config=config, source_dir="./test")
        prior_outputs = {}

        with patch("fitz_planner_mcp.planning.pipeline.stages.context.KragClient") as MockKrag:
            mock_client = MagicMock()
            mock_client.multi_query.return_value = "## Codebase Context\n\n### Architecture\nDjango MVC pattern"
            MockKrag.from_config.return_value = mock_client

            messages = stage.build_prompt("Build a blog platform", prior_outputs)

            # Verify KRAG context appears in prompt
            prompt_content = messages[0]["content"]
            assert "## Codebase Context" in prompt_content
            assert "Django MVC pattern" in prompt_content


class TestArchitectureStageKragIntegration:
    """Test ArchitectureStage uses KRAG from prior_outputs."""

    def test_architecture_stage_uses_krag(self):
        """Architecture stage reads _krag_client from prior_outputs and injects context."""
        stage = ArchitectureStage()

        # Simulate prior_outputs from ContextStage
        mock_client = MagicMock()
        mock_client.multi_query.return_value = "## Codebase Context\n\n### Patterns\nRepository pattern"
        prior_outputs = {"_krag_client": mock_client}

        messages = stage.build_prompt("Build API", prior_outputs)

        # Verify multi_query was called with architecture-specific queries
        mock_client.multi_query.assert_called_once()
        queries = mock_client.multi_query.call_args[0][0]
        assert len(queries) == 2
        assert "design patterns" in queries[0]
        assert "architectural decisions" in queries[1]

        # Verify KRAG context appears in prompt
        prompt_content = messages[0]["content"]
        assert "Repository pattern" in prompt_content


class TestDesignStageKragIntegration:
    """Test DesignStage uses KRAG from prior_outputs."""

    def test_design_stage_uses_krag(self):
        """Design stage reads _krag_client from prior_outputs and injects context."""
        stage = DesignStage()

        mock_client = MagicMock()
        mock_client.multi_query.return_value = "## Codebase Context\n\n### Models\nUser, Post, Comment"
        prior_outputs = {"_krag_client": mock_client}

        messages = stage.build_prompt("Build blog", prior_outputs)

        # Verify multi_query was called with design-specific queries
        mock_client.multi_query.assert_called_once()
        queries = mock_client.multi_query.call_args[0][0]
        assert len(queries) == 2
        assert "classes" in queries[0] or "data models" in queries[0]
        assert "interfaces" in queries[1] or "contracts" in queries[1]

        # Verify KRAG context appears in prompt
        prompt_content = messages[0]["content"]
        assert "User, Post, Comment" in prompt_content


class TestRoadmapStageKragIntegration:
    """Test RoadmapStage uses KRAG from prior_outputs."""

    def test_roadmap_stage_uses_krag(self):
        """Roadmap stage reads _krag_client from prior_outputs and injects context."""
        stage = RoadmapStage()

        mock_client = MagicMock()
        mock_client.multi_query.return_value = "## Codebase Context\n\n### Dependencies\nauth depends on users"
        prior_outputs = {"_krag_client": mock_client}

        messages = stage.build_prompt("Build features", prior_outputs)

        # Verify multi_query was called with roadmap-specific queries
        mock_client.multi_query.assert_called_once()
        queries = mock_client.multi_query.call_args[0][0]
        assert len(queries) == 2
        assert "dependencies" in queries[0]
        assert "TODOs" in queries[1] or "incomplete" in queries[1]

        # Verify KRAG context appears in prompt
        prompt_content = messages[0]["content"]
        assert "auth depends on users" in prompt_content


class TestRiskStageKragIntegration:
    """Test RiskStage uses KRAG from prior_outputs."""

    def test_risk_stage_uses_krag(self):
        """Risk stage reads _krag_client from prior_outputs and injects context."""
        stage = RiskStage()

        mock_client = MagicMock()
        mock_client.multi_query.return_value = "## Codebase Context\n\n### Security\nJWT auth, XSS prevention"
        prior_outputs = {"_krag_client": mock_client}

        messages = stage.build_prompt("Build system", prior_outputs)

        # Verify multi_query was called with risk-specific queries
        mock_client.multi_query.assert_called_once()
        queries = mock_client.multi_query.call_args[0][0]
        assert len(queries) == 2
        assert "security" in queries[0] or "error handling" in queries[0]
        assert "dependencies" in queries[1]

        # Verify KRAG context appears in prompt
        prompt_content = messages[0]["content"]
        assert "JWT auth" in prompt_content


class TestStageWithoutKragClient:
    """Test stages work when _krag_client not in prior_outputs (backward compat)."""

    @pytest.mark.parametrize(
        "stage_class",
        [ArchitectureStage, DesignStage, RoadmapStage, RiskStage],
    )
    def test_stage_without_krag_client_works(self, stage_class):
        """Any stage works when _krag_client not in prior_outputs."""
        stage = stage_class()
        prior_outputs = {}  # No _krag_client

        # Should not raise, should build prompt normally
        messages = stage.build_prompt("Build something", prior_outputs)

        assert len(messages) >= 1
        assert messages[0]["role"] == "user"
        # krag_context should be empty string (not break prompt)


class TestKragContextInPrompts:
    """Test that KRAG markdown actually appears in the built prompt messages."""

    def test_krag_context_appears_in_prompt(self):
        """Verify that KRAG markdown actually appears in the built prompt messages."""
        config = FitzPlannerConfig()
        config.krag.enabled = True
        stage = ContextStage(config=config, source_dir="./")
        prior_outputs = {}

        with patch("fitz_planner_mcp.planning.pipeline.stages.context.KragClient") as MockKrag:
            mock_client = MagicMock()
            # Return substantial context
            mock_client.multi_query.return_value = """## Codebase Context

### What is the project architecture overview and main modules?

The project uses a layered architecture with:
- Core domain models
- Service layer for business logic
- API layer with REST endpoints

### What key interfaces and integration points exist in the codebase?

Key interfaces:
- IUserRepository (data access)
- IAuthService (authentication)
- IEmailService (notifications)
"""
            MockKrag.from_config.return_value = mock_client

            messages = stage.build_prompt("Build a user management system", prior_outputs)

            prompt_content = messages[0]["content"]

            # Verify all KRAG content appears
            assert "## Codebase Context" in prompt_content
            assert "layered architecture" in prompt_content
            assert "IUserRepository" in prompt_content
            assert "IAuthService" in prompt_content


class TestEmptyKragContext:
    """Test that empty KRAG context doesn't break prompt formatting."""

    def test_empty_krag_context_harmless(self):
        """When KRAG returns "", prompt still valid (no broken formatting)."""
        config = FitzPlannerConfig()
        config.krag.enabled = True
        stage = ContextStage(config=config, source_dir="./")
        prior_outputs = {}

        with patch("fitz_planner_mcp.planning.pipeline.stages.context.KragClient") as MockKrag:
            mock_client = MagicMock()
            mock_client.multi_query.return_value = ""  # Empty context
            MockKrag.from_config.return_value = mock_client

            messages = stage.build_prompt("Build something", prior_outputs)

            # Should still build valid prompt
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            prompt_content = messages[0]["content"]

            # Should contain user's description
            assert "Build something" in prompt_content

            # Should not have broken placeholders
            assert "{krag_context}" not in prompt_content


class TestCreateStagesFactory:
    """Test create_stages() factory function."""

    def test_create_stages_no_config(self):
        """create_stages() without config returns 5 stages (no KRAG)."""
        stages = create_stages()

        assert len(stages) == 5
        assert stages[0].name == "context"
        assert stages[1].name == "architecture"
        assert stages[2].name == "design"
        assert stages[3].name == "roadmap"
        assert stages[4].name == "risk"

    def test_create_stages_with_config(self):
        """create_stages() with config passes config to ContextStage."""
        config = FitzPlannerConfig()
        config.krag.enabled = True

        stages = create_stages(config=config, source_dir="./test")

        # Verify ContextStage has config
        context_stage = stages[0]
        assert isinstance(context_stage, ContextStage)
        # Check that config was passed (internal attribute check)
        assert hasattr(context_stage, "_config")
        assert context_stage._config == config

    def test_create_stages_equivalent_to_default_stages(self):
        """create_stages() without args produces equivalent stages to DEFAULT_STAGES."""
        from fitz_planner_mcp.planning.pipeline.stages import DEFAULT_STAGES

        stages = create_stages()

        # Same count
        assert len(stages) == len(DEFAULT_STAGES)

        # Same names and progress ranges
        for i, (created, default) in enumerate(zip(stages, DEFAULT_STAGES)):
            assert created.name == default.name, f"Stage {i} name mismatch"
            assert created.progress_range == default.progress_range, f"Stage {i} progress range mismatch"


class TestKragClientSharing:
    """Test that KragClient is shared across stages via prior_outputs."""

    @pytest.mark.asyncio
    async def test_krag_client_not_recreated(self):
        """Verify KragClient created once by ContextStage, reused by other stages."""
        config = FitzPlannerConfig()
        config.krag.enabled = True

        stages = create_stages(config=config, source_dir="./")
        context_stage = stages[0]
        architecture_stage = stages[1]

        prior_outputs = {}

        with patch("fitz_planner_mcp.planning.pipeline.stages.context.KragClient") as MockKrag:
            mock_client = MagicMock()
            mock_client.multi_query.return_value = "## Codebase Context\n\nTest"
            MockKrag.from_config.return_value = mock_client

            # ContextStage creates client
            context_stage.build_prompt("Build something", prior_outputs)

            # Verify client was created
            MockKrag.from_config.assert_called_once()
            assert "_krag_client" in prior_outputs

            # ArchitectureStage reuses client (no new creation)
            architecture_stage.build_prompt("Build something", prior_outputs)

            # KragClient.from_config should still be called only once
            MockKrag.from_config.assert_called_once()

            # But multi_query should be called twice (once per stage)
            assert mock_client.multi_query.call_count == 2
