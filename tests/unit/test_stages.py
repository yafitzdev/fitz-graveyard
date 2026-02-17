# tests/unit/test_stages.py
"""Unit tests for all 5 pipeline stages."""

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from fitz_planner_mcp.planning.pipeline.stages import (
    DEFAULT_STAGES,
    ArchitectureStage,
    ContextStage,
    DesignStage,
    RiskStage,
    RoadmapStage,
    extract_json,
)


@dataclass
class MockLLMResponse:
    """Mock LLM response for testing."""

    content: str


class TestJsonExtraction:
    """Test JSON extraction from various LLM output formats."""

    def test_direct_json(self):
        """Direct JSON parsing (no wrapper)."""
        raw = '{"key": "value", "number": 42}'
        result = extract_json(raw)
        assert result == {"key": "value", "number": 42}

    def test_code_fence_json(self):
        """JSON in code fence with json tag."""
        raw = """
Here's the result:
```json
{
  "key": "value",
  "list": [1, 2, 3]
}
```
"""
        result = extract_json(raw)
        assert result == {"key": "value", "list": [1, 2, 3]}

    def test_code_fence_no_tag(self):
        """JSON in code fence without json tag."""
        raw = """
```
{"key": "value"}
```
"""
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_bare_block(self):
        """JSON as bare code block surrounded by text."""
        raw = """
Some preamble text
{"extracted": true, "data": ["a", "b"]}
Some trailing text
"""
        result = extract_json(raw)
        assert result == {"extracted": True, "data": ["a", "b"]}

    def test_invalid_json(self):
        """No valid JSON raises ValueError."""
        raw = "This is just plain text with no JSON at all"
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            extract_json(raw)


class TestContextStage:
    """Test ContextStage implementation."""

    @pytest.fixture
    def stage(self):
        return ContextStage()

    def test_name(self, stage):
        assert stage.name == "context"

    def test_progress_range(self, stage):
        assert stage.progress_range == (0.10, 0.25)

    def test_build_prompt(self, stage):
        messages = stage.build_prompt("Build a blog platform", {})
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Build a blog platform" in messages[0]["content"]

    def test_parse_output(self, stage):
        raw = json.dumps(
            {
                "project_description": "A blogging platform",
                "key_requirements": ["User auth", "Post creation"],
                "constraints": ["Must use Python"],
                "existing_context": "",
                "stakeholders": ["Bloggers", "Readers"],
                "scope_boundaries": {"in_scope": ["Posts"], "out_of_scope": ["Ads"]},
            }
        )
        result = stage.parse_output(raw)
        assert result["project_description"] == "A blogging platform"
        assert len(result["key_requirements"]) == 2
        assert "Must use Python" in result["constraints"]

    @pytest.mark.asyncio
    async def test_execute_success(self, stage):
        """Test successful execution with mock LLM."""
        mock_client = AsyncMock()
        mock_client.generate_chat.return_value = MockLLMResponse(
            content=json.dumps(
                {
                    "project_description": "Test project",
                    "key_requirements": ["Req 1"],
                    "constraints": [],
                    "existing_context": "",
                    "stakeholders": [],
                    "scope_boundaries": {},
                }
            )
        )

        result = await stage.execute(mock_client, "Build something", {})

        assert result.success is True
        assert result.stage_name == "context"
        assert result.output["project_description"] == "Test project"
        mock_client.generate_chat.assert_called_once()


class TestArchitectureStage:
    """Test ArchitectureStage with two-stage prompting."""

    @pytest.fixture
    def stage(self):
        return ArchitectureStage()

    def test_name(self, stage):
        assert stage.name == "architecture"

    def test_progress_range(self, stage):
        assert stage.progress_range == (0.25, 0.45)

    def test_build_prompt_no_prior(self, stage):
        """Build prompt without prior context."""
        messages = stage.build_prompt("Build an API", {})
        assert len(messages) == 1
        assert "Build an API" in messages[0]["content"]

    def test_build_prompt_with_context(self, stage):
        """Build prompt with prior context output."""
        prior = {
            "context": {
                "project_description": "API platform",
                "key_requirements": ["REST", "GraphQL"],
                "constraints": ["Must scale"],
            }
        }
        messages = stage.build_prompt("Build an API", prior)
        assert "API platform" in messages[0]["content"]
        assert "REST" in messages[0]["content"]

    def test_parse_output(self, stage):
        raw = json.dumps(
            {
                "approaches": [
                    {
                        "name": "Monolith",
                        "description": "Single app",
                        "pros": ["Simple"],
                        "cons": ["Scaling"],
                        "complexity": "low",
                        "best_for": ["MVPs"],
                    }
                ],
                "recommended": "Monolith",
                "reasoning": "Best for MVP",
                "key_tradeoffs": {"simplicity": "vs scalability"},
                "technology_considerations": ["Python", "Flask"],
            }
        )
        result = stage.parse_output(raw)
        assert result["recommended"] == "Monolith"
        assert len(result["approaches"]) == 1

    @pytest.mark.asyncio
    async def test_execute_two_stage(self, stage):
        """Test two-stage execution (reasoning + formatting)."""
        mock_client = AsyncMock()

        # First call: free-form reasoning
        # Second call: JSON formatting
        mock_client.generate_chat.side_effect = [
            MockLLMResponse(content="I think a monolith is best because..."),
            MockLLMResponse(
                content=json.dumps(
                    {
                        "approaches": [],
                        "recommended": "Monolith",
                        "reasoning": "Best for MVP",
                        "key_tradeoffs": {},
                        "technology_considerations": [],
                    }
                )
            ),
        ]

        result = await stage.execute(mock_client, "Build API", {})

        assert result.success is True
        assert result.stage_name == "architecture"
        assert result.output["recommended"] == "Monolith"
        # Two LLM calls (reasoning + formatting)
        assert mock_client.generate_chat.call_count == 2


class TestDesignStage:
    """Test DesignStage with ADRs."""

    @pytest.fixture
    def stage(self):
        return DesignStage()

    def test_name(self, stage):
        assert stage.name == "design"

    def test_progress_range(self, stage):
        assert stage.progress_range == (0.45, 0.65)

    def test_build_prompt_with_prior(self, stage):
        """Include context and architecture in prompt."""
        prior = {
            "context": {
                "project_description": "Blog",
                "key_requirements": ["Auth"],
            },
            "architecture": {
                "recommended": "Monolith",
                "reasoning": "Simple MVP",
                "key_tradeoffs": {"simple": "vs scalable"},
            },
        }
        messages = stage.build_prompt("Build blog", prior)
        assert "Blog" in messages[0]["content"]
        assert "Monolith" in messages[0]["content"]

    def test_parse_output(self, stage):
        raw = json.dumps(
            {
                "adrs": [
                    {
                        "title": "Use JWT",
                        "context": "Need auth",
                        "decision": "JWT tokens",
                        "rationale": "Stateless",
                        "consequences": ["Easy to scale"],
                        "alternatives_considered": ["Sessions"],
                    }
                ],
                "components": [],
                "data_model": {"User": ["id", "email"]},
                "integration_points": ["Stripe API"],
            }
        )
        result = stage.parse_output(raw)
        assert len(result["adrs"]) == 1
        assert result["adrs"][0]["title"] == "Use JWT"

    @pytest.mark.asyncio
    async def test_execute_with_prior_outputs(self, stage):
        """Ensure prior outputs are incorporated."""
        mock_client = AsyncMock()
        mock_client.generate_chat.return_value = MockLLMResponse(
            content=json.dumps(
                {
                    "adrs": [],
                    "components": [],
                    "data_model": {},
                    "integration_points": [],
                }
            )
        )

        prior = {"context": {"project_description": "Test"}}
        result = await stage.execute(mock_client, "Build", prior)

        assert result.success is True
        # Verify prompt included prior outputs
        call_args = mock_client.generate_chat.call_args
        messages = call_args.kwargs["messages"]
        assert "Test" in messages[0]["content"]


class TestRoadmapStage:
    """Test RoadmapStage with phases and dependencies."""

    @pytest.fixture
    def stage(self):
        return RoadmapStage()

    def test_name(self, stage):
        assert stage.name == "roadmap"

    def test_progress_range(self, stage):
        assert stage.progress_range == (0.65, 0.80)

    def test_parse_output(self, stage):
        raw = json.dumps(
            {
                "phases": [
                    {
                        "number": 1,
                        "name": "Foundation",
                        "objective": "Setup infra",
                        "deliverables": ["Database", "Auth"],
                        "dependencies": [],
                        "estimated_complexity": "medium",
                        "key_risks": ["Data migration"],
                    },
                    {
                        "number": 2,
                        "name": "Features",
                        "objective": "Build features",
                        "deliverables": ["Posts"],
                        "dependencies": [1],
                        "estimated_complexity": "high",
                        "key_risks": [],
                    },
                ],
                "critical_path": [1, 2],
                "parallel_opportunities": [],
                "total_phases": 2,
            }
        )
        result = stage.parse_output(raw)
        assert result["total_phases"] == 2
        assert len(result["phases"]) == 2
        assert result["phases"][1]["dependencies"] == [1]

    @pytest.mark.asyncio
    async def test_execute(self, stage):
        mock_client = AsyncMock()
        mock_client.generate_chat.return_value = MockLLMResponse(
            content=json.dumps(
                {
                    "phases": [],
                    "critical_path": [],
                    "parallel_opportunities": [],
                    "total_phases": 0,
                }
            )
        )

        result = await stage.execute(mock_client, "Build", {})
        assert result.success is True


class TestRiskStage:
    """Test RiskStage with severity and mitigation."""

    @pytest.fixture
    def stage(self):
        return RiskStage()

    def test_name(self, stage):
        assert stage.name == "risk"

    def test_progress_range(self, stage):
        assert stage.progress_range == (0.80, 0.95)

    def test_parse_output(self, stage):
        raw = json.dumps(
            {
                "risks": [
                    {
                        "category": "technical",
                        "description": "Database scaling",
                        "impact": "high",
                        "likelihood": "medium",
                        "mitigation": "Use sharding",
                        "contingency": "Move to managed DB",
                        "affected_phases": [2, 3],
                    }
                ],
                "overall_risk_level": "medium",
                "recommended_contingencies": ["Have backup DB"],
            }
        )
        result = stage.parse_output(raw)
        assert result["overall_risk_level"] == "medium"
        assert len(result["risks"]) == 1
        assert result["risks"][0]["impact"] == "high"

    @pytest.mark.asyncio
    async def test_execute(self, stage):
        mock_client = AsyncMock()
        mock_client.generate_chat.return_value = MockLLMResponse(
            content=json.dumps(
                {
                    "risks": [],
                    "overall_risk_level": "low",
                    "recommended_contingencies": [],
                }
            )
        )

        result = await stage.execute(mock_client, "Build", {})
        assert result.success is True


class TestDefaultStages:
    """Test DEFAULT_STAGES list."""

    def test_count(self):
        """Should have exactly 5 stages."""
        assert len(DEFAULT_STAGES) == 5

    def test_names(self):
        """Verify stage names in order."""
        expected = ["context", "architecture", "design", "roadmap", "risk"]
        actual = [stage.name for stage in DEFAULT_STAGES]
        assert actual == expected

    def test_progress_ranges_non_overlapping(self):
        """Progress ranges should be sequential and non-overlapping."""
        ranges = [stage.progress_range for stage in DEFAULT_STAGES]

        # Check sequential (end of N == start of N+1 is ideal, but just check non-overlap)
        for i in range(len(ranges) - 1):
            current_end = ranges[i][1]
            next_start = ranges[i + 1][0]
            assert (
                current_end <= next_start
            ), f"Overlap between stage {i} and {i+1}: {current_end} > {next_start}"

        # Check first starts at or after 0.0
        assert ranges[0][0] >= 0.0

        # Check last ends at or before 1.0
        assert ranges[-1][1] <= 1.0

    def test_all_stages_instantiable(self):
        """All stages should be instantiable."""
        for stage in DEFAULT_STAGES:
            assert hasattr(stage, "name")
            assert hasattr(stage, "progress_range")
            assert hasattr(stage, "build_prompt")
            assert hasattr(stage, "parse_output")
            assert hasattr(stage, "execute")
