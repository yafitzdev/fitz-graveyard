# tests/unit/test_stages.py
"""Unit tests for pipeline stages (3 merged stages + per-field extraction)."""

import json
from unittest.mock import AsyncMock

import pytest

from fitz_graveyard.planning.pipeline.stages import (
    DEFAULT_STAGES,
    ArchitectureDesignStage,
    ContextStage,
    RoadmapRiskStage,
    extract_json,
)
from fitz_graveyard.planning.pipeline.stages.base import _count_unclosed_delimiters
from fitz_graveyard.planning.pipeline.stages.roadmap_risk import _remove_dependency_cycles


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

    def test_truncated_json_repaired(self):
        """Truncated JSON (unclosed braces) is repaired by appending closing delimiters."""
        raw = '{"approaches": [{"name": "Option A", "pros": ["fast"'
        result = extract_json(raw)
        assert result["approaches"][0]["name"] == "Option A"
        assert result["approaches"][0]["pros"] == ["fast"]

    def test_truncated_json_with_braces_in_strings(self):
        """String-aware repair: braces inside JSON strings don't confuse the counter."""
        raw = '{"regex": "\\\\{.*\\\\}", "data": [{"name": "test"'
        result = extract_json(raw)
        assert result["regex"] == "\\{.*\\}"
        assert result["data"][0]["name"] == "test"

    def test_truncated_inside_string_value(self):
        """Repair handles truncation mid-string (LLM hit token limit)."""
        raw = '{"adrs": [{"title": "Use YAML", "context": "Dynamic routing ba'
        result = extract_json(raw)
        assert result["adrs"][0]["title"] == "Use YAML"
        assert "context" in result["adrs"][0] or len(result["adrs"]) >= 1

    def test_truncated_after_comma_in_object(self):
        """Repair handles truncation after a comma between key-value pairs."""
        raw = '{"name": "Monolith", "description": "Single app",'
        result = extract_json(raw)
        assert result["name"] == "Monolith"
        assert result["description"] == "Single app"

    def test_truncated_deep_nesting(self):
        """Repair handles deeply nested truncation."""
        raw = '{"phases": [{"number": 1, "tasks": [{"name": "Setup"'
        result = extract_json(raw)
        assert result["phases"][0]["number"] == 1

    def test_json_with_trailing_text_after_balanced_braces(self):
        """Repair handles trailing text after balanced JSON (e.g. model adds commentary)."""
        raw = '{"name": "test", "items": [1, 2]}\nHere is the result above.'
        result = extract_json(raw)
        assert result["name"] == "test"

    def test_severely_truncated_large_json(self):
        """Repair handles large JSON truncated far from the end (needs many iterations)."""
        # Build a large JSON with many entries, then truncate mid-way through
        entries = [{"id": i, "name": f"entry_{i}", "desc": "x" * 50} for i in range(20)]
        full = json.dumps({"items": entries})
        # Truncate at ~60% — leaves many unclosed delimiters
        truncated = full[:int(len(full) * 0.6)]
        result = extract_json(truncated)
        assert "items" in result
        assert len(result["items"]) > 0

    def test_invalid_json(self):
        """No valid JSON raises ValueError with preview."""
        raw = "This is just plain text with no JSON at all"
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            extract_json(raw)

    def test_invalid_json_error_includes_length(self):
        """Error message includes char count and preview."""
        raw = "No JSON here at all, just text."
        with pytest.raises(ValueError, match=r"\d+ chars"):
            extract_json(raw)


class TestCountUnclosedDelimiters:
    """Test string-aware delimiter counting."""

    def test_simple_balanced(self):
        braces, brackets, in_str = _count_unclosed_delimiters('{"a": [1, 2]}')
        assert braces == 0
        assert brackets == 0
        assert in_str is False

    def test_unclosed_brace(self):
        braces, brackets, in_str = _count_unclosed_delimiters('{"a": "b"')
        assert braces == 1
        assert brackets == 0
        assert in_str is False

    def test_braces_inside_strings_ignored(self):
        braces, brackets, in_str = _count_unclosed_delimiters('{"regex": "\\\\{.*\\\\}"')
        assert braces == 1  # only the outer { is unclosed
        assert brackets == 0

    def test_escaped_quotes(self):
        braces, brackets, in_str = _count_unclosed_delimiters('{"key": "val\\\\"')
        assert braces == 1

    def test_unclosed_bracket(self):
        braces, brackets, in_str = _count_unclosed_delimiters('{"a": [1, 2')
        assert braces == 1
        assert brackets == 1

    def test_truncated_inside_string(self):
        braces, brackets, in_str = _count_unclosed_delimiters('{"key": "truncated val')
        assert braces == 1
        assert in_str is True


class TestArchitectureDesignRecommendedValidation:
    """Test fuzzy matching of recommended approach name in merged stage."""

    @pytest.fixture
    def stage(self):
        return ArchitectureDesignStage()

    def test_exact_match_unchanged(self, stage):
        raw = json.dumps({
            "approaches": [{"name": "Monolith", "description": "Single app", "pros": [], "cons": [], "complexity": "low", "best_for": []}],
            "recommended": "Monolith",
            "reasoning": "Best for MVP",
            "key_tradeoffs": {},
            "technology_considerations": [],
            "adrs": [],
            "components": [],
            "data_model": {},
            "integration_points": [],
            "artifacts": [],
            "scope_statement": "",
        })
        result = stage.parse_output(raw)
        assert result["architecture"]["recommended"] == "Monolith"

    def test_fuzzy_match_corrected(self, stage):
        raw = json.dumps({
            "approaches": [{"name": "Microservices Architecture", "description": "Distributed", "pros": [], "cons": [], "complexity": "high", "best_for": []}],
            "recommended": "Microservices",
            "reasoning": "Best for scale",
            "key_tradeoffs": {},
            "technology_considerations": [],
            "adrs": [],
            "components": [],
            "data_model": {},
            "integration_points": [],
            "artifacts": [],
            "scope_statement": "",
        })
        result = stage.parse_output(raw)
        assert result["architecture"]["recommended"] == "Microservices Architecture"

    def test_no_match_uses_first(self, stage):
        raw = json.dumps({
            "approaches": [{"name": "Monolith", "description": "Single", "pros": [], "cons": [], "complexity": "low", "best_for": []}],
            "recommended": "Something Completely Different",
            "reasoning": "Reason",
            "key_tradeoffs": {},
            "technology_considerations": [],
            "adrs": [],
            "components": [],
            "data_model": {},
            "integration_points": [],
            "artifacts": [],
            "scope_statement": "",
        })
        result = stage.parse_output(raw)
        assert result["architecture"]["recommended"] == "Monolith"

    def test_empty_approaches_unchanged(self, stage):
        raw = json.dumps({
            "approaches": [],
            "recommended": "Anything",
            "reasoning": "Reason",
            "key_tradeoffs": {},
            "technology_considerations": [],
            "adrs": [],
            "components": [],
            "data_model": {},
            "integration_points": [],
            "artifacts": [],
            "scope_statement": "",
        })
        result = stage.parse_output(raw)
        assert result["architecture"]["recommended"] == "Anything"


class TestRoadmapCycleDetection:
    """Test dependency cycle removal in roadmap phases."""

    def test_valid_deps_unchanged(self):
        phases = [
            {"number": 1, "dependencies": []},
            {"number": 2, "dependencies": [1]},
            {"number": 3, "dependencies": [1, 2]},
        ]
        result = _remove_dependency_cycles(phases)
        assert result[1]["dependencies"] == [1]
        assert result[2]["dependencies"] == [1, 2]

    def test_back_edge_removed(self):
        phases = [
            {"number": 1, "dependencies": []},
            {"number": 2, "dependencies": [1, 3]},  # 3 is a forward ref
            {"number": 3, "dependencies": [2]},
        ]
        result = _remove_dependency_cycles(phases)
        assert result[1]["dependencies"] == [1]  # 3 removed (>= 2)

    def test_self_reference_removed(self):
        phases = [
            {"number": 1, "dependencies": [1]},  # self-ref
        ]
        result = _remove_dependency_cycles(phases)
        assert result[0]["dependencies"] == []

    def test_nonexistent_dep_removed(self):
        phases = [
            {"number": 1, "dependencies": []},
            {"number": 2, "dependencies": [1, 99]},  # 99 doesn't exist
        ]
        result = _remove_dependency_cycles(phases)
        assert result[1]["dependencies"] == [1]


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
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Build a blog platform" in messages[1]["content"]

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
    async def test_execute_per_field(self, stage):
        """Test per-field extraction: 1 reasoning + 1 critique + 4 field groups = 6 LLM calls."""
        mock_client = AsyncMock()
        mock_client.generate.side_effect = [
            # 1. Reasoning
            "Detailed reasoning about project context...",
            # 2. Self-critique
            "Reviewed and refined reasoning about project context...",
            # Group 1: description
            json.dumps({
                "project_description": "Test project",
                "key_requirements": ["Req 1"],
                "constraints": ["Python only"],
                "existing_context": "",
            }),
            # Group 2: stakeholders
            json.dumps({
                "stakeholders": ["Developers"],
                "scope_boundaries": {"in_scope": ["API"], "out_of_scope": ["UI"]},
            }),
            # Group 3: files
            json.dumps({
                "existing_files": [],
                "needed_artifacts": ["config.yaml"],
            }),
            # Group 4: assumptions
            json.dumps({
                "assumptions": [],
            }),
        ]

        result = await stage.execute(mock_client, "Build something", {})

        assert result.success is True
        assert result.stage_name == "context"
        assert result.output["project_description"] == "Test project"
        assert result.output["constraints"] == ["Python only"]
        assert result.output["needed_artifacts"] == ["config.yaml"]
        assert mock_client.generate.call_count == 6  # 1 reasoning + 1 critique + 4 groups

    @pytest.mark.asyncio
    async def test_execute_partial_failure(self, stage):
        """One field group failing still produces a valid result with defaults."""
        mock_client = AsyncMock()
        mock_client.generate.side_effect = [
            "Reasoning...",
            "Reviewed reasoning...",
            # Group 1: description — valid
            json.dumps({
                "project_description": "Test project",
                "key_requirements": ["Req 1"],
                "constraints": [],
                "existing_context": "",
            }),
            # Group 2: stakeholders — FAILS
            "This is not JSON at all",
            # Group 3: files — valid
            json.dumps({"existing_files": [], "needed_artifacts": []}),
            # Group 4: assumptions — valid
            json.dumps({"assumptions": []}),
        ]

        result = await stage.execute(mock_client, "Build something", {})
        assert result.success is True
        # stakeholders group failed → defaults
        assert result.output["stakeholders"] == []
        assert result.output["scope_boundaries"] == {}
        # Other fields still populated
        assert result.output["project_description"] == "Test project"


class TestArchitectureDesignStage:
    """Test merged ArchitectureDesignStage with per-field extraction."""

    @pytest.fixture
    def stage(self):
        return ArchitectureDesignStage()

    def test_name(self, stage):
        assert stage.name == "architecture_design"

    def test_progress_range(self, stage):
        assert stage.progress_range == (0.25, 0.65)

    def test_build_prompt_no_prior(self, stage):
        """Build prompt without prior context."""
        messages = stage.build_prompt("Build an API", {})
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        assert "Build an API" in messages[1]["content"]

    def test_build_prompt_with_context(self, stage):
        """Build prompt with prior context output."""
        prior = {
            "context": {
                "project_description": "API platform",
                "key_requirements": ["REST", "GraphQL"],
                "constraints": ["Must scale"],
                "existing_files": ["src/api.py — existing API"],
                "needed_artifacts": ["openapi.yaml — API spec"],
            }
        }
        messages = stage.build_prompt("Build an API", prior)
        assert "API platform" in messages[1]["content"]
        assert "REST" in messages[1]["content"]
        assert "src/api.py" in messages[1]["content"]
        assert "openapi.yaml" in messages[1]["content"]

    def test_parse_output_splits_correctly(self, stage):
        """parse_output splits combined JSON into architecture and design sub-dicts."""
        raw = json.dumps({
            "approaches": [
                {"name": "Monolith", "description": "Single app", "pros": ["Simple"],
                 "cons": ["Scaling"], "complexity": "low", "best_for": ["MVPs"]},
            ],
            "recommended": "Monolith",
            "reasoning": "Best for MVP",
            "key_tradeoffs": {"simplicity": "vs scalability"},
            "technology_considerations": ["Python", "Flask"],
            "adrs": [
                {"title": "Use JWT", "context": "Need auth", "decision": "JWT tokens",
                 "rationale": "Stateless", "consequences": ["Easy to scale"],
                 "alternatives_considered": ["Sessions"]},
            ],
            "components": [],
            "data_model": {"User": ["id", "email"]},
            "integration_points": ["Stripe API"],
            "artifacts": [
                {"filename": "config.yaml", "content": "key: value", "purpose": "App config"},
            ],
            "scope_statement": "Small API — one service, one database.",
        })
        result = stage.parse_output(raw)

        # Check split structure
        assert "architecture" in result
        assert "design" in result

        # Architecture fields
        assert result["architecture"]["recommended"] == "Monolith"
        assert len(result["architecture"]["approaches"]) == 1
        assert result["architecture"]["scope_statement"] == "Small API — one service, one database."

        # Design fields
        assert len(result["design"]["adrs"]) == 1
        assert result["design"]["adrs"][0]["title"] == "Use JWT"
        assert result["design"]["data_model"] == {"User": ["id", "email"]}
        assert len(result["design"]["artifacts"]) == 1
        assert result["design"]["artifacts"][0]["filename"] == "config.yaml"

    @pytest.mark.asyncio
    async def test_execute_per_field(self, stage):
        """Test per-field extraction: 1 reasoning + 1 critique + 6 field groups = 8 LLM calls."""
        mock_client = AsyncMock()
        # 1 reasoning + 1 critique + 6 field group extractions
        mock_client.generate.side_effect = [
            "Detailed reasoning about architecture and design...",
            "Reviewed and refined reasoning...",
            # Group 1: approaches
            json.dumps({
                "approaches": [{"name": "Monolith", "description": "Single app", "pros": ["Simple"], "cons": ["Scaling"], "complexity": "low", "best_for": ["MVP"]}],
                "recommended": "Monolith",
                "reasoning": "Simple approach",
                "scope_statement": "Small project",
            }),
            # Group 2: tradeoffs
            json.dumps({
                "key_tradeoffs": {"simplicity": "vs scale"},
                "technology_considerations": ["Python"],
            }),
            # Group 3: adrs
            json.dumps({
                "adrs": [],
            }),
            # Group 4: components
            json.dumps({
                "components": [],
                "data_model": {},
            }),
            # Group 5: integrations
            json.dumps({
                "integration_points": [],
            }),
            # Group 6: artifacts
            json.dumps({
                "artifacts": [],
            }),
        ]

        result = await stage.execute(mock_client, "Build API", {})
        assert result.success is True
        assert "architecture" in result.output
        assert "design" in result.output
        assert result.output["architecture"]["recommended"] == "Monolith"
        assert result.output["architecture"]["key_tradeoffs"] == {"simplicity": "vs scale"}
        assert mock_client.generate.call_count == 8  # 1 reasoning + 1 critique + 6 groups

    @pytest.mark.asyncio
    async def test_execute_passes_krag_context_selectively(self, stage):
        """Codebase context is passed to approaches, adrs, artifacts but not others."""
        mock_client = AsyncMock()
        krag = "## Codebase Summary\nThis project has openai.py and config.py."
        prior = {"_gathered_context": krag}

        mock_client.generate.side_effect = [
            "Reasoning...",
            "Reviewed reasoning...",  # critique
            json.dumps({"approaches": [], "recommended": "", "reasoning": "", "scope_statement": ""}),
            json.dumps({"key_tradeoffs": {}, "technology_considerations": []}),
            json.dumps({"adrs": []}),
            json.dumps({"components": [], "data_model": {}}),
            json.dumps({"integration_points": []}),
            json.dumps({"artifacts": []}),
        ]

        result = await stage.execute(mock_client, "Build API", prior)
        assert result.success is True

        # Calls: [0]=reasoning, [1]=critique, [2]=approaches, [3]=tradeoffs, [4]=adrs,
        #        [5]=components, [6]=integrations, [7]=artifacts
        calls = mock_client.generate.call_args_list

        # approaches (call 2), adrs (call 4), artifacts (call 7) should have krag
        assert "Codebase Summary" in calls[2].kwargs["messages"][1]["content"]
        assert "Codebase Summary" in calls[4].kwargs["messages"][1]["content"]
        assert "Codebase Summary" in calls[7].kwargs["messages"][1]["content"]

        # tradeoffs (call 3), components (call 5), integrations (call 6) should NOT
        assert "Codebase Summary" not in calls[3].kwargs["messages"][1]["content"]
        assert "Codebase Summary" not in calls[5].kwargs["messages"][1]["content"]
        assert "Codebase Summary" not in calls[6].kwargs["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_execute_partial_failure(self, stage):
        """One field group failing still produces a valid plan with defaults."""
        mock_client = AsyncMock()
        mock_client.generate.side_effect = [
            "Reasoning...",
            "Reviewed reasoning...",  # critique
            # Group 1: approaches — valid
            json.dumps({
                "approaches": [{"name": "Mono", "description": "Single", "pros": [], "cons": [], "complexity": "low", "best_for": []}],
                "recommended": "Mono",
                "reasoning": "Simple",
                "scope_statement": "",
            }),
            # Group 2: tradeoffs — FAILS (invalid JSON)
            "This is not JSON at all",
            # Group 3: adrs — valid
            json.dumps({"adrs": []}),
            # Group 4: components — valid
            json.dumps({"components": [], "data_model": {}}),
            # Group 5: integrations — valid
            json.dumps({"integration_points": []}),
            # Group 6: artifacts — valid
            json.dumps({"artifacts": []}),
        ]

        result = await stage.execute(mock_client, "Build API", {})
        assert result.success is True
        # tradeoffs group failed → defaults
        assert result.output["architecture"]["key_tradeoffs"] == {}
        assert result.output["architecture"]["technology_considerations"] == []
        # Other fields still populated
        assert result.output["architecture"]["recommended"] == "Mono"

    @pytest.mark.asyncio
    async def test_execute_all_groups_fail(self, stage):
        """All field groups failing still produces a skeleton plan."""
        mock_client = AsyncMock()
        mock_client.generate.side_effect = [
            "Reasoning...",
            "Reviewed reasoning...",  # critique
            # All 6 groups fail
            "not json", "not json", "not json", "not json", "not json", "not json",
        ]

        result = await stage.execute(mock_client, "Build API", {})
        assert result.success is True
        # All defaults
        assert result.output["architecture"]["approaches"] == []
        assert result.output["design"]["adrs"] == []
        assert result.output["design"]["components"] == []


class TestRoadmapRiskStage:
    """Test merged RoadmapRiskStage with per-field extraction."""

    @pytest.fixture
    def stage(self):
        return RoadmapRiskStage()

    def test_name(self, stage):
        assert stage.name == "roadmap_risk"

    def test_progress_range(self, stage):
        assert stage.progress_range == (0.65, 0.95)

    def test_build_prompt_with_prior(self, stage):
        """Include context and architecture+design in prompt."""
        prior = {
            "context": {
                "project_description": "Blog",
                "key_requirements": ["Auth"],
                "constraints": ["Python"],
                "scope_boundaries": {},
            },
            "architecture": {
                "recommended": "Monolith",
                "reasoning": "Simple",
                "key_tradeoffs": {},
            },
            "design": {
                "components": [{"name": "API", "purpose": "REST endpoints", "interfaces": ["GET /posts"], "dependencies": ["DB"]}],
                "adrs": [{"title": "Use JWT", "decision": "JWT tokens for auth", "rationale": "Stateless"}],
                "integration_points": ["Stripe"],
                "artifacts": [{"filename": "schema.sql", "content": "CREATE TABLE posts;", "purpose": "DB schema"}],
            },
        }
        messages = stage.build_prompt("Build blog", prior)
        content = messages[1]["content"]
        assert "Monolith" in content
        assert "API" in content
        # Full ADR details should be passed
        assert "JWT tokens for auth" in content
        # Component details
        assert "REST endpoints" in content
        # Artifacts mentioned
        assert "schema.sql" in content

    def test_parse_output_splits_correctly(self, stage):
        """parse_output splits combined JSON into roadmap and risk sub-dicts."""
        raw = json.dumps({
            "phases": [
                {"number": 1, "name": "Foundation", "objective": "Setup",
                 "deliverables": ["DB"], "dependencies": [],
                 "estimated_complexity": "medium", "key_risks": []},
            ],
            "critical_path": [1],
            "parallel_opportunities": [],
            "total_phases": 1,
            "risks": [
                {"category": "technical", "description": "DB scaling",
                 "impact": "high", "likelihood": "medium",
                 "mitigation": "Use sharding", "contingency": "Managed DB",
                 "affected_phases": [1]},
            ],
            "overall_risk_level": "medium",
            "recommended_contingencies": ["Have backup"],
        })
        result = stage.parse_output(raw)

        assert "roadmap" in result
        assert "risk" in result

        assert result["roadmap"]["total_phases"] == 1
        assert len(result["roadmap"]["phases"]) == 1
        assert result["risk"]["overall_risk_level"] == "medium"
        assert len(result["risk"]["risks"]) == 1

    @pytest.mark.asyncio
    async def test_execute_per_field(self, stage):
        """Test per-field extraction: 1 reasoning + 1 critique + 3 field groups = 5 LLM calls."""
        mock_client = AsyncMock()
        mock_client.generate.side_effect = [
            "Reasoning about roadmap and risks...",
            "Reviewed and refined reasoning...",  # critique
            # Group 1: phases
            json.dumps({
                "phases": [
                    {"number": 1, "name": "Setup", "objective": "Initialize", "deliverables": ["DB"],
                     "dependencies": [], "estimated_complexity": "low", "key_risks": []},
                ],
            }),
            # Group 2: scheduling
            json.dumps({
                "critical_path": [1],
                "parallel_opportunities": [],
                "total_phases": 1,
            }),
            # Group 3: risks
            json.dumps({
                "risks": [],
                "overall_risk_level": "low",
                "recommended_contingencies": [],
            }),
        ]

        result = await stage.execute(mock_client, "Build", {})
        assert result.success is True
        assert "roadmap" in result.output
        assert "risk" in result.output
        assert result.output["roadmap"]["total_phases"] == 1
        assert result.output["risk"]["overall_risk_level"] == "low"
        assert mock_client.generate.call_count == 5  # 1 reasoning + 1 critique + 3 groups

    @pytest.mark.asyncio
    async def test_execute_partial_failure(self, stage):
        """One field group failing uses defaults."""
        mock_client = AsyncMock()
        mock_client.generate.side_effect = [
            "Reasoning...",
            "Reviewed reasoning...",  # critique
            # Group 1: phases — valid
            json.dumps({"phases": []}),
            # Group 2: scheduling — FAILS
            "not json",
            # Group 3: risks — valid
            json.dumps({"risks": [], "overall_risk_level": "low", "recommended_contingencies": []}),
        ]

        result = await stage.execute(mock_client, "Build", {})
        assert result.success is True
        # scheduling failed → defaults
        assert result.output["roadmap"]["critical_path"] == []
        assert result.output["roadmap"]["parallel_opportunities"] == []


class TestDefaultStages:
    """Test DEFAULT_STAGES list."""

    def test_count(self):
        """Should have exactly 3 stages."""
        assert len(DEFAULT_STAGES) == 3

    def test_names(self):
        """Verify stage names in order."""
        expected = ["context", "architecture_design", "roadmap_risk"]
        actual = [stage.name for stage in DEFAULT_STAGES]
        assert actual == expected

    def test_progress_ranges_non_overlapping(self):
        """Progress ranges should be sequential and non-overlapping."""
        ranges = [stage.progress_range for stage in DEFAULT_STAGES]

        for i in range(len(ranges) - 1):
            current_end = ranges[i][1]
            next_start = ranges[i + 1][0]
            assert (
                current_end <= next_start
            ), f"Overlap between stage {i} and {i+1}: {current_end} > {next_start}"

        assert ranges[0][0] >= 0.0
        assert ranges[-1][1] <= 1.0

    def test_all_stages_instantiable(self):
        """All stages should be instantiable."""
        for stage in DEFAULT_STAGES:
            assert hasattr(stage, "name")
            assert hasattr(stage, "progress_range")
            assert hasattr(stage, "build_prompt")
            assert hasattr(stage, "parse_output")
            assert hasattr(stage, "execute")


class TestExtractFieldGroup:
    """Test the _extract_field_group helper on PipelineStage."""

    @pytest.mark.asyncio
    async def test_valid_json_extraction(self):
        """Returns parsed dict when LLM returns valid JSON."""
        stage = ContextStage()
        mock_client = AsyncMock()
        mock_client.generate.return_value = '{"key": "value", "items": [1, 2]}'

        result = await stage._extract_field_group(
            mock_client,
            "Some reasoning text",
            ["key", "items"],
            '{"key": "str", "items": []}',
            "test_group",
        )
        assert result == {"key": "value", "items": [1, 2]}

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty(self):
        """Returns {} when LLM returns invalid JSON."""
        stage = ContextStage()
        mock_client = AsyncMock()
        mock_client.generate.return_value = "This is not JSON at all"

        result = await stage._extract_field_group(
            mock_client,
            "Some reasoning text",
            ["key"],
            '{"key": "str"}',
            "test_group",
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_client_error_returns_empty(self):
        """Returns {} when LLM client raises an exception."""
        stage = ContextStage()
        mock_client = AsyncMock()
        mock_client.generate.side_effect = RuntimeError("Connection failed")

        result = await stage._extract_field_group(
            mock_client,
            "Some reasoning text",
            ["key"],
            '{"key": "str"}',
            "test_group",
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_reports_substep(self):
        """Reports extracting:{label} substep."""
        stage = ContextStage()
        reported = []

        async def track(phase: str) -> None:
            reported.append(phase)

        stage.set_substep_callback(track)
        mock_client = AsyncMock()
        mock_client.generate.return_value = '{"x": 1}'

        await stage._extract_field_group(
            mock_client, "reasoning", ["x"], '{"x": 0}', "my_group"
        )
        assert "context:extracting:my_group" in reported

    @pytest.mark.asyncio
    async def test_extra_context_included_in_prompt(self):
        """extra_context is injected into the extraction prompt."""
        stage = ContextStage()
        mock_client = AsyncMock()
        mock_client.generate.return_value = '{"x": 1}'

        await stage._extract_field_group(
            mock_client, "reasoning", ["x"], '{"x": 0}', "grp",
            extra_context="## Codebase: has openai.py",
        )
        prompt = mock_client.generate.call_args.kwargs["messages"][1]["content"]
        assert "Codebase: has openai.py" in prompt
        assert "CODEBASE CONTEXT" in prompt

    @pytest.mark.asyncio
    async def test_no_extra_context_no_block(self):
        """Without extra_context, no codebase context block in prompt."""
        stage = ContextStage()
        mock_client = AsyncMock()
        mock_client.generate.return_value = '{"x": 1}'

        await stage._extract_field_group(
            mock_client, "reasoning", ["x"], '{"x": 0}', "grp",
        )
        prompt = mock_client.generate.call_args.kwargs["messages"][1]["content"]
        assert "CODEBASE CONTEXT" not in prompt


class TestSinglePassWithFallback:
    """Test the _single_pass_with_fallback method on PipelineStage."""

    @pytest.mark.asyncio
    async def test_single_pass_success(self):
        """Single-pass succeeds when LLM returns valid JSON."""
        stage = ContextStage()
        mock_client = AsyncMock()
        valid_json = '{"project_description": "Test", "key_requirements": []}'
        mock_client.generate.return_value = valid_json

        messages = [
            {"role": "system", "content": "You are an architect"},
            {"role": "user", "content": "Plan something"},
        ]
        result = await stage._single_pass_with_fallback(mock_client, messages, '{"schema": "here"}')

        assert result == valid_json
        assert mock_client.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_single_pass_fallback(self):
        """Falls back to two-pass when single-pass returns non-JSON."""
        stage = ContextStage()
        mock_client = AsyncMock()
        valid_json = json.dumps({
            "project_description": "Test",
            "key_requirements": [],
            "constraints": [],
            "existing_context": "",
            "stakeholders": [],
            "scope_boundaries": {},
        })
        # First call: single-pass returns prose
        # Second call: two-pass reasoning
        # Third call: two-pass formatting returns JSON
        mock_client.generate.side_effect = [
            "This is not JSON, let me think about it...",
            "Here's my reasoning about the project...",
            valid_json,
        ]

        messages = [
            {"role": "system", "content": "You are an architect"},
            {"role": "user", "content": "Plan something"},
        ]
        result = await stage._single_pass_with_fallback(mock_client, messages, '{"schema": "here"}')

        assert result == valid_json
        assert mock_client.generate.call_count == 3


class TestSubstepCallback:
    """Test sub-step callback mechanism in PipelineStage."""

    @pytest.mark.asyncio
    async def test_context_reports_reasoning_critiquing_and_extracting(self):
        """Context stage reports reasoning + critiquing + extracting:{group} substeps."""
        stage = ContextStage()
        reported = []

        async def track_substep(phase: str) -> None:
            reported.append(phase)

        stage.set_substep_callback(track_substep)

        mock_client = AsyncMock()
        mock_client.generate.side_effect = [
            "Reasoning...",
            "Reviewed reasoning...",  # critique
            json.dumps({"project_description": "Test", "key_requirements": [], "constraints": [], "existing_context": ""}),
            json.dumps({"stakeholders": [], "scope_boundaries": {}}),
            json.dumps({"existing_files": [], "needed_artifacts": []}),
            json.dumps({"assumptions": []}),
        ]

        result = await stage.execute(mock_client, "Build API", {})
        assert result.success is True
        assert "context:reasoning" in reported
        assert "context:critiquing" in reported
        assert "context:extracting:description" in reported
        assert "context:extracting:stakeholders" in reported
        assert "context:extracting:files" in reported
        assert "context:extracting:assumptions" in reported

    @pytest.mark.asyncio
    async def test_per_field_reports_reasoning_critiquing_and_extracting(self):
        """Per-field extraction reports reasoning + critiquing + extracting:{group} substeps."""
        stage = ArchitectureDesignStage()
        reported = []

        async def track_substep(phase: str) -> None:
            reported.append(phase)

        stage.set_substep_callback(track_substep)

        mock_client = AsyncMock()
        mock_client.generate.side_effect = [
            "Reasoning...",
            "Reviewed reasoning...",  # critique
            json.dumps({"approaches": [], "recommended": "", "reasoning": "", "scope_statement": ""}),
            json.dumps({"key_tradeoffs": {}, "technology_considerations": []}),
            json.dumps({"adrs": []}),
            json.dumps({"components": [], "data_model": {}}),
            json.dumps({"integration_points": []}),
            json.dumps({"artifacts": []}),
        ]

        result = await stage.execute(mock_client, "Build API", {})
        assert result.success is True
        assert "architecture_design:reasoning" in reported
        assert "architecture_design:critiquing" in reported
        assert "architecture_design:extracting:approaches" in reported
        assert "architecture_design:extracting:tradeoffs" in reported
        assert "architecture_design:extracting:adrs" in reported
        assert "architecture_design:extracting:components" in reported
        assert "architecture_design:extracting:integrations" in reported
        assert "architecture_design:extracting:artifacts" in reported

    @pytest.mark.asyncio
    async def test_no_callback_no_error(self):
        """Execution works fine without a substep callback set."""
        stage = ArchitectureDesignStage()

        mock_client = AsyncMock()
        mock_client.generate.side_effect = [
            "Reasoning...",
            "Reviewed reasoning...",  # critique
            json.dumps({"approaches": [], "recommended": "", "reasoning": "", "scope_statement": ""}),
            json.dumps({"key_tradeoffs": {}, "technology_considerations": []}),
            json.dumps({"adrs": []}),
            json.dumps({"components": [], "data_model": {}}),
            json.dumps({"integration_points": []}),
            json.dumps({"artifacts": []}),
        ]

        result = await stage.execute(mock_client, "Build API", {})
        assert result.success is True

    def test_set_substep_callback(self):
        """set_substep_callback stores the callback."""
        stage = ContextStage()
        assert stage._substep_cb is None

        async def cb(phase: str) -> None:
            pass

        stage.set_substep_callback(cb)
        assert stage._substep_cb is cb

        stage.set_substep_callback(None)
        assert stage._substep_cb is None


class TestGenerationStrategy:
    """Test generation_strategy property and _generate_structured routing."""

    def test_default_strategy_is_single_pass(self):
        """Default generation_strategy should be 'single_pass'."""
        stage = ContextStage()
        assert stage.generation_strategy == "single_pass"

    def test_roadmap_risk_is_single_pass(self):
        stage = RoadmapRiskStage()
        assert stage.generation_strategy == "single_pass"

    @pytest.mark.asyncio
    async def test_single_pass_strategy_tries_json_first(self):
        """Single-pass strategy tries JSON directly (1 LLM call on success)."""
        stage = ContextStage()
        mock_client = AsyncMock()
        valid_json = '{"project_description": "Test"}'
        mock_client.generate.return_value = valid_json

        result = await stage._generate_structured(
            mock_client,
            [{"role": "user", "content": "test"}],
            '{"schema": "here"}',
        )
        assert result == valid_json
        assert mock_client.generate.call_count == 1


class TestGatheredContextCap:
    """Test _get_gathered_context caps large context strings."""

    def test_short_context_unchanged(self):
        stage = ContextStage()
        ctx = "Short context"
        result = stage._get_gathered_context({"_gathered_context": ctx})
        assert result == ctx

    def test_empty_context(self):
        stage = ContextStage()
        result = stage._get_gathered_context({})
        assert result == ""

    def test_long_context_trimmed(self):
        stage = ContextStage()
        long_ctx = "x" * 20000
        result = stage._get_gathered_context({"_gathered_context": long_ctx})
        assert len(result) < 20000
        assert result.endswith("[... context trimmed for brevity]")
        assert result.startswith("x" * 100)

    def test_context_at_limit_unchanged(self):
        stage = ContextStage()
        ctx = "x" * 16000
        result = stage._get_gathered_context({"_gathered_context": ctx})
        assert result == ctx

    def test_context_just_over_limit_trimmed(self):
        stage = ContextStage()
        ctx = "x" * 16001
        result = stage._get_gathered_context({"_gathered_context": ctx})
        assert len(result) < 16100  # 16000 + trim message
        assert "[... context trimmed for brevity]" in result


class TestRawSummaries:
    """Test _get_raw_summaries for detailed per-file context."""

    def test_returns_raw_summaries(self):
        stage = ContextStage()
        raw = "### file.py\nSummary of file"
        result = stage._get_raw_summaries({"_raw_summaries": raw})
        assert result == raw

    def test_falls_back_to_gathered_context(self):
        stage = ContextStage()
        result = stage._get_raw_summaries({"_gathered_context": "synthesized context"})
        assert result == "synthesized context"

    def test_empty_raw_falls_back(self):
        stage = ContextStage()
        result = stage._get_raw_summaries({"_raw_summaries": "", "_gathered_context": "fallback"})
        assert result == "fallback"

    def test_long_raw_trimmed(self):
        stage = ContextStage()
        long_raw = "x" * 28000
        result = stage._get_raw_summaries({"_raw_summaries": long_raw})
        assert len(result) < 28000
        assert "[... summaries trimmed for brevity]" in result

    def test_raw_at_limit_unchanged(self):
        stage = ContextStage()
        raw = "x" * 24000
        result = stage._get_raw_summaries({"_raw_summaries": raw})
        assert result == raw


class TestSelfCritique:
    """Test the _self_critique method on PipelineStage."""

    @pytest.mark.asyncio
    async def test_returns_refined_reasoning(self):
        """Self-critique returns the refined reasoning text."""
        stage = ContextStage()
        mock_client = AsyncMock()
        mock_client.generate.return_value = "Refined and improved reasoning about the project."

        result = await stage._self_critique(
            mock_client, "Original reasoning text.", "Build an API",
        )
        assert result == "Refined and improved reasoning about the project."

    @pytest.mark.asyncio
    async def test_includes_codebase_context(self):
        """When krag_context is provided, it appears in the critique prompt."""
        stage = ContextStage()
        mock_client = AsyncMock()
        mock_client.generate.return_value = "Refined reasoning with codebase awareness."

        await stage._self_critique(
            mock_client, "Original.", "Build API",
            krag_context="## Files\n- src/api.py: REST endpoints",
        )
        prompt = mock_client.generate.call_args.kwargs["messages"][1]["content"]
        assert "src/api.py" in prompt
        assert "ACTUAL CODEBASE" in prompt

    @pytest.mark.asyncio
    async def test_falls_back_on_error(self):
        """On LLM error, returns original reasoning unchanged."""
        stage = ContextStage()
        mock_client = AsyncMock()
        mock_client.generate.side_effect = RuntimeError("Connection failed")

        result = await stage._self_critique(
            mock_client, "Original reasoning.", "Build API",
        )
        assert result == "Original reasoning."

    @pytest.mark.asyncio
    async def test_falls_back_on_short_output(self):
        """If critique is too short (likely error/refusal), keeps original."""
        stage = ContextStage()
        mock_client = AsyncMock()
        mock_client.generate.return_value = "OK"  # Way too short

        original = "A" * 100  # 100 chars, critique needs >30 chars (30%)
        result = await stage._self_critique(mock_client, original, "Build API")
        assert result == original

    @pytest.mark.asyncio
    async def test_reports_critiquing_substep(self):
        """Self-critique reports a 'critiquing' substep."""
        stage = ContextStage()
        reported = []

        async def track(phase: str) -> None:
            reported.append(phase)

        stage.set_substep_callback(track)
        mock_client = AsyncMock()
        mock_client.generate.return_value = "Refined reasoning text here."

        await stage._self_critique(mock_client, "Original.", "Build API")
        assert "context:critiquing" in reported
