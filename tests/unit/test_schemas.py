# tests/unit/test_schemas.py
"""Tests for planning stage Pydantic schemas."""

import pytest
from datetime import datetime

from fitz_graveyard.planning.schemas import (
    Assumption,
    ContextOutput,
    ArchitectureOutput,
    Approach,
    DesignOutput,
    ADR,
    Artifact,
    ComponentDesign,
    RoadmapOutput,
    Phase,
    RiskOutput,
    Risk,
    PlanOutput,
)
from fitz_graveyard.planning.prompts import load_prompt


class TestContextOutput:
    """Tests for ContextOutput schema."""

    def test_minimal_valid(self):
        """Test creation with minimal required fields."""
        context = ContextOutput(
            project_description="Build a task management app",
        )
        assert context.project_description == "Build a task management app"
        assert context.key_requirements == []
        assert context.constraints == []
        assert context.existing_context == ""

    def test_full_valid(self):
        """Test creation with all fields populated."""
        context = ContextOutput(
            project_description="Build a task management app",
            key_requirements=["User authentication", "Task CRUD"],
            constraints=["Must use PostgreSQL", "Deploy to AWS"],
            existing_context="Existing API v1 with REST endpoints",
            stakeholders=["Product team", "End users"],
            scope_boundaries={
                "in_scope": ["Tasks", "Users"],
                "out_of_scope": ["Billing", "Analytics"],
            },
        )
        assert len(context.key_requirements) == 2
        assert len(context.constraints) == 2
        assert "in_scope" in context.scope_boundaries

    def test_with_assumptions(self):
        """Test creation with assumptions field."""
        context = ContextOutput(
            project_description="Build an API",
            assumptions=[
                Assumption(
                    assumption="REST API, not GraphQL",
                    impact="Architecture changes significantly if GraphQL",
                    confidence="medium",
                ),
                Assumption(
                    assumption="Single-tenant deployment",
                    impact="Data isolation strategy changes for multi-tenant",
                    confidence="low",
                ),
            ],
        )
        assert len(context.assumptions) == 2
        assert context.assumptions[0].assumption == "REST API, not GraphQL"
        assert context.assumptions[1].confidence == "low"

    def test_with_existing_files_and_artifacts(self):
        """Test creation with existing_files and needed_artifacts."""
        context = ContextOutput(
            project_description="Build plugin",
            existing_files=["src/config.py — config loader"],
            needed_artifacts=["openai.yaml — plugin config file"],
        )
        assert len(context.existing_files) == 1
        assert len(context.needed_artifacts) == 1

    def test_existing_files_default_empty(self):
        """existing_files defaults to empty list."""
        context = ContextOutput(project_description="Test")
        assert context.existing_files == []
        assert context.needed_artifacts == []

    def test_assumptions_default_empty(self):
        """Assumptions defaults to empty list."""
        context = ContextOutput(project_description="Test")
        assert context.assumptions == []

    def test_assumption_default_confidence(self):
        """Assumption confidence defaults to medium."""
        a = Assumption(assumption="test", impact="test impact")
        assert a.confidence == "medium"


class TestArchitectureOutput:
    """Tests for ArchitectureOutput schema."""

    def test_minimal_valid(self):
        """Test creation with minimal required fields."""
        arch = ArchitectureOutput(
            recommended="Monolith",
            reasoning="Simplest approach for initial MVP",
        )
        assert arch.recommended == "Monolith"
        assert arch.approaches == []

    def test_scope_statement_default_empty(self):
        """scope_statement defaults to empty string."""
        arch = ArchitectureOutput(recommended="Monolith", reasoning="Simple")
        assert arch.scope_statement == ""

    def test_with_scope_statement(self):
        """Test creation with scope_statement."""
        arch = ArchitectureOutput(
            recommended="Monolith",
            reasoning="Simple",
            scope_statement="This task produces one YAML file.",
        )
        assert arch.scope_statement == "This task produces one YAML file."

    def test_with_approaches(self):
        """Test creation with multiple approaches."""
        approaches = [
            Approach(
                name="Monolith",
                description="Single deployable application",
                pros=["Simple deployment", "Easy development"],
                cons=["Harder to scale"],
                complexity="low",
                best_for=["MVPs", "Small teams"],
            ),
            Approach(
                name="Microservices",
                description="Distributed services",
                pros=["Independent scaling"],
                cons=["Operational complexity"],
                complexity="high",
                best_for=["Large scale", "Multiple teams"],
            ),
        ]
        arch = ArchitectureOutput(
            approaches=approaches,
            recommended="Monolith",
            reasoning="Team size and timeline favor simplicity",
            key_tradeoffs={"simplicity": "vs scalability"},
        )
        assert len(arch.approaches) == 2
        assert arch.approaches[0].name == "Monolith"


class TestDesignOutput:
    """Tests for DesignOutput schema."""

    def test_minimal_valid(self):
        """Test creation with minimal fields."""
        design = DesignOutput()
        assert design.adrs == []
        assert design.components == []
        assert design.data_model == {}

    def test_artifacts_default_empty(self):
        """Artifacts defaults to empty list."""
        design = DesignOutput()
        assert design.artifacts == []

    def test_with_artifacts(self):
        """Test creation with artifacts."""
        design = DesignOutput(
            artifacts=[
                Artifact(
                    filename="openai.yaml",
                    content="provider:\n  name: openai\n",
                    purpose="Plugin config for OpenAI",
                ),
            ],
        )
        assert len(design.artifacts) == 1
        assert design.artifacts[0].filename == "openai.yaml"
        assert "provider:" in design.artifacts[0].content

    def test_with_adrs_and_components(self):
        """Test creation with ADRs and components."""
        adrs = [
            ADR(
                title="Use PostgreSQL for persistence",
                context="Need reliable transactional storage",
                decision="Use PostgreSQL 15",
                rationale="Team expertise and ACID guarantees",
                consequences=["Requires server hosting"],
                alternatives_considered=["SQLite", "MongoDB"],
            )
        ]
        components = [
            ComponentDesign(
                name="TaskService",
                purpose="Manage task lifecycle",
                responsibilities=["Create tasks", "Update status"],
                interfaces=["REST API", "GraphQL"],
                dependencies=["Database", "AuthService"],
            )
        ]
        design = DesignOutput(
            adrs=adrs,
            components=components,
            data_model={"Task": ["id", "title", "status"]},
            integration_points=["Slack API", "Email service"],
        )
        assert len(design.adrs) == 1
        assert len(design.components) == 1
        assert "Task" in design.data_model


class TestRoadmapOutput:
    """Tests for RoadmapOutput schema."""

    def test_minimal_valid(self):
        """Test creation with minimal fields."""
        roadmap = RoadmapOutput()
        assert roadmap.phases == []
        assert roadmap.total_phases == 0

    def test_phase_new_fields_default_empty(self):
        """verification_command and estimated_effort default to empty string."""
        phase = Phase(number=1, name="Test", objective="Test")
        assert phase.verification_command == ""
        assert phase.estimated_effort == ""

    def test_phase_with_verification_and_effort(self):
        """Test Phase with verification_command and estimated_effort."""
        phase = Phase(
            number=1,
            name="Setup",
            objective="Init project",
            verification_command="python -m pytest tests/ -v",
            estimated_effort="~30 min",
        )
        assert phase.verification_command == "python -m pytest tests/ -v"
        assert phase.estimated_effort == "~30 min"

    def test_with_phases_and_dependencies(self):
        """Test creation with phases and dependency graph."""
        phases = [
            Phase(
                number=1,
                name="Foundation",
                objective="Set up core infrastructure",
                deliverables=["Database schema", "API skeleton"],
                dependencies=[],
                estimated_complexity="low",
            ),
            Phase(
                number=2,
                name="Authentication",
                objective="Implement user auth",
                deliverables=["Login", "Registration"],
                dependencies=[1],
                estimated_complexity="medium",
                key_risks=["OAuth integration complexity"],
            ),
        ]
        roadmap = RoadmapOutput(
            phases=phases,
            critical_path=[1, 2],
            parallel_opportunities=[],
            total_phases=2,
        )
        assert len(roadmap.phases) == 2
        assert roadmap.phases[1].dependencies == [1]
        assert roadmap.total_phases == 2


class TestRiskOutput:
    """Tests for RiskOutput schema."""

    def test_minimal_valid(self):
        """Test creation with minimal fields."""
        risk_output = RiskOutput()
        assert risk_output.risks == []
        assert risk_output.overall_risk_level == "medium"

    def test_risk_verification_default_empty(self):
        """verification defaults to empty string."""
        risk = Risk(
            category="technical", description="test",
            impact="high", likelihood="medium", mitigation="test",
        )
        assert risk.verification == ""

    def test_risk_with_verification(self):
        """Test Risk with verification field."""
        risk = Risk(
            category="technical",
            description="API returns null",
            impact="high",
            likelihood="medium",
            mitigation="Add null check",
            verification="assert response['content'] is not None",
        )
        assert "assert" in risk.verification

    def test_with_risks(self):
        """Test creation with identified risks."""
        risks = [
            Risk(
                category="technical",
                description="Third-party API rate limits",
                impact="high",
                likelihood="medium",
                mitigation="Implement caching and backoff",
                contingency="Switch to alternative provider",
                affected_phases=[3, 4],
            ),
            Risk(
                category="resource",
                description="Limited DevOps expertise",
                impact="medium",
                likelihood="high",
                mitigation="Use managed services (PaaS)",
                contingency="Hire consultant",
                affected_phases=[5],
            ),
        ]
        risk_output = RiskOutput(
            risks=risks,
            overall_risk_level="high",
            recommended_contingencies=["Budget for consultants", "Build in buffer time"],
        )
        assert len(risk_output.risks) == 2
        assert risk_output.risks[0].category == "technical"


class TestPlanOutput:
    """Tests for complete PlanOutput schema."""

    def test_minimal_valid(self):
        """Test creation with all required stage outputs."""
        plan = PlanOutput(
            context=ContextOutput(project_description="Test project"),
            architecture=ArchitectureOutput(
                recommended="Monolith",
                reasoning="Simple approach",
            ),
            design=DesignOutput(),
            roadmap=RoadmapOutput(),
            risk=RiskOutput(),
        )
        assert plan.context.project_description == "Test project"
        assert plan.architecture.recommended == "Monolith"
        assert isinstance(plan.generated_at, datetime)

    def test_with_quality_metadata(self):
        """Test creation with quality scores and provenance."""
        plan = PlanOutput(
            context=ContextOutput(project_description="Test project"),
            architecture=ArchitectureOutput(
                recommended="Monolith",
                reasoning="Simple",
            ),
            design=DesignOutput(),
            roadmap=RoadmapOutput(),
            risk=RiskOutput(),
            section_scores={
                "context": 0.95,
                "architecture": 0.88,
                "design": 0.92,
            },
            overall_quality_score=0.91,
            git_sha="abc123def456",
        )
        assert plan.overall_quality_score == 0.91
        assert plan.git_sha == "abc123def456"
        assert len(plan.section_scores) == 3

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored (not rejected)."""
        # This should not raise an error
        plan = PlanOutput(
            context=ContextOutput(project_description="Test"),
            architecture=ArchitectureOutput(recommended="X", reasoning="Y"),
            design=DesignOutput(),
            roadmap=RoadmapOutput(),
            risk=RiskOutput(),
            unknown_field="should be ignored",
        )
        assert plan.context.project_description == "Test"


class TestConfigValidation:
    """Tests for config unknown key warnings."""

    def test_warn_unknown_top_level_key(self, caplog):
        """Unknown top-level key triggers warning."""
        import logging
        from fitz_graveyard.config.loader import _warn_unknown_keys
        from fitz_graveyard.config.schema import FitzPlannerConfig

        yaml_data = {"provider": "ollama", "tiemout": 600}
        with caplog.at_level(logging.WARNING):
            _warn_unknown_keys(yaml_data, FitzPlannerConfig)
        assert any("tiemout" in msg for msg in caplog.messages)

    def test_warn_unknown_nested_key(self, caplog):
        """Unknown nested key triggers warning with full path."""
        import logging
        from fitz_graveyard.config.loader import _warn_unknown_keys
        from fitz_graveyard.config.schema import FitzPlannerConfig

        yaml_data = {"ollama": {"base_url": "http://localhost:11434", "mdoel": "test"}}
        with caplog.at_level(logging.WARNING):
            _warn_unknown_keys(yaml_data, FitzPlannerConfig)
        assert any("ollama.mdoel" in msg for msg in caplog.messages)

    def test_no_warning_for_valid_keys(self, caplog):
        """Valid keys don't trigger warnings."""
        import logging
        from fitz_graveyard.config.loader import _warn_unknown_keys
        from fitz_graveyard.config.schema import FitzPlannerConfig

        yaml_data = {"provider": "ollama", "ollama": {"base_url": "http://localhost:11434"}}
        with caplog.at_level(logging.WARNING):
            _warn_unknown_keys(yaml_data, FitzPlannerConfig)
        assert not any("Unknown config key" in msg for msg in caplog.messages)



class TestPromptLoading:
    """Tests for prompt loading utilities."""

    def test_load_context_prompt(self):
        """Test loading context stage prompt."""
        prompt = load_prompt("context")
        assert "{description}" in prompt
        assert "{krag_context}" in prompt
        assert "requirements" in prompt.lower()

    def test_load_architecture_prompt(self):
        """Test loading architecture stage prompt."""
        prompt = load_prompt("architecture")
        assert "{context}" in prompt
        assert "{krag_context}" in prompt
        assert "approach" in prompt.lower()

    def test_load_architecture_format_prompt(self):
        """Test loading architecture formatting prompt."""
        prompt = load_prompt("architecture_format")
        assert "{reasoning}" in prompt
        assert "{schema}" in prompt

    def test_load_design_prompt(self):
        """Test loading design stage prompt."""
        prompt = load_prompt("design")
        assert "{context}" in prompt
        assert "{architecture}" in prompt
        assert "data model" in prompt.lower()

    def test_load_roadmap_prompt(self):
        """Test loading roadmap stage prompt."""
        prompt = load_prompt("roadmap")
        assert "{context}" in prompt
        assert "{architecture}" in prompt
        assert "{design}" in prompt
        assert "phases" in prompt.lower()

    def test_load_risk_prompt(self):
        """Test loading risk stage prompt."""
        prompt = load_prompt("risk")
        assert "{context}" in prompt
        assert "{roadmap}" in prompt
        assert "risk" in prompt.lower()

    def test_nonexistent_prompt_raises(self):
        """Test that loading nonexistent prompt raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_stage")
