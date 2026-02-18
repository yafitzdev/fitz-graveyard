# tests/unit/test_schemas.py
"""Tests for planning stage Pydantic schemas."""

import pytest
from datetime import datetime

from fitz_graveyard.planning.schemas import (
    ContextOutput,
    ArchitectureOutput,
    Approach,
    DesignOutput,
    ADR,
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


class TestPromptLoading:
    """Tests for prompt loading utilities."""

    def test_load_context_prompt(self):
        """Test loading context stage prompt."""
        prompt = load_prompt("context")
        assert "Do NOT write any code" in prompt
        assert "{description}" in prompt
        assert "project_description" in prompt

    def test_load_architecture_prompt(self):
        """Test loading architecture stage prompt."""
        prompt = load_prompt("architecture")
        assert "Do NOT write any code" in prompt
        assert "{context}" in prompt
        assert "architectural approaches" in prompt.lower()

    def test_load_architecture_format_prompt(self):
        """Test loading architecture formatting prompt."""
        prompt = load_prompt("architecture_format")
        assert "{reasoning}" in prompt
        assert "{schema}" in prompt

    def test_load_design_prompt(self):
        """Test loading design stage prompt."""
        prompt = load_prompt("design")
        assert "Do NOT write any code" in prompt
        assert "{context}" in prompt
        assert "{architecture}" in prompt
        assert "ADR" in prompt

    def test_load_roadmap_prompt(self):
        """Test loading roadmap stage prompt."""
        prompt = load_prompt("roadmap")
        assert "Do NOT write any code" in prompt
        assert "{context}" in prompt
        assert "{architecture}" in prompt
        assert "{design}" in prompt
        assert "phases" in prompt.lower()

    def test_load_risk_prompt(self):
        """Test loading risk stage prompt."""
        prompt = load_prompt("risk")
        assert "Do NOT write any code" in prompt
        assert "{context}" in prompt
        assert "{roadmap}" in prompt
        assert "mitigation" in prompt.lower()

    def test_nonexistent_prompt_raises(self):
        """Test that loading nonexistent prompt raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_stage")
