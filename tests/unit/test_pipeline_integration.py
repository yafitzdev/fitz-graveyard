# tests/unit/test_pipeline_integration.py
"""
Full pipeline integration tests with mock LLM.

Tests cover:
    - End-to-end pipeline execution with all stages
    - Confidence scoring integration
    - Plan output validation
    - Markdown rendering
    - Checkpoint recovery integration
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fitz_planner_mcp.models.sqlite_store import SQLiteJobStore
from fitz_planner_mcp.planning.confidence.flagging import SectionFlagger
from fitz_planner_mcp.planning.confidence.scorer import ConfidenceScorer
from fitz_planner_mcp.planning.pipeline.checkpoint import CheckpointManager
from fitz_planner_mcp.planning.pipeline.orchestrator import PlanningPipeline
from fitz_planner_mcp.planning.pipeline.output import PlanRenderer
from fitz_planner_mcp.planning.pipeline.stages import DEFAULT_STAGES
from fitz_planner_mcp.planning.schemas.plan_output import PlanOutput


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> SQLiteJobStore:
    """Create a temporary SQLite store for testing."""
    db_path = str(tmp_path / "test_pipeline_integration.db")
    store = SQLiteJobStore(db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_full_pipeline_execution(store: SQLiteJobStore):
    """Test full pipeline execution with mock LLM."""
    # Add a test job to the store
    from fitz_planner_mcp.models.jobs import JobRecord, JobState
    from datetime import datetime, timezone

    job = JobRecord(
        job_id="test_job",
        description="Create REST API service",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.QUEUED,
        progress=0.0,
        current_phase=None,
        quality_score=None,
        created_at=datetime.now(timezone.utc),
    )
    await store.add(job)

    # Create mock LLM client
    mock_client = AsyncMock()

    # Mock responses for each stage
    mock_responses = {
        "context": '{"project_description": "API service", "key_requirements": [], "constraints": [], "existing_context": "", "stakeholders": [], "scope_boundaries": {}}',
        "architecture_reasoning": "We should use microservices because...",
        "architecture_format": '{"approaches": [{"name": "Monolith", "description": "Single app", "pros": ["Simple"], "cons": ["Scale"]}], "recommended": "Monolith", "reasoning": "Best for now"}',
        "design": '{"adrs": [], "components": [], "data_model": {}, "integration_points": []}',
        "roadmap": '{"phases": [], "critical_path": [], "parallel_opportunities": [], "total_phases": 0}',
        "risk": '{"risks": []}',
    }

    call_count = [0]

    async def mock_generate_chat(messages):
        # Return appropriate response based on call count
        call_count[0] += 1
        stage_responses = list(mock_responses.values())
        if call_count[0] <= len(stage_responses):
            content = stage_responses[call_count[0] - 1]
        else:
            content = "{}"

        result = MagicMock()
        result.content = content
        return result

    mock_client.generate_chat = mock_generate_chat

    # Create pipeline
    checkpoint_mgr = CheckpointManager(store._db_path)
    pipeline = PlanningPipeline(DEFAULT_STAGES, checkpoint_mgr)

    # Execute pipeline
    result = await pipeline.execute(
        client=mock_client,
        job_id="test_job",
        job_description="Create REST API service",
        resume=False,
    )

    # Verify success
    assert result.success is True
    assert result.failed_stage is None
    assert result.error is None

    # Verify all stages completed
    assert "context" in result.outputs
    assert "architecture" in result.outputs
    assert "design" in result.outputs
    assert "roadmap" in result.outputs
    assert "risk" in result.outputs

    # Verify git SHA captured
    assert result.git_sha is not None or result.git_sha == ""


@pytest.mark.asyncio
async def test_pipeline_with_confidence_scoring(store: SQLiteJobStore):
    """Test pipeline integration with confidence scoring."""
    # Add a test job to the store
    from fitz_planner_mcp.models.jobs import JobRecord, JobState
    from datetime import datetime, timezone

    job = JobRecord(
        job_id="test_job",
        description="Test project",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.QUEUED,
        progress=0.0,
        current_phase=None,
        quality_score=None,
        created_at=datetime.now(timezone.utc),
    )
    await store.add(job)

    # Create mock LLM client
    mock_client = AsyncMock()

    # Simple mock response
    async def mock_generate_chat(messages):
        result = MagicMock()
        result.content = '{"project_description": "Test", "key_requirements": [], "constraints": [], "existing_context": "", "stakeholders": [], "scope_boundaries": {}}'
        return result

    mock_client.generate_chat = mock_generate_chat
    mock_client.generate = AsyncMock(return_value="yes")  # For scorer

    # Create pipeline
    checkpoint_mgr = CheckpointManager(store._db_path)
    pipeline = PlanningPipeline(DEFAULT_STAGES[:1], checkpoint_mgr)  # Just context stage

    # Execute pipeline
    result = await pipeline.execute(
        client=mock_client,
        job_id="test_job",
        job_description="Test project",
        resume=False,
    )

    assert result.success is True

    # Create confidence scorer
    scorer = ConfidenceScorer(mock_client)

    # Score the context output
    context_str = str(result.outputs["context"])
    score = await scorer.score_section("context", context_str)

    # Verify score is in valid range
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_plan_output_creation_and_rendering():
    """Test PlanOutput creation and markdown rendering."""
    # Import schemas
    from fitz_planner_mcp.planning.schemas.context import ContextOutput
    from fitz_planner_mcp.planning.schemas.architecture import ArchitectureOutput, Approach
    from fitz_planner_mcp.planning.schemas.design import DesignOutput, ADR, ComponentDesign
    from fitz_planner_mcp.planning.schemas.roadmap import RoadmapOutput, Phase
    from fitz_planner_mcp.planning.schemas.risk import RiskOutput, Risk

    # Create PlanOutput with all required fields using actual schema models
    plan_output = PlanOutput(
        context=ContextOutput(
            project_description="REST API service",
            key_requirements=["Authentication", "Rate limiting"],
            constraints=["Must use Python"],
            existing_context="",
            stakeholders=["Engineering team"],
            scope_boundaries={},
        ),
        architecture=ArchitectureOutput(
            approaches=[
                Approach(
                    name="Monolith",
                    description="Single application",
                    pros=["Simple deployment"],
                    cons=["Hard to scale"],
                    complexity="low",
                    best_for=["Small teams"],
                )
            ],
            recommended="Monolith",
            reasoning="Best for initial version",
            key_tradeoffs={"simplicity": "vs scalability"},
            technology_considerations=["Python", "FastAPI"],
        ),
        design=DesignOutput(
            adrs=[
                ADR(
                    title="Use FastAPI",
                    context="Need web framework",
                    decision="FastAPI framework",
                    rationale="Modern, fast",
                    alternatives=["Flask", "Django"],
                )
            ],
            components=[
                ComponentDesign(
                    name="AuthService",
                    purpose="Authentication",
                    responsibilities=["Handle authentication"],
                    interfaces=["POST /login"],
                    dependencies=[],
                )
            ],
            data_model={"User": ["id", "email"], "Session": ["id", "user_id"]},
            integration_points=[],
        ),
        roadmap=RoadmapOutput(
            phases=[
                Phase(
                    number=1,
                    name="MVP",
                    objective="Core functionality",
                    tasks=["Setup project", "Add auth"],
                    dependencies=[],
                    duration="2 weeks",
                    deliverables=[],
                )
            ],
            critical_path=[1],
            parallel_opportunities=[],
        ),
        risk=RiskOutput(
            risks=[
                Risk(
                    title="Scale issues",
                    category="technical",
                    description="May not scale well",
                    impact="high",
                    likelihood="medium",
                    mitigation="Monitor performance",
                    contingency="Migrate to microservices",
                    affected_phases=[1],
                )
            ]
        ),
        section_scores={"context": 0.9, "architecture": 0.85},
        overall_quality_score=0.87,
        git_sha="abc1234",
        generated_at=datetime.now(),
    )

    # Render to markdown
    renderer = PlanRenderer()
    markdown = renderer.render(plan_output)

    # Verify markdown structure
    assert "---" in markdown  # Frontmatter
    assert "generated_at:" in markdown
    assert "git_sha:" in markdown
    assert "overall_quality_score:" in markdown
    assert "# Project:" in markdown
    assert "## Context" in markdown
    assert "## Architecture" in markdown
    assert "## Design" in markdown
    assert "## Roadmap" in markdown
    assert "## Risk Analysis" in markdown
    assert "REST API service" in markdown
    assert "Authentication" in markdown
    assert "Monolith" in markdown
    assert "FastAPI" in markdown


@pytest.mark.asyncio
async def test_section_flagging():
    """Test section flagging for low-confidence sections."""
    # Create flagger
    flagger = SectionFlagger(default_threshold=0.7, security_threshold=0.9)

    # Test regular section above threshold
    is_flagged, reason = flagger.flag_section("design", 0.8)
    assert is_flagged is False

    # Test regular section below threshold
    is_flagged, reason = flagger.flag_section("design", 0.6)
    assert is_flagged is True
    assert "0.6" in reason

    # Test security section below security threshold
    is_flagged, reason = flagger.flag_section("security", 0.85)
    assert is_flagged is True
    assert "Security-sensitive" in reason

    # Test security section above security threshold
    is_flagged, reason = flagger.flag_section("security", 0.95)
    assert is_flagged is False


@pytest.mark.asyncio
async def test_overall_quality_score():
    """Test overall quality score computation."""
    flagger = SectionFlagger()

    section_scores = {
        "context": 0.9,
        "architecture": 0.85,
        "design": 0.8,
        "roadmap": 0.75,
        "risk": 0.7,
    }

    overall = flagger.compute_overall_score(section_scores)

    # Verify average
    expected = (0.9 + 0.85 + 0.8 + 0.75 + 0.7) / 5
    assert abs(overall - expected) < 0.01


@pytest.mark.asyncio
async def test_checkpoint_recovery_with_pipeline(store: SQLiteJobStore):
    """Test checkpoint recovery integration with pipeline."""
    # Add a test job to the store
    from fitz_planner_mcp.models.jobs import JobRecord, JobState
    from datetime import datetime, timezone

    job = JobRecord(
        job_id="test_job",
        description="Test project",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.QUEUED,
        progress=0.0,
        current_phase=None,
        quality_score=None,
        created_at=datetime.now(timezone.utc),
    )
    await store.add(job)

    # Create mock LLM client
    mock_client = AsyncMock()

    # Mock response
    async def mock_generate_chat(messages):
        result = MagicMock()
        result.content = '{"project_description": "Test", "requirements": [], "constraints": [], "stakeholders": []}'
        return result

    mock_client.generate_chat = mock_generate_chat

    # Create pipeline
    checkpoint_mgr = CheckpointManager(store._db_path)
    pipeline = PlanningPipeline(DEFAULT_STAGES[:1], checkpoint_mgr)

    # Execute pipeline (first time)
    result1 = await pipeline.execute(
        client=mock_client,
        job_id="test_job",
        job_description="Test project",
        resume=False,
    )

    assert result1.success is True

    # Execute again with resume=True (should skip completed stages)
    result2 = await pipeline.execute(
        client=mock_client,
        job_id="test_job",
        job_description="Test project",
        resume=True,
    )

    assert result2.success is True
    assert result2.outputs == result1.outputs
