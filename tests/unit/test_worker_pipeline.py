# tests/unit/test_worker_pipeline.py
"""
Integration tests for BackgroundWorker + PlanningPipeline.

Tests cover:
    - Worker using pipeline for plan generation
    - Progress updates during pipeline execution
    - Confidence scoring integration
    - Plan output file generation
    - Pipeline failure handling
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from fitz_graveyard.background.worker import BackgroundWorker
from fitz_graveyard.config.schema import FitzPlannerConfig
from fitz_graveyard.models.jobs import JobRecord, JobState
from fitz_graveyard.models.sqlite_store import SQLiteJobStore


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> SQLiteJobStore:
    """Create a temporary SQLite store for testing."""
    db_path = str(tmp_path / "test_worker_pipeline.db")
    store = SQLiteJobStore(db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_worker_uses_pipeline(store: SQLiteJobStore, tmp_path: Path):
    """Test that worker uses PlanningPipeline when OllamaClient configured."""
    # Create config with output directory
    config = FitzPlannerConfig()
    config.output.plans_dir = str(tmp_path / "plans")

    # Create mock OllamaClient
    mock_client = AsyncMock()
    mock_client.health_check = AsyncMock(return_value=True)
    mock_client.generate = AsyncMock(return_value="yes")  # For confidence scorer

    # Mock pipeline result (outputs are dicts from model_dump())
    mock_pipeline_result = MagicMock()
    mock_pipeline_result.success = True
    mock_pipeline_result.git_sha = "abc1234"
    mock_pipeline_result.outputs = {
        "context": {
            "project_description": "Test project",
            "key_requirements": [],
            "constraints": [],
            "existing_context": "",
            "stakeholders": [],
            "scope_boundaries": {},
        },
        "architecture": {
            "approaches": [],
            "recommended": "Test approach",
            "reasoning": "Test reasoning",  # Note: it's "reasoning" not "rationale"
            "key_tradeoffs": {},
            "technology_considerations": [],
        },
        "design": {
            "adrs": [],
            "components": [],
            "data_model": {},
            "integration_points": [],
        },
        "roadmap": {
            "phases": [],
            "critical_path": [],
            "parallel_opportunities": [],
            "total_phases": 0,
        },
        "risk": {"risks": []},
    }

    # Add a queued job
    job = JobRecord(
        job_id="test_job",
        description="Create API endpoints",
        timeline="2 weeks",
        context="FastAPI backend",
        integration_points=["database"],
        state=JobState.QUEUED,
        progress=0.0,
        current_phase=None,
        quality_score=None,
        created_at=datetime.now(timezone.utc),
    )
    await store.add(job)

    # Create worker with mocked pipeline execution
    worker = BackgroundWorker(
        store, config=config, poll_interval=0.1, ollama_client=mock_client
    )

    # Mock the pipeline execute method
    with patch.object(
        worker._pipeline, "execute", return_value=mock_pipeline_result
    ) as mock_execute:
        await worker.start()

        # Wait for job to complete
        await asyncio.sleep(0.5)

        # Stop worker
        await worker.stop()

        # Verify pipeline was called
        mock_execute.assert_called_once()
        args, kwargs = mock_execute.call_args
        assert kwargs["job_id"] == "test_job"
        assert kwargs["job_description"] == "Create API endpoints"
        assert "progress_callback" in kwargs

    # Verify job is complete
    result = await store.get("test_job")
    assert result is not None
    assert result.state == JobState.COMPLETE
    assert result.progress == 1.0
    assert result.file_path is not None
    assert ".md" in result.file_path
    assert result.quality_score is not None


@pytest.mark.asyncio
async def test_worker_progress_updates(store: SQLiteJobStore, tmp_path: Path):
    """Test that worker updates job progress during pipeline execution."""
    # Create config
    config = FitzPlannerConfig()
    config.output.plans_dir = str(tmp_path / "plans")

    # Create mock OllamaClient
    mock_client = AsyncMock()
    mock_client.health_check = AsyncMock(return_value=True)
    mock_client.generate = AsyncMock(return_value="yes")  # For confidence scorer

    # Mock pipeline that calls progress callback
    async def mock_execute(client, job_id, job_description, resume, progress_callback):
        # Simulate stage progress updates
        if progress_callback:
            await progress_callback(0.1, "context")
            await asyncio.sleep(0.05)
            await progress_callback(0.25, "context_complete")
            await asyncio.sleep(0.05)
            await progress_callback(0.45, "architecture_complete")

        # Return result
        result = MagicMock()
        result.success = True
        result.git_sha = "abc1234"
        result.outputs = {
            "context": {"project_description": "Test", "key_requirements": [], "constraints": [], "existing_context": "", "stakeholders": [], "scope_boundaries": {}},
            "architecture": {"approaches": [], "recommended": "Test", "reasoning": "Test", "key_tradeoffs": {}, "technology_considerations": []},
            "design": {"adrs": [], "components": [], "data_model": {}, "integration_points": []},
            "roadmap": {"phases": [], "critical_path": [], "parallel_opportunities": [], "total_phases": 0},
            "risk": {"risks": []},
        }
        return result

    # Add a queued job
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

    # Create worker
    worker = BackgroundWorker(
        store, config=config, poll_interval=0.1, ollama_client=mock_client
    )

    # Mock pipeline execute
    with patch.object(worker._pipeline, "execute", side_effect=mock_execute):
        await worker.start()
        await asyncio.sleep(0.5)
        await worker.stop()

    # Verify job completed
    result = await store.get("test_job")
    assert result is not None
    assert result.state == JobState.COMPLETE


@pytest.mark.asyncio
async def test_worker_pipeline_failure(store: SQLiteJobStore, tmp_path: Path):
    """Test that worker handles pipeline failures gracefully."""
    # Create config
    config = FitzPlannerConfig()
    config.output.plans_dir = str(tmp_path / "plans")

    # Create mock OllamaClient
    mock_client = AsyncMock()
    mock_client.health_check = AsyncMock(return_value=True)

    # Mock pipeline result with failure
    mock_pipeline_result = MagicMock()
    mock_pipeline_result.success = False
    mock_pipeline_result.failed_stage = "architecture"
    mock_pipeline_result.error = "Failed to parse JSON"

    # Add a queued job
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

    # Create worker
    worker = BackgroundWorker(
        store, config=config, poll_interval=0.1, ollama_client=mock_client
    )

    # Mock pipeline execute
    with patch.object(worker._pipeline, "execute", return_value=mock_pipeline_result):
        await worker.start()
        await asyncio.sleep(0.5)
        await worker.stop()

    # Verify job failed
    result = await store.get("test_job")
    assert result is not None
    assert result.state == JobState.FAILED
    assert "architecture" in result.error
    assert "Failed to parse JSON" in result.error


@pytest.mark.asyncio
async def test_worker_without_pipeline_stub(store: SQLiteJobStore):
    """Test worker stub behavior when no pipeline configured."""
    # Add a queued job
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

    # Create worker without OllamaClient (no pipeline)
    worker = BackgroundWorker(store, poll_interval=0.1, ollama_client=None)
    await worker.start()
    await asyncio.sleep(0.5)
    await worker.stop()

    # Verify job completed with stub
    result = await store.get("test_job")
    assert result is not None
    assert result.state == JobState.COMPLETE
    assert result.progress == 1.0
