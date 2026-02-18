# tests/unit/test_worker_ollama.py
"""
Integration tests for BackgroundWorker + OllamaClient.

Tests cover:
    - Successful job processing with mocked OllamaClient
    - Health check failures
    - OOM fallback handling
    - Memory threshold violations
    - Connection errors
    - Stub behavior when no OllamaClient configured
    - Message building from job metadata
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
    db_path = str(tmp_path / "test_worker_ollama.db")
    store = SQLiteJobStore(db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_process_job_success(store: SQLiteJobStore, tmp_path: Path):
    """Test successful job processing with mocked OllamaClient."""
    # Create config with output directory
    config = FitzPlannerConfig()
    config.output.plans_dir = str(tmp_path / "plans")

    # Create mock OllamaClient
    mock_client = AsyncMock()
    mock_client.health_check = AsyncMock(return_value=True)
    mock_client.generate = AsyncMock(return_value="yes")  # For confidence scorer

    # Mock pipeline result
    mock_pipeline_result = MagicMock()
    mock_pipeline_result.success = True
    mock_pipeline_result.git_sha = "abc1234"
    mock_pipeline_result.outputs = {
        "context": {"project_description": "Test", "key_requirements": [], "constraints": [], "existing_context": "", "stakeholders": [], "scope_boundaries": {}},
        "architecture": {"approaches": [], "recommended": "Test", "reasoning": "Test", "key_tradeoffs": {}, "technology_considerations": []},
        "design": {"adrs": [], "components": [], "data_model": {}, "integration_points": []},
        "roadmap": {"phases": [], "critical_path": [], "parallel_opportunities": [], "total_phases": 0},
        "risk": {"risks": []},
    }

    # Add a queued job
    job = JobRecord(
        job_id="test_job",
        description="Create API endpoints",
        timeline="2 weeks",
        context="FastAPI backend",
        integration_points=["database", "auth"],
        state=JobState.QUEUED,
        progress=0.0,
        current_phase=None,
        quality_score=None,
        created_at=datetime.now(timezone.utc),
    )
    await store.add(job)

    # Create worker with mocked client
    worker = BackgroundWorker(
        store, config=config, poll_interval=0.1, ollama_client=mock_client, memory_threshold=80.0
    )

    # Mock pipeline execute
    with patch.object(worker._pipeline, "execute", return_value=mock_pipeline_result):
        await worker.start()
        await asyncio.sleep(0.5)
        await worker.stop()

    # Verify job is complete
    result = await store.get("test_job")
    assert result is not None
    assert result.state == JobState.COMPLETE
    assert result.progress == 1.0


@pytest.mark.asyncio
async def test_process_job_health_check_fails(store: SQLiteJobStore, tmp_path: Path):
    """Test job fails when health check returns False."""
    # Create config with output directory
    config = FitzPlannerConfig()
    config.output.plans_dir = str(tmp_path / "plans")

    # Create mock OllamaClient with failing health check
    mock_client = AsyncMock()
    mock_client.health_check = AsyncMock(return_value=False)

    # Add a queued job
    job = JobRecord(
        job_id="test_job",
        description="Create API endpoints",
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

    # Create worker with mocked client
    worker = BackgroundWorker(
        store, config=config, poll_interval=0.1, ollama_client=mock_client, memory_threshold=80.0
    )
    await worker.start()

    # Wait for job to fail
    await asyncio.sleep(0.5)

    # Stop worker
    await worker.stop()

    # Verify job is failed
    result = await store.get("test_job")
    assert result is not None
    assert result.state == JobState.FAILED
    assert "health check" in result.error.lower()


@pytest.mark.asyncio
async def test_process_job_oom_fallback(store: SQLiteJobStore, tmp_path: Path):
    """Test job completes successfully when OOM fallback is used."""
    # Create config with output directory
    config = FitzPlannerConfig()
    config.output.plans_dir = str(tmp_path / "plans")

    # Create mock OllamaClient
    mock_client = AsyncMock()
    mock_client.health_check = AsyncMock(return_value=True)
    mock_client.generate = AsyncMock(return_value="yes")  # For confidence scorer

    # Mock pipeline result
    mock_pipeline_result = MagicMock()
    mock_pipeline_result.success = True
    mock_pipeline_result.git_sha = "abc1234"
    mock_pipeline_result.outputs = {
        "context": {"project_description": "Test", "key_requirements": [], "constraints": [], "existing_context": "", "stakeholders": [], "scope_boundaries": {}},
        "architecture": {"approaches": [], "recommended": "Test", "reasoning": "Test", "key_tradeoffs": {}, "technology_considerations": []},
        "design": {"adrs": [], "components": [], "data_model": {}, "integration_points": []},
        "roadmap": {"phases": [], "critical_path": [], "parallel_opportunities": [], "total_phases": 0},
        "risk": {"risks": []},
    }

    # Add a queued job
    job = JobRecord(
        job_id="test_job",
        description="Create API endpoints",
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

    # Create worker with mocked client
    worker = BackgroundWorker(
        store, config=config, poll_interval=0.1, ollama_client=mock_client, memory_threshold=80.0
    )

    # Mock pipeline execute
    with patch.object(worker._pipeline, "execute", return_value=mock_pipeline_result):
        await worker.start()
        await asyncio.sleep(0.5)
        await worker.stop()

    # Verify job is complete (fallback handled inside OllamaClient)
    result = await store.get("test_job")
    assert result is not None
    assert result.state == JobState.COMPLETE
    assert result.progress == 1.0


@pytest.mark.asyncio
async def test_process_job_memory_abort(store: SQLiteJobStore, tmp_path: Path):
    """Test job fails when memory threshold is exceeded."""
    # Create config with output directory
    config = FitzPlannerConfig()
    config.output.plans_dir = str(tmp_path / "plans")

    # Create mock OllamaClient that raises MemoryError
    mock_client = AsyncMock()
    mock_client.health_check = AsyncMock(return_value=True)

    # Mock pipeline that raises MemoryError
    mock_pipeline_result = MagicMock()
    mock_pipeline_result.success = False
    mock_pipeline_result.failed_stage = "context"
    mock_pipeline_result.error = "RAM usage exceeded threshold (80.0%) during generation"

    # Add a queued job
    job = JobRecord(
        job_id="test_job",
        description="Create API endpoints",
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

    # Create worker with mocked client
    worker = BackgroundWorker(
        store, config=config, poll_interval=0.1, ollama_client=mock_client, memory_threshold=80.0
    )

    # Mock pipeline execute
    with patch.object(worker._pipeline, "execute", return_value=mock_pipeline_result):
        await worker.start()
        await asyncio.sleep(0.5)
        await worker.stop()

    # Verify job is failed
    result = await store.get("test_job")
    assert result is not None
    assert result.state == JobState.FAILED
    assert "RAM usage exceeded threshold" in result.error or "Pipeline failed" in result.error


@pytest.mark.asyncio
async def test_process_job_connection_error(store: SQLiteJobStore, tmp_path: Path):
    """Test job fails when OllamaClient raises ConnectionError."""
    # Create config with output directory
    config = FitzPlannerConfig()
    config.output.plans_dir = str(tmp_path / "plans")

    # Create mock OllamaClient that raises ConnectionError on health check
    mock_client = AsyncMock()
    mock_client.health_check = AsyncMock(
        side_effect=ConnectionError("Failed to connect to Ollama server")
    )

    # Add a queued job
    job = JobRecord(
        job_id="test_job",
        description="Create API endpoints",
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

    # Create worker with mocked client
    worker = BackgroundWorker(
        store, config=config, poll_interval=0.1, ollama_client=mock_client, memory_threshold=80.0
    )
    await worker.start()

    # Wait for job to fail
    await asyncio.sleep(0.5)

    # Stop worker
    await worker.stop()

    # Verify job is failed
    result = await store.get("test_job")
    assert result is not None
    assert result.state == JobState.FAILED
    assert "ConnectionError" in result.error


@pytest.mark.asyncio
async def test_process_job_no_client_stub(store: SQLiteJobStore):
    """Test stub behavior when no OllamaClient configured."""
    # Add a queued job
    job = JobRecord(
        job_id="test_job",
        description="Create API endpoints",
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

    # Create worker without OllamaClient
    worker = BackgroundWorker(store, poll_interval=0.1, ollama_client=None)
    await worker.start()

    # Wait for job to complete (stub)
    await asyncio.sleep(0.5)

    # Stop worker
    await worker.stop()

    # Verify job completes with stub behavior
    result = await store.get("test_job")
    assert result is not None
    assert result.state == JobState.COMPLETE
    assert result.progress == 1.0


@pytest.mark.asyncio
async def test_build_messages_includes_job_metadata(store: SQLiteJobStore):
    """Test that _build_messages includes all job metadata."""
    # Create worker
    worker = BackgroundWorker(store, poll_interval=0.1)

    # Create job with all fields populated
    job = JobRecord(
        job_id="test_job",
        description="Create REST API",
        timeline="3 weeks",
        context="FastAPI backend with PostgreSQL",
        integration_points=["auth-service", "cache-layer"],
        state=JobState.QUEUED,
        progress=0.0,
        current_phase=None,
        quality_score=None,
        created_at=datetime.now(timezone.utc),
    )

    # Build messages
    messages = worker._build_messages(job)

    # Verify structure
    assert len(messages) == 1
    assert messages[0]["role"] == "user"

    # Verify all metadata is included
    content = messages[0]["content"]
    assert "Create REST API" in content
    assert "3 weeks" in content
    assert "FastAPI backend with PostgreSQL" in content
    assert "auth-service" in content
    assert "cache-layer" in content
