# tests/unit/test_tools_integration.py
"""
Integration tests for tool implementations with SQLiteJobStore.

Tests the tools (not the MCP wrappers) against persistent storage.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from pathlib import Path

from fitz_planner_mcp.config.loader import load_config
from fitz_planner_mcp.models.jobs import JobRecord, JobState, generate_job_id
from fitz_planner_mcp.models.sqlite_store import SQLiteJobStore
from fitz_planner_mcp.tools.check_status import check_status
from fitz_planner_mcp.tools.create_plan import create_plan
from fitz_planner_mcp.tools.get_plan import get_plan
from fitz_planner_mcp.tools.list_plans import list_plans
from fitz_planner_mcp.tools.retry_job import retry_job
from fastmcp.exceptions import ToolError


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    """Create a temporary SQLite store for testing."""
    db_path = tmp_path / "test_jobs.db"
    s = SQLiteJobStore(str(db_path))
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def config():
    """Load test configuration."""
    return load_config()


@pytest.mark.asyncio
async def test_create_plan_persists(store, config):
    """Test that create_plan persists jobs to SQLite."""
    result = await create_plan(
        description="Build a test feature",
        timeline="1 week",
        context="Testing context",
        integration_points=["api1", "api2"],
        api_review=False,
        store=store,
        config=config,
    )

    # Verify response
    assert result["status"] == "queued"
    assert "job_id" in result
    job_id = result["job_id"]

    # Verify persistence
    record = await store.get(job_id)
    assert record is not None
    assert record.description == "Build a test feature"
    assert record.timeline == "1 week"
    assert record.context == "Testing context"
    assert record.integration_points == ["api1", "api2"]
    assert record.state == JobState.QUEUED
    assert record.progress == 0.0


@pytest.mark.asyncio
async def test_check_status_queued(store, config):
    """Test check_status for a queued job."""
    # Create a job
    result = await create_plan(
        description="Test job",
        timeline=None,
        context=None,
        integration_points=None,
        api_review=False,
        store=store,
        config=config,
    )
    job_id = result["job_id"]

    # Check status
    status = await check_status(job_id, store=store)

    assert status["job_id"] == job_id
    assert status["state"] == "queued"
    assert status["progress"] == 0.0
    assert "queued" in status["message"].lower()


@pytest.mark.asyncio
async def test_check_status_interrupted(store):
    """Test check_status for an interrupted job."""
    # Create and interrupt a job
    job_id = generate_job_id()
    record = JobRecord(
        job_id=job_id,
        description="Test interrupted job",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.INTERRUPTED,
        progress=0.3,
        current_phase="phase1",
        quality_score=None,
        created_at=datetime.now(timezone.utc),
        file_path=None,
        error="Server restart",
    )
    await store.add(record)

    # Check status
    status = await check_status(job_id, store=store)

    assert status["state"] == "interrupted"
    assert "interrupted" in status["message"].lower()
    assert "retry_job" in status["message"].lower()
    assert status["error"] == "Server restart"


@pytest.mark.asyncio
async def test_check_status_awaiting_review(store):
    """Test check_status for a job awaiting API review confirmation."""
    # Create a job awaiting review with cost estimate
    job_id = generate_job_id()
    record = JobRecord(
        job_id=job_id,
        description="Test awaiting review job",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.AWAITING_REVIEW,
        progress=0.5,
        current_phase="review_paused",
        quality_score=None,
        created_at=datetime.now(timezone.utc),
        file_path=None,
        api_review=True,
        cost_estimate_json='{"total_cost_usd": 0.25, "breakdown": {"openai": 0.25}}',
    )
    await store.add(record)

    # Check status
    status = await check_status(job_id, store=store)

    assert status["state"] == "awaiting_review"
    assert "awaiting" in status["message"].lower()
    assert "confirm_review" in status["message"].lower()
    assert "cancel_review" in status["message"].lower()
    assert "$0.2500" in status["message"]  # Cost should be in message
    assert status["cost_estimate"] is not None
    assert status["cost_estimate"]["total_cost_usd"] == 0.25


@pytest.mark.asyncio
async def test_list_plans_with_multiple(store, config):
    """Test list_plans returns all jobs."""
    # Create 3 jobs
    job_ids = []
    for i in range(3):
        result = await create_plan(
            description=f"Test job {i}",
            timeline=None,
            context=None,
            integration_points=None,
            api_review=False,
            store=store,
            config=config,
        )
        job_ids.append(result["job_id"])

    # List plans
    result = await list_plans(store=store)

    assert result["total"] == 3
    assert len(result["plans"]) == 3

    # Verify job IDs present
    plan_ids = [p["job_id"] for p in result["plans"]]
    for job_id in job_ids:
        assert job_id in plan_ids


@pytest.mark.asyncio
async def test_retry_failed_job(store):
    """Test retry_job for a failed job."""
    # Create a failed job
    job_id = generate_job_id()
    record = JobRecord(
        job_id=job_id,
        description="Failed job",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.FAILED,
        progress=0.5,
        current_phase="phase2",
        quality_score=None,
        created_at=datetime.now(timezone.utc),
        file_path=None,
        error="Something went wrong",
    )
    await store.add(record)

    # Retry the job
    result = await retry_job(job_id, store=store)

    assert result["job_id"] == job_id
    assert result["status"] == "re-queued"
    assert "re-queued" in result["message"].lower()

    # Verify job state updated
    updated = await store.get(job_id)
    assert updated.state == JobState.QUEUED
    assert updated.progress == 0.0
    assert updated.error is None
    assert updated.current_phase is None


@pytest.mark.asyncio
async def test_retry_interrupted_job(store):
    """Test retry_job for an interrupted job."""
    # Create an interrupted job
    job_id = generate_job_id()
    record = JobRecord(
        job_id=job_id,
        description="Interrupted job",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.INTERRUPTED,
        progress=0.7,
        current_phase="phase3",
        quality_score=None,
        created_at=datetime.now(timezone.utc),
        file_path=None,
        error="Server shutdown",
    )
    await store.add(record)

    # Retry the job
    result = await retry_job(job_id, store=store)

    assert result["status"] == "re-queued"

    # Verify state reset
    updated = await store.get(job_id)
    assert updated.state == JobState.QUEUED
    assert updated.progress == 0.0
    assert updated.error is None


@pytest.mark.asyncio
async def test_retry_running_job_fails(store):
    """Test that retrying a running job raises ToolError."""
    # Create a running job
    job_id = generate_job_id()
    record = JobRecord(
        job_id=job_id,
        description="Running job",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.RUNNING,
        progress=0.5,
        current_phase="phase2",
        quality_score=None,
        created_at=datetime.now(timezone.utc),
        file_path=None,
    )
    await store.add(record)

    # Attempt retry - should fail
    with pytest.raises(ToolError) as exc_info:
        await retry_job(job_id, store=store)

    assert "running" in str(exc_info.value).lower()
    assert "failed or interrupted" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_retry_queued_job_fails(store, config):
    """Test that retrying a queued job raises ToolError."""
    # Create a queued job
    result = await create_plan(
        description="Queued job",
        timeline=None,
        context=None,
        integration_points=None,
        api_review=False,
        store=store,
        config=config,
    )
    job_id = result["job_id"]

    # Attempt retry - should fail
    with pytest.raises(ToolError) as exc_info:
        await retry_job(job_id, store=store)

    assert "queued" in str(exc_info.value).lower()
    assert "failed or interrupted" in str(exc_info.value).lower()
