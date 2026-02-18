# tests/unit/test_cancel_review.py
"""
Tests for cancel_review tool.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone

from fitz_planner_mcp.models.jobs import JobRecord, JobState, generate_job_id
from fitz_planner_mcp.models.jobs import InMemoryJobStore
from fitz_planner_mcp.tools.cancel_review import cancel_review
from fastmcp.exceptions import ToolError


@pytest_asyncio.fixture
async def store():
    """Create an in-memory store for testing."""
    return InMemoryJobStore()


@pytest.mark.asyncio
async def test_cancel_review_success(store):
    """Test cancel_review marks job as complete without review."""
    # Create a job awaiting review
    job_id = generate_job_id()
    record = JobRecord(
        job_id=job_id,
        description="Test job",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.AWAITING_REVIEW,
        progress=0.5,
        current_phase="review_paused",
        quality_score=None,
        created_at=datetime.now(timezone.utc),
        api_review=True,
        cost_estimate_json='{"total_cost_usd": 0.15}',
    )
    await store.add(record)

    # Cancel review
    result = await cancel_review(job_id, store=store)

    # Verify response
    assert result["job_id"] == job_id
    assert result["status"] == "review_skipped"
    assert "cancelled" in result["message"].lower()

    # Verify job state updated
    updated = await store.get(job_id)
    assert updated.state == JobState.COMPLETE
    assert updated.current_phase == "review_cancelled"


@pytest.mark.asyncio
async def test_cancel_review_not_found(store):
    """Test cancel_review with non-existent job."""
    with pytest.raises(ToolError) as exc_info:
        await cancel_review("nonexistent", store=store)

    assert "not found" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_cancel_review_wrong_state(store):
    """Test cancel_review with job not in awaiting_review state."""
    # Create a running job
    job_id = generate_job_id()
    record = JobRecord(
        job_id=job_id,
        description="Running job",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.RUNNING,
        progress=0.3,
        current_phase="phase1",
        quality_score=None,
        created_at=datetime.now(timezone.utc),
        api_review=True,
    )
    await store.add(record)

    # Attempt to cancel - should fail
    with pytest.raises(ToolError) as exc_info:
        await cancel_review(job_id, store=store)

    assert "not awaiting review" in str(exc_info.value).lower()
    assert "running" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_cancel_review_invalid_job_id(store):
    """Test cancel_review with invalid job ID format."""
    with pytest.raises(ToolError) as exc_info:
        await cancel_review("", store=store)

    assert "invalid" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()
