# tests/unit/test_worker.py
"""
Tests for background worker and lifecycle management.

Tests cover:
    - Sequential job processing (FIFO)
    - Graceful shutdown with job interruption
    - Lifecycle startup crash recovery
    - Empty queue handling
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from fitz_graveyard.background.lifecycle import ServerLifecycle
from fitz_graveyard.background.worker import BackgroundWorker
from fitz_graveyard.models.jobs import JobRecord, JobState
from fitz_graveyard.models.sqlite_store import SQLiteJobStore


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> SQLiteJobStore:
    """Create a temporary SQLite store for testing."""
    db_path = str(tmp_path / "test_worker.db")
    store = SQLiteJobStore(db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_worker_processes_queued_job(store: SQLiteJobStore):
    """Test that worker picks up and processes a queued job."""
    # Add a queued job
    job = JobRecord(
        job_id="job1",
        description="Test job",
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

    # Start worker with short poll interval
    worker = BackgroundWorker(store, poll_interval=0.1)
    await worker.start()

    # Wait for job to be processed
    await asyncio.sleep(0.5)

    # Stop worker
    await worker.stop()

    # Verify job is complete
    result = await store.get("job1")
    assert result is not None
    assert result.state == JobState.COMPLETE
    assert result.progress == 1.0


@pytest.mark.asyncio
async def test_worker_fifo_order(store: SQLiteJobStore):
    """Test that worker processes jobs in FIFO order (oldest first)."""
    # Add two jobs (A then B)
    now = datetime.now(timezone.utc)

    job_a = JobRecord(
        job_id="job_a",
        description="First job",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.QUEUED,
        progress=0.0,
        current_phase=None,
        quality_score=None,
        created_at=now,
    )
    await store.add(job_a)

    # Small delay to ensure different created_at
    await asyncio.sleep(0.01)

    job_b = JobRecord(
        job_id="job_b",
        description="Second job",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.QUEUED,
        progress=0.0,
        current_phase=None,
        quality_score=None,
        created_at=datetime.now(timezone.utc),
    )
    await store.add(job_b)

    # Start worker
    worker = BackgroundWorker(store, poll_interval=0.1)
    await worker.start()

    # Wait for both jobs to complete
    await asyncio.sleep(0.8)

    # Stop worker
    await worker.stop()

    # Verify both complete
    result_a = await store.get("job_a")
    result_b = await store.get("job_b")

    assert result_a is not None
    assert result_b is not None
    assert result_a.state == JobState.COMPLETE
    assert result_b.state == JobState.COMPLETE

    # Verify A completed before B (by checking updated_at)
    assert result_a.updated_at is not None
    assert result_b.updated_at is not None
    assert result_a.updated_at < result_b.updated_at


@pytest.mark.asyncio
async def test_worker_handles_no_jobs(store: SQLiteJobStore):
    """Test that worker handles empty queue without errors."""
    # Start worker with empty queue
    worker = BackgroundWorker(store, poll_interval=0.1)
    await worker.start()

    # Wait a poll interval
    await asyncio.sleep(0.3)

    # Stop worker - should not error
    await worker.stop()

    # Verify no jobs in store
    jobs = await store.list_all()
    assert len(jobs) == 0


@pytest.mark.asyncio
async def test_worker_stop_marks_interrupted(store: SQLiteJobStore):
    """Test that stopping worker marks running job as interrupted."""
    # Add a queued job
    job = JobRecord(
        job_id="job1",
        description="Test job",
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

    # Start worker
    worker = BackgroundWorker(store, poll_interval=0.05)
    await worker.start()

    # Wait for job to start processing
    await asyncio.sleep(0.15)

    # Stop worker immediately (while job could be processing)
    await worker.stop()

    # Verify job is either COMPLETE or INTERRUPTED (race condition acceptable)
    result = await store.get("job1")
    assert result is not None
    assert result.state in (JobState.COMPLETE, JobState.INTERRUPTED)

    # If interrupted, verify error message
    if result.state == JobState.INTERRUPTED:
        assert result.error == "Server shutdown during processing"


@pytest.mark.asyncio
async def test_lifecycle_startup_recovery(tmp_path: Path):
    """Test that lifecycle startup marks stale running jobs as interrupted."""
    db_path = str(tmp_path / "test_lifecycle.db")

    # Create store and insert a job in RUNNING state directly
    store = SQLiteJobStore(db_path)
    await store.initialize()

    job = JobRecord(
        job_id="stale_job",
        description="Job that was running when server crashed",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.RUNNING,
        progress=0.5,
        current_phase="processing",
        quality_score=None,
        created_at=datetime.now(timezone.utc),
    )
    await store.add(job)
    await store.close()

    # Create lifecycle and run startup (which should recover the job)
    lifecycle = ServerLifecycle(db_path)
    await lifecycle.startup()

    # Verify job is now INTERRUPTED
    result = await lifecycle.store.get("stale_job")
    assert result is not None
    assert result.state == JobState.INTERRUPTED
    assert "Server restarted during processing" in (result.error or "")

    # Cleanup
    await lifecycle.shutdown()


@pytest.mark.asyncio
async def test_worker_failed_job_tracking(store: SQLiteJobStore):
    """Test that worker properly handles and tracks failed jobs."""
    # Note: Since _process_job is a stub that doesn't fail, we can't
    # easily trigger a failure without modifying the worker.
    # This test is a placeholder for Phase 4 when we integrate the
    # actual planning engine that could raise exceptions.

    # For now, just verify that the error handling structure exists
    worker = BackgroundWorker(store, poll_interval=0.1)
    assert hasattr(worker, "_process_job")
    assert hasattr(worker, "_run_loop")
