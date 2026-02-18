# tests/unit/test_sqlite_store.py
"""
Unit tests for SQLiteJobStore persistence.

Tests CRUD operations, crash recovery, FIFO ordering, and serialization.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from fitz_graveyard.models.jobs import JobRecord, JobState
from fitz_graveyard.models.sqlite_store import SQLiteJobStore


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> SQLiteJobStore:
    """Create and initialize a test SQLite store."""
    db_path = str(tmp_path / "test_jobs.db")
    store = SQLiteJobStore(db_path)
    await store.initialize()
    return store


@pytest.fixture
def sample_record() -> JobRecord:
    """Create a sample job record for testing."""
    return JobRecord(
        job_id="test-job-001",
        description="Test planning job",
        timeline="Q1 2024",
        context="Test context for planning",
        integration_points=["fitz-ai", "local-llm"],
        state=JobState.QUEUED,
        progress=0.0,
        current_phase=None,
        quality_score=None,
        created_at=datetime.now(timezone.utc),
        file_path=None,
        error=None,
        updated_at=None,
    )


@pytest.mark.asyncio
async def test_add_and_get_roundtrip(store: SQLiteJobStore, sample_record: JobRecord):
    """Test adding a job and retrieving it preserves all fields."""
    await store.add(sample_record)

    retrieved = await store.get(sample_record.job_id)

    assert retrieved is not None
    assert retrieved.job_id == sample_record.job_id
    assert retrieved.description == sample_record.description
    assert retrieved.timeline == sample_record.timeline
    assert retrieved.context == sample_record.context
    assert retrieved.integration_points == sample_record.integration_points
    assert retrieved.state == sample_record.state
    assert retrieved.progress == sample_record.progress
    assert retrieved.current_phase == sample_record.current_phase
    assert retrieved.quality_score == sample_record.quality_score
    assert retrieved.file_path == sample_record.file_path
    assert retrieved.error == sample_record.error
    # Timestamps may differ slightly due to serialization
    assert abs((retrieved.created_at - sample_record.created_at).total_seconds()) < 1
    assert retrieved.updated_at is not None  # Auto-set on add


@pytest.mark.asyncio
async def test_add_duplicate_raises_error(store: SQLiteJobStore, sample_record: JobRecord):
    """Test adding a job with duplicate ID raises ValueError."""
    await store.add(sample_record)

    with pytest.raises(ValueError, match="already exists"):
        await store.add(sample_record)


@pytest.mark.asyncio
async def test_list_all_returns_newest_first(store: SQLiteJobStore):
    """Test list_all returns jobs ordered by created_at (newest first)."""
    # Create 3 jobs with different timestamps
    job1 = JobRecord(
        job_id="job-001",
        description="First job",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.QUEUED,
        progress=0.0,
        current_phase=None,
        quality_score=None,
        created_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        file_path=None,
        error=None,
        updated_at=None,
    )

    job2 = JobRecord(
        job_id="job-002",
        description="Second job",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.RUNNING,
        progress=0.5,
        current_phase=None,
        quality_score=None,
        created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        file_path=None,
        error=None,
        updated_at=None,
    )

    job3 = JobRecord(
        job_id="job-003",
        description="Third job",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.COMPLETE,
        progress=1.0,
        current_phase=None,
        quality_score=0.9,
        created_at=datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
        file_path=None,
        error=None,
        updated_at=None,
    )

    await store.add(job1)
    await store.add(job2)
    await store.add(job3)

    all_jobs = await store.list_all()

    assert len(all_jobs) == 3
    assert all_jobs[0].job_id == "job-003"  # Newest first
    assert all_jobs[1].job_id == "job-002"
    assert all_jobs[2].job_id == "job-001"  # Oldest last


@pytest.mark.asyncio
async def test_update_changes_fields_and_sets_updated_at(
    store: SQLiteJobStore, sample_record: JobRecord
):
    """Test update modifies specific fields and sets updated_at."""
    await store.add(sample_record)

    # Update multiple fields
    await store.update(
        sample_record.job_id,
        state=JobState.RUNNING,
        progress=0.5,
        current_phase="Phase 1",
    )

    updated = await store.get(sample_record.job_id)

    assert updated is not None
    assert updated.state == JobState.RUNNING
    assert updated.progress == 0.5
    assert updated.current_phase == "Phase 1"
    assert updated.updated_at is not None
    assert updated.updated_at > sample_record.created_at  # Updated after creation


@pytest.mark.asyncio
async def test_update_nonexistent_job_raises_error(store: SQLiteJobStore):
    """Test updating a non-existent job raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        await store.update("nonexistent-job", state=JobState.COMPLETE)


@pytest.mark.asyncio
async def test_update_invalid_field_raises_error(store: SQLiteJobStore, sample_record: JobRecord):
    """Test updating with invalid field name raises ValueError."""
    await store.add(sample_record)

    with pytest.raises(ValueError, match="Invalid field names"):
        await store.update(sample_record.job_id, invalid_field="value")


@pytest.mark.asyncio
async def test_get_next_queued_returns_oldest_queued_job(store: SQLiteJobStore):
    """Test get_next_queued returns oldest queued job (FIFO)."""
    # Create multiple queued jobs
    job1 = JobRecord(
        job_id="queued-001",
        description="First queued",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.QUEUED,
        progress=0.0,
        current_phase=None,
        quality_score=None,
        created_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        file_path=None,
        error=None,
        updated_at=None,
    )

    job2 = JobRecord(
        job_id="running-001",
        description="Running job",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.RUNNING,
        progress=0.3,
        current_phase=None,
        quality_score=None,
        created_at=datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
        file_path=None,
        error=None,
        updated_at=None,
    )

    job3 = JobRecord(
        job_id="queued-002",
        description="Second queued",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.QUEUED,
        progress=0.0,
        current_phase=None,
        quality_score=None,
        created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        file_path=None,
        error=None,
        updated_at=None,
    )

    await store.add(job1)
    await store.add(job2)
    await store.add(job3)

    next_job = await store.get_next_queued()

    assert next_job is not None
    assert next_job.job_id == "queued-001"  # Oldest queued job


@pytest.mark.asyncio
async def test_get_next_queued_returns_none_when_no_queued_jobs(store: SQLiteJobStore):
    """Test get_next_queued returns None when no queued jobs exist."""
    # Add only non-queued jobs
    job = JobRecord(
        job_id="running-001",
        description="Running job",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.RUNNING,
        progress=0.5,
        current_phase=None,
        quality_score=None,
        created_at=datetime.now(timezone.utc),
        file_path=None,
        error=None,
        updated_at=None,
    )

    await store.add(job)

    next_job = await store.get_next_queued()

    assert next_job is None


@pytest.mark.asyncio
async def test_crash_recovery_marks_running_jobs_as_interrupted(tmp_path: Path):
    """Test crash recovery marks running jobs as interrupted on startup."""
    db_path = str(tmp_path / "crash_test.db")

    # Create store and add a running job
    store1 = SQLiteJobStore(db_path)
    await store1.initialize()

    running_job = JobRecord(
        job_id="running-job",
        description="Was running before crash",
        timeline=None,
        context=None,
        integration_points=[],
        state=JobState.RUNNING,
        progress=0.7,
        current_phase="Phase 2",
        quality_score=None,
        created_at=datetime.now(timezone.utc),
        file_path=None,
        error=None,
        updated_at=None,
    )

    await store1.add(running_job)
    await store1.close()

    # Simulate crash and restart - create new store instance
    store2 = SQLiteJobStore(db_path)
    await store2.initialize()  # This should trigger crash recovery

    recovered_job = await store2.get("running-job")

    assert recovered_job is not None
    assert recovered_job.state == JobState.INTERRUPTED
    assert recovered_job.error == "Server restarted during processing"
    assert recovered_job.updated_at is not None

    await store2.close()


@pytest.mark.asyncio
async def test_get_returns_none_for_nonexistent_job(store: SQLiteJobStore):
    """Test get returns None for non-existent job ID."""
    result = await store.get("nonexistent-job")
    assert result is None


@pytest.mark.asyncio
async def test_integration_points_serialization(store: SQLiteJobStore):
    """Test integration_points list is correctly serialized and deserialized."""
    job = JobRecord(
        job_id="serialize-test",
        description="Test serialization",
        timeline=None,
        context=None,
        integration_points=["point-1", "point-2", "point-3"],
        state=JobState.QUEUED,
        progress=0.0,
        current_phase=None,
        quality_score=None,
        created_at=datetime.now(timezone.utc),
        file_path=None,
        error=None,
        updated_at=None,
    )

    await store.add(job)
    retrieved = await store.get(job.job_id)

    assert retrieved is not None
    assert retrieved.integration_points == ["point-1", "point-2", "point-3"]
    assert isinstance(retrieved.integration_points, list)
