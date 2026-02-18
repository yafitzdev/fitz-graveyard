# fitz_graveyard/models/jobs.py
"""
Job tracking models and in-memory storage.

Internal models (NOT exposed via MCP) for managing planning jobs.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from fitz_graveyard.models.store import JobStore

logger = logging.getLogger(__name__)


class JobState(Enum):
    """Job lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETE = "complete"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class JobRecord:
    """
    Internal job record (NOT Pydantic - not exposed via MCP).

    Tracks all job state and metadata.
    """

    job_id: str
    description: str
    timeline: str | None
    context: str | None
    integration_points: list[str]
    state: JobState
    progress: float
    current_phase: str | None
    quality_score: float | None
    created_at: datetime
    file_path: str | None = None
    error: str | None = None  # Error message if state=FAILED
    pipeline_state: str | None = None  # JSON checkpoint data
    updated_at: datetime | None = None
    api_review: bool = False  # Whether user opted into API review
    cost_estimate_json: str | None = None  # Serialized CostEstimate JSON (set by worker)
    review_result_json: str | None = None  # Serialized list of ReviewResult JSON (set after review)


class InMemoryJobStore(JobStore):
    """
    Simple in-memory job storage.

    Thread-safe for single-process usage.
    Implements JobStore protocol with async wrappers around sync operations.
    """

    def __init__(self) -> None:
        """Initialize empty job store."""
        self._jobs: dict[str, JobRecord] = {}
        logger.info("Initialized InMemoryJobStore")

    async def add(self, record: JobRecord) -> None:
        """
        Add a job record to the store.

        Args:
            record: JobRecord to add

        Raises:
            ValueError: If job_id already exists
        """
        if record.job_id in self._jobs:
            raise ValueError(f"Job {record.job_id} already exists")

        self._jobs[record.job_id] = record
        logger.info(f"Added job {record.job_id} to store")

    async def get(self, job_id: str) -> JobRecord | None:
        """
        Get a job record by ID.

        Args:
            job_id: Job identifier

        Returns:
            JobRecord if found, None otherwise
        """
        return self._jobs.get(job_id)

    async def list_all(self) -> list[JobRecord]:
        """
        List all job records.

        Returns:
            List of all JobRecords, ordered by creation time (newest first)
        """
        return sorted(
            self._jobs.values(), key=lambda r: r.created_at, reverse=True
        )

    async def update(self, job_id: str, **kwargs) -> None:
        """
        Update fields on an existing job record.

        Args:
            job_id: Job identifier
            **kwargs: Fields to update

        Raises:
            ValueError: If job_id doesn't exist
        """
        record = self._jobs.get(job_id)
        if not record:
            raise ValueError(f"Job {job_id} not found")

        for key, value in kwargs.items():
            if hasattr(record, key):
                setattr(record, key, value)
            else:
                logger.warning(f"Ignored unknown field '{key}' in update")

        logger.info(f"Updated job {job_id}: {kwargs}")

    async def get_next_queued(self) -> JobRecord | None:
        """
        Get the next queued job (FIFO - oldest first).

        Returns:
            JobRecord with state=QUEUED (oldest), or None if no queued jobs
        """
        queued = [r for r in self._jobs.values() if r.state == JobState.QUEUED]
        if not queued:
            return None
        return min(queued, key=lambda r: r.created_at)


def generate_job_id() -> str:
    """
    Generate a unique job ID.

    Returns:
        12-character hex string (UUID4 truncated)
    """
    return uuid4().hex[:12]
