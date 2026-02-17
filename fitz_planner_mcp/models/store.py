# fitz_planner_mcp/models/store.py
"""
Job store protocol definition.

Defines the abstract interface that both InMemoryJobStore and SQLiteJobStore implement.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fitz_planner_mcp.models.jobs import JobRecord


class JobStore(ABC):
    """
    Abstract base class for job storage implementations.

    Both in-memory and persistent (SQLite) stores implement this protocol.
    """

    @abstractmethod
    async def add(self, record: "JobRecord") -> None:
        """
        Add a job record to the store.

        Args:
            record: JobRecord to add

        Raises:
            ValueError: If job_id already exists
        """
        pass

    @abstractmethod
    async def get(self, job_id: str) -> "JobRecord | None":
        """
        Get a job record by ID.

        Args:
            job_id: Job identifier

        Returns:
            JobRecord if found, None otherwise
        """
        pass

    @abstractmethod
    async def list_all(self) -> "list[JobRecord]":
        """
        List all job records.

        Returns:
            List of all JobRecords, ordered by creation time (newest first)
        """
        pass

    @abstractmethod
    async def update(self, job_id: str, **kwargs) -> None:
        """
        Update fields on an existing job record.

        Args:
            job_id: Job identifier
            **kwargs: Fields to update

        Raises:
            ValueError: If job_id doesn't exist
        """
        pass

    @abstractmethod
    async def get_next_queued(self) -> "JobRecord | None":
        """
        Get the next queued job (FIFO - oldest first).

        Returns:
            JobRecord with state=QUEUED (oldest), or None if no queued jobs
        """
        pass
