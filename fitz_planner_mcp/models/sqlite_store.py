# fitz_planner_mcp/models/sqlite_store.py
"""
SQLite-backed job persistence.

Provides async CRUD operations with WAL mode and IMMEDIATE transactions
for crash recovery and concurrent access safety.
"""

import json
import logging
from datetime import datetime, timezone

import aiosqlite

from fitz_planner_mcp.models.jobs import JobRecord, JobState
from fitz_planner_mcp.models.schema import init_db
from fitz_planner_mcp.models.store import JobStore

logger = logging.getLogger(__name__)


class SQLiteJobStore(JobStore):
    """
    Async SQLite-backed job storage.

    Features:
        - WAL mode for concurrent reads/writes
        - IMMEDIATE transactions for write safety
        - Crash recovery (running → interrupted on startup)
        - No persistent connections (avoids resource leaks)
    """

    def __init__(self, db_path: str) -> None:
        """
        Initialize SQLite job store.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = db_path
        logger.info(f"Created SQLiteJobStore with path: {db_path}")

    async def initialize(self) -> None:
        """
        Initialize database schema and perform crash recovery.

        Crash recovery: Mark any jobs with state='running' as 'interrupted'.
        """
        # Initialize schema with WAL mode
        await init_db(self._db_path)

        # Crash recovery: mark running jobs as interrupted
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("BEGIN IMMEDIATE")
            cursor = await db.execute(
                "UPDATE jobs SET state = ?, error = ?, updated_at = ? WHERE state = ?",
                (
                    JobState.INTERRUPTED.value,
                    "Server restarted during processing",
                    datetime.now(timezone.utc).isoformat(),
                    JobState.RUNNING.value,
                ),
            )
            recovered = cursor.rowcount
            await db.commit()

            if recovered > 0:
                logger.warning(
                    f"Crash recovery: marked {recovered} running job(s) as interrupted"
                )

    async def add(self, record: JobRecord) -> None:
        """
        Add a job record to the store.

        Args:
            record: JobRecord to add

        Raises:
            ValueError: If job_id already exists
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("BEGIN IMMEDIATE")

            try:
                # Check for duplicate
                cursor = await db.execute("SELECT id FROM jobs WHERE id = ?", (record.job_id,))
                if await cursor.fetchone():
                    raise ValueError(f"Job {record.job_id} already exists")

                # Serialize fields
                integration_points_json = json.dumps(record.integration_points)
                created_at_iso = record.created_at.isoformat()
                updated_at_iso = (
                    record.updated_at.isoformat()
                    if record.updated_at
                    else datetime.now(timezone.utc).isoformat()
                )

                # Insert
                await db.execute(
                    """
                    INSERT INTO jobs (
                        id, description, timeline, context, integration_points,
                        state, progress, current_phase, quality_score, file_path,
                        error, pipeline_state, created_at, updated_at,
                        api_review, cost_estimate_json, review_result_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.job_id,
                        record.description,
                        record.timeline,
                        record.context,
                        integration_points_json,
                        record.state.value,
                        record.progress,
                        record.current_phase,
                        record.quality_score,
                        record.file_path,
                        record.error,
                        record.pipeline_state,
                        created_at_iso,
                        updated_at_iso,
                        1 if record.api_review else 0,
                        record.cost_estimate_json,
                        record.review_result_json,
                    ),
                )

                await db.commit()
                logger.info(f"Added job {record.job_id} to SQLite store")

            except Exception:
                await db.rollback()
                raise

    async def get(self, job_id: str) -> JobRecord | None:
        """
        Get a job record by ID.

        Args:
            job_id: Job identifier

        Returns:
            JobRecord if found, None otherwise
        """
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = await cursor.fetchone()

            if not row:
                return None

            return self._row_to_record(row)

    async def list_all(self) -> list[JobRecord]:
        """
        List all job records.

        Returns:
            List of all JobRecords, ordered by creation time (newest first)
        """
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            rows = await cursor.fetchall()

            return [self._row_to_record(row) for row in rows]

    async def update(self, job_id: str, **kwargs) -> None:
        """
        Update fields on an existing job record.

        Args:
            job_id: Job identifier
            **kwargs: Fields to update

        Raises:
            ValueError: If job_id doesn't exist or invalid field name
        """
        # Validate field names against JobRecord
        valid_fields = {
            "description",
            "timeline",
            "context",
            "integration_points",
            "state",
            "progress",
            "current_phase",
            "quality_score",
            "file_path",
            "error",
            "pipeline_state",
            "updated_at",
            "api_review",
            "cost_estimate_json",
            "review_result_json",
        }

        invalid = set(kwargs.keys()) - valid_fields
        if invalid:
            raise ValueError(f"Invalid field names: {invalid}")

        if not kwargs:
            return  # Nothing to update

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("BEGIN IMMEDIATE")

            try:
                # Check job exists
                cursor = await db.execute("SELECT id FROM jobs WHERE id = ?", (job_id,))
                if not await cursor.fetchone():
                    raise ValueError(f"Job {job_id} not found")

                # Build SET clause
                set_parts = []
                values = []

                for key, value in kwargs.items():
                    if key == "state" and isinstance(value, JobState):
                        value = value.value
                    elif key == "integration_points" and isinstance(value, list):
                        value = json.dumps(value)
                    elif key in ("created_at", "updated_at") and isinstance(value, datetime):
                        value = value.isoformat()
                    elif key == "api_review" and isinstance(value, bool):
                        value = 1 if value else 0

                    set_parts.append(f"{key} = ?")
                    values.append(value)

                # Always update updated_at
                if "updated_at" not in kwargs:
                    set_parts.append("updated_at = ?")
                    values.append(datetime.now(timezone.utc).isoformat())

                values.append(job_id)  # For WHERE clause

                sql = f"UPDATE jobs SET {', '.join(set_parts)} WHERE id = ?"
                await db.execute(sql, values)

                await db.commit()
                logger.info(f"Updated job {job_id}: {list(kwargs.keys())}")

            except Exception:
                await db.rollback()
                raise

    async def get_next_queued(self) -> JobRecord | None:
        """
        Get the next queued job (FIFO - oldest first).

        Returns:
            JobRecord with state=QUEUED (oldest), or None if no queued jobs
        """
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM jobs WHERE state = ? ORDER BY created_at ASC LIMIT 1",
                (JobState.QUEUED.value,),
            )
            row = await cursor.fetchone()

            if not row:
                return None

            return self._row_to_record(row)

    async def close(self) -> None:
        """
        Checkpoint WAL and close database.

        Truncates WAL file to avoid unbounded growth.
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                logger.info("WAL checkpoint completed")
        except Exception as e:
            logger.warning(f"WAL checkpoint failed: {e}")

    def _row_to_record(self, row: aiosqlite.Row) -> JobRecord:
        """
        Convert SQLite row to JobRecord.

        Args:
            row: SQLite row (with row_factory=aiosqlite.Row)

        Returns:
            JobRecord instance
        """
        # Helper to safely get column value with default
        def safe_get(name: str, default=None):
            try:
                return row[name]
            except (KeyError, IndexError):
                return default

        return JobRecord(
            job_id=row["id"],
            description=row["description"],
            timeline=row["timeline"],
            context=row["context"],
            integration_points=json.loads(row["integration_points"])
            if row["integration_points"]
            else [],
            state=JobState(row["state"]),
            progress=row["progress"],
            current_phase=row["current_phase"],
            quality_score=row["quality_score"],
            created_at=datetime.fromisoformat(row["created_at"]),
            file_path=row["file_path"],
            error=row["error"],
            pipeline_state=row["pipeline_state"],
            updated_at=datetime.fromisoformat(row["updated_at"])
            if row["updated_at"]
            else None,
            api_review=bool(safe_get("api_review", 0)),
            cost_estimate_json=safe_get("cost_estimate_json"),
            review_result_json=safe_get("review_result_json"),
        )
