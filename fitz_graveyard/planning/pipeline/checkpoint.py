# fitz_graveyard/planning/pipeline/checkpoint.py
"""
Checkpoint management for pipeline recovery.

Persists stage outputs to SQLite jobs.pipeline_state column,
enabling crash recovery and incremental progress.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages pipeline checkpoint persistence via SQLite.

    Stores completed stage outputs in the jobs.pipeline_state column
    for crash recovery. Pipeline can resume from the last completed stage.
    """

    def __init__(self, db_path: str) -> None:
        """
        Initialize checkpoint manager.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = db_path
        logger.info(f"Created CheckpointManager with db: {db_path}")

    async def save_stage(
        self, job_id: str, stage_name: str, stage_output: dict[str, Any]
    ) -> None:
        """
        Save a completed stage's output as a checkpoint.

        Args:
            job_id: Job identifier
            stage_name: Name of the completed stage
            stage_output: Parsed stage output dictionary

        Raises:
            ValueError: If job_id doesn't exist
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("BEGIN IMMEDIATE")

            try:
                # Load existing checkpoint state
                cursor = await db.execute(
                    "SELECT pipeline_state FROM jobs WHERE id = ?", (job_id,)
                )
                row = await cursor.fetchone()

                if not row:
                    raise ValueError(f"Job {job_id} not found")

                # Parse existing state (or start fresh)
                existing_state = json.loads(row[0]) if row[0] else {}

                # Add this stage's output with timestamp
                existing_state[stage_name] = {
                    "output": stage_output,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }

                # Save back to DB
                await db.execute(
                    "UPDATE jobs SET pipeline_state = ? WHERE id = ?",
                    (json.dumps(existing_state), job_id),
                )

                await db.commit()
                logger.info(f"Saved checkpoint for stage '{stage_name}' in job {job_id}")

            except Exception:
                await db.rollback()
                raise

    async def load_checkpoint(self, job_id: str) -> dict[str, Any]:
        """
        Load all completed stage outputs for a job.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary mapping stage_name -> stage_output
            (empty dict if no checkpoint exists)

        Raises:
            ValueError: If job_id doesn't exist
        """
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT pipeline_state FROM jobs WHERE id = ?", (job_id,)
            )
            row = await cursor.fetchone()

            if not row:
                raise ValueError(f"Job {job_id} not found")

            if not row[0]:
                logger.info(f"No checkpoint found for job {job_id}")
                return {}

            checkpoint = json.loads(row[0])
            logger.info(
                f"Loaded checkpoint for job {job_id}: {list(checkpoint.keys())}"
            )

            # Unwrap timestamped format and warn on stale checkpoints
            STALE_HOURS = 24
            unwrapped = {}
            for key, val in checkpoint.items():
                if isinstance(val, dict) and "output" in val and "completed_at" in val:
                    # New timestamped format — unwrap
                    try:
                        completed = datetime.fromisoformat(val["completed_at"])
                        age_hours = (datetime.now(timezone.utc) - completed).total_seconds() / 3600
                        if age_hours > STALE_HOURS:
                            logger.warning(
                                f"Checkpoint '{key}' is {age_hours:.0f}h old (>{STALE_HOURS}h)"
                            )
                    except (ValueError, TypeError):
                        pass
                    unwrapped[key] = val["output"]
                else:
                    # Old format (plain dict) — use as-is
                    unwrapped[key] = val
            return unwrapped

    async def clear_checkpoint(self, job_id: str) -> None:
        """
        Clear all checkpoint data for a job.

        Used when starting a new planning run (not resuming).

        Args:
            job_id: Job identifier

        Raises:
            ValueError: If job_id doesn't exist
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("BEGIN IMMEDIATE")

            try:
                cursor = await db.execute("SELECT id FROM jobs WHERE id = ?", (job_id,))
                if not await cursor.fetchone():
                    raise ValueError(f"Job {job_id} not found")

                await db.execute(
                    "UPDATE jobs SET pipeline_state = NULL WHERE id = ?", (job_id,)
                )

                await db.commit()
                logger.info(f"Cleared checkpoint for job {job_id}")

            except Exception:
                await db.rollback()
                raise
