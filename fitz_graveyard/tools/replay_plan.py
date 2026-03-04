# fitz_graveyard/tools/replay_plan.py
"""
replay_plan tool implementation.

Creates a new job that reuses agent context from a completed job,
skipping the expensive agent gathering pass and re-running only
the planning stages.
"""

import json
import logging
from datetime import datetime, timezone

from fitz_graveyard.models.jobs import JobRecord, JobState, generate_job_id
from fitz_graveyard.models.store import JobStore

logger = logging.getLogger(__name__)

# Keys from the checkpoint that represent agent/pre-planning work
_AGENT_KEYS = {"_agent_context"}


async def replay_plan(
    source_job_id: str,
    store: JobStore,
    db_path: str,
) -> dict:
    """
    Create a new job that reuses agent context from a completed job.

    Copies the agent gathering checkpoint so the new job skips straight
    to the planning stages. Useful for rapid iteration on planning logic
    without re-running the expensive codebase exploration.

    Args:
        source_job_id: Job ID to copy agent context from
        store: Job storage instance
        db_path: Path to SQLite database (for checkpoint access)

    Returns:
        Dict with new job_id and source_job_id

    Raises:
        ValueError: If source job doesn't exist or has no agent context
    """
    import aiosqlite

    # Verify source job exists
    source_job = await store.get(source_job_id)
    if not source_job:
        raise ValueError(f"Job '{source_job_id}' not found")

    # Load checkpoint from source job
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT pipeline_state FROM jobs WHERE id = ?", (source_job_id,)
        )
        row = await cursor.fetchone()

    if not row or not row[0]:
        raise ValueError(f"Job '{source_job_id}' has no checkpoint data")

    checkpoint = json.loads(row[0])

    if "_agent_context" not in checkpoint:
        raise ValueError(
            f"Job '{source_job_id}' has no agent context to reuse. "
            "Only jobs that completed agent gathering can be replayed."
        )

    # Build partial checkpoint with only agent data
    partial = {}
    for key in _AGENT_KEYS:
        if key in checkpoint:
            partial[key] = checkpoint[key]

    # Create new job with same description
    new_job_id = generate_job_id()
    record = JobRecord(
        job_id=new_job_id,
        description=source_job.description,
        timeline=source_job.timeline,
        context=source_job.context,
        integration_points=source_job.integration_points,
        state=JobState.QUEUED,
        progress=0.0,
        current_phase=None,
        quality_score=None,
        created_at=datetime.now(timezone.utc),
        file_path=None,
        api_review=source_job.api_review,
        source_dir=source_job.source_dir,
    )
    await store.add(record)

    # Write partial checkpoint to new job
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE jobs SET pipeline_state = ? WHERE id = ?",
            (json.dumps(partial), new_job_id),
        )
        await db.commit()

    agent_ctx = checkpoint["_agent_context"]
    if isinstance(agent_ctx, dict) and "output" in agent_ctx:
        agent_ctx = agent_ctx["output"]
    synth_len = len(agent_ctx.get("synthesized", ""))
    raw_len = len(agent_ctx.get("raw_summaries", ""))

    logger.info(
        f"Created replay job {new_job_id} from {source_job_id}: "
        f"reusing agent context (synthesized={synth_len}, raw={raw_len} chars)"
    )

    return {
        "job_id": new_job_id,
        "source_job_id": source_job_id,
        "description": source_job.description,
    }
