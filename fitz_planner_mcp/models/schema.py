# fitz_planner_mcp/models/schema.py
"""
Database schema definition for SQLite job persistence.

Provides DDL for tables, indexes, and schema initialization.
"""

import logging

import aiosqlite

logger = logging.getLogger(__name__)

# Schema version for future migrations
SCHEMA_VERSION = 1

# Jobs table DDL
JOBS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    timeline TEXT,
    context TEXT,
    integration_points TEXT,
    state TEXT NOT NULL CHECK(state IN ('queued', 'running', 'complete', 'failed', 'interrupted')),
    progress REAL DEFAULT 0.0 CHECK(progress >= 0.0 AND progress <= 1.0),
    current_phase TEXT,
    quality_score REAL,
    file_path TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

# Index for efficient FIFO queue queries (state + created_at)
JOBS_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_jobs_state_created ON jobs(state, created_at)"


async def init_db(db_path: str) -> None:
    """
    Initialize database schema with WAL mode and optimal settings.

    Args:
        db_path: Path to SQLite database file

    Settings:
        - WAL mode: Concurrent reads + writes
        - synchronous=NORMAL: Good durability/performance balance
        - busy_timeout=5000ms: Retry on SQLITE_BUSY
        - foreign_keys=ON: Enforce constraints
    """
    async with aiosqlite.connect(db_path) as db:
        # Enable WAL mode for concurrent access
        await db.execute("PRAGMA journal_mode=WAL")

        # Performance/durability balance
        await db.execute("PRAGMA synchronous=NORMAL")

        # Retry on database busy (5 seconds)
        await db.execute("PRAGMA busy_timeout=5000")

        # Enable foreign key constraints
        await db.execute("PRAGMA foreign_keys=ON")

        # Create tables and indexes
        await db.execute(JOBS_TABLE_SQL)
        await db.execute(JOBS_INDEX_SQL)

        await db.commit()

        logger.info(f"Initialized database at {db_path} (schema v{SCHEMA_VERSION})")
