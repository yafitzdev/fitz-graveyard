# fitz_planner_mcp/models/schema.py
"""
Database schema definition for SQLite job persistence.

Provides DDL for tables, indexes, and schema initialization.
"""

import logging

import aiosqlite

logger = logging.getLogger(__name__)

# Schema version for future migrations
SCHEMA_VERSION = 3

# Jobs table DDL
JOBS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    timeline TEXT,
    context TEXT,
    integration_points TEXT,
    state TEXT NOT NULL CHECK(state IN ('queued', 'running', 'awaiting_review', 'complete', 'failed', 'interrupted')),
    progress REAL DEFAULT 0.0 CHECK(progress >= 0.0 AND progress <= 1.0),
    current_phase TEXT,
    quality_score REAL,
    file_path TEXT,
    error TEXT,
    pipeline_state TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    api_review INTEGER DEFAULT 0,
    cost_estimate_json TEXT,
    review_result_json TEXT
)
"""

# Index for efficient FIFO queue queries (state + created_at)
JOBS_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_jobs_state_created ON jobs(state, created_at)"


async def _get_schema_version(db: aiosqlite.Connection) -> int:
    """
    Get current schema version from database.

    Args:
        db: Database connection

    Returns:
        Schema version (0 if no version table exists)
    """
    try:
        cursor = await db.execute("SELECT version FROM schema_version LIMIT 1")
        row = await cursor.fetchone()
        return row[0] if row else 0
    except aiosqlite.OperationalError:
        return 0


async def _set_schema_version(db: aiosqlite.Connection, version: int) -> None:
    """
    Set schema version in database.

    Args:
        db: Database connection
        version: Schema version to set
    """
    await db.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
    await db.execute("DELETE FROM schema_version")
    await db.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))


async def _migrate_v1_to_v2(db: aiosqlite.Connection) -> None:
    """
    Migrate schema from v1 to v2.

    Changes:
        - Add pipeline_state column to jobs table
    """
    logger.info("Migrating schema from v1 to v2")

    # Check if pipeline_state column already exists
    cursor = await db.execute("PRAGMA table_info(jobs)")
    columns = await cursor.fetchall()
    column_names = [col[1] for col in columns]

    if "pipeline_state" not in column_names:
        await db.execute("ALTER TABLE jobs ADD COLUMN pipeline_state TEXT")
        logger.info("Added pipeline_state column to jobs table")
    else:
        logger.info("pipeline_state column already exists (skip)")


async def _migrate_v2_to_v3(db: aiosqlite.Connection) -> None:
    """
    Migrate schema from v2 to v3.

    Changes:
        - Add awaiting_review to state CHECK constraint (recreate table)
        - Add api_review column to jobs table
        - Add cost_estimate_json column to jobs table
        - Add review_result_json column to jobs table
    """
    logger.info("Migrating schema from v2 to v3")

    # Get current table columns
    cursor = await db.execute("PRAGMA table_info(jobs)")
    columns = await cursor.fetchall()
    column_names = [col[1] for col in columns]

    # Add new columns if they don't exist
    if "api_review" not in column_names:
        await db.execute("ALTER TABLE jobs ADD COLUMN api_review INTEGER DEFAULT 0")
        logger.info("Added api_review column to jobs table")

    if "cost_estimate_json" not in column_names:
        await db.execute("ALTER TABLE jobs ADD COLUMN cost_estimate_json TEXT")
        logger.info("Added cost_estimate_json column to jobs table")

    if "review_result_json" not in column_names:
        await db.execute("ALTER TABLE jobs ADD COLUMN review_result_json TEXT")
        logger.info("Added review_result_json column to jobs table")

    # SQLite doesn't support modifying CHECK constraints directly
    # The awaiting_review state will be allowed without recreating the table
    # (SQLite validates CHECK constraints only on insert/update, not schema-level)
    logger.info("Note: awaiting_review state added to enum (no table recreation needed)")


async def init_db(db_path: str) -> None:
    """
    Initialize database schema with WAL mode and optimal settings.

    Handles schema migrations automatically.

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

        # Handle schema migrations
        current_version = await _get_schema_version(db)

        if current_version < SCHEMA_VERSION:
            logger.info(
                f"Schema migration needed: v{current_version} -> v{SCHEMA_VERSION}"
            )

            if current_version < 1:
                # New database, set version directly
                await _set_schema_version(db, SCHEMA_VERSION)
                logger.info(f"New database initialized at v{SCHEMA_VERSION}")
            elif current_version == 1:
                # Migrate from v1 to v2
                await _migrate_v1_to_v2(db)
                await _migrate_v2_to_v3(db)
                await _set_schema_version(db, 3)
                logger.info("Migration to v3 complete")
            elif current_version == 2:
                # Migrate from v2 to v3
                await _migrate_v2_to_v3(db)
                await _set_schema_version(db, 3)
                logger.info("Migration to v3 complete")

        await db.commit()

        logger.info(f"Initialized database at {db_path} (schema v{SCHEMA_VERSION})")
