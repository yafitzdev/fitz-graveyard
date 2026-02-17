# tests/unit/test_pipeline_infra.py
"""
Unit tests for pipeline infrastructure.

Tests PipelineStage ABC, PlanningPipeline orchestrator,
CheckpointManager, and schema migration.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite
import pytest
import pytest_asyncio

from fitz_planner_mcp.models.schema import SCHEMA_VERSION, init_db
from fitz_planner_mcp.planning.pipeline import (
    CheckpointManager,
    PipelineStage,
    PlanningPipeline,
)
from fitz_planner_mcp.planning.pipeline.orchestrator import PipelineResult, get_git_sha
from fitz_planner_mcp.planning.pipeline.stages.base import StageResult


# Mock LLM client for testing
@dataclass
class MockLLMResponse:
    """Mock LLM response."""

    content: str


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: dict[str, str] | None = None):
        """
        Initialize mock client.

        Args:
            responses: Dict mapping stage_name -> response content
        """
        self.responses = responses or {}
        self.call_count = 0

    async def generate_chat(self, messages: list[dict]) -> MockLLMResponse:
        """
        Mock generate_chat method.

        Returns predetermined response based on call count.
        """
        self.call_count += 1
        stage_key = f"stage_{self.call_count}"
        content = self.responses.get(stage_key, f"Mock response {self.call_count}")
        return MockLLMResponse(content=content)


# Mock pipeline stages for testing
class MockStageA(PipelineStage):
    """Mock stage A."""

    @property
    def name(self) -> str:
        return "stage_a"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.0, 0.5)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any]
    ) -> list[dict]:
        return [
            {"role": "system", "content": "You are a planner"},
            {"role": "user", "content": job_description},
        ]

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        return {"stage": "a", "content": raw_output}


class MockStageB(PipelineStage):
    """Mock stage B."""

    @property
    def name(self) -> str:
        return "stage_b"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.5, 1.0)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any]
    ) -> list[dict]:
        # Uses output from stage_a
        stage_a_content = prior_outputs.get("stage_a", {}).get("content", "")
        return [
            {"role": "system", "content": "You are a planner"},
            {
                "role": "user",
                "content": f"Job: {job_description}\nStage A: {stage_a_content}",
            },
        ]

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        return {"stage": "b", "content": raw_output}


class FailingStage(PipelineStage):
    """Mock stage that always fails."""

    @property
    def name(self) -> str:
        return "failing_stage"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.0, 1.0)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any]
    ) -> list[dict]:
        return [{"role": "user", "content": job_description}]

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        raise ValueError("Parse failed intentionally")


# Fixtures
@pytest_asyncio.fixture
async def temp_db(tmp_path: Path) -> str:
    """Create temporary test database."""
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)
    return db_path


@pytest_asyncio.fixture
async def checkpoint_manager(temp_db: str) -> CheckpointManager:
    """Create CheckpointManager for testing."""
    return CheckpointManager(temp_db)


@pytest_asyncio.fixture
async def mock_client() -> MockLLMClient:
    """Create mock LLM client."""
    return MockLLMClient(
        responses={"stage_1": "Response A", "stage_2": "Response B"}
    )


# Tests for PipelineStage ABC
@pytest.mark.asyncio
async def test_pipeline_stage_execute_success(mock_client: MockLLMClient):
    """Test successful stage execution."""
    stage = MockStageA()
    result = await stage.execute(mock_client, "Build a web app", {})

    assert result.success is True
    assert result.stage_name == "stage_a"
    assert result.output == {"stage": "a", "content": "Response A"}
    assert result.raw_output == "Response A"
    assert result.error is None


@pytest.mark.asyncio
async def test_pipeline_stage_execute_failure(mock_client: MockLLMClient):
    """Test stage execution failure (parse error)."""
    stage = FailingStage()
    result = await stage.execute(mock_client, "Build a web app", {})

    assert result.success is False
    assert result.stage_name == "failing_stage"
    assert result.output == {}
    assert result.error == "Parse failed intentionally"


@pytest.mark.asyncio
async def test_pipeline_stage_prior_outputs(mock_client: MockLLMClient):
    """Test stage receives prior outputs."""
    stage_b = MockStageB()
    prior = {"stage_a": {"content": "Previous result"}}

    # Build prompt should use prior output
    messages = stage_b.build_prompt("Build API", prior)
    assert len(messages) == 2
    assert "Previous result" in messages[1]["content"]


# Tests for CheckpointManager
@pytest.mark.asyncio
async def test_checkpoint_save_and_load(
    checkpoint_manager: CheckpointManager, temp_db: str
):
    """Test saving and loading checkpoints."""
    # Create a job
    async with aiosqlite.connect(temp_db) as db:
        await db.execute(
            """INSERT INTO jobs (
                id, description, state, progress, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("job123", "Test job", "running", 0.0, "2024-01-01", "2024-01-01"),
        )
        await db.commit()

    # Save checkpoint for stage A
    await checkpoint_manager.save_stage(
        "job123", "stage_a", {"result": "data_a"}
    )

    # Load checkpoint
    checkpoint = await checkpoint_manager.load_checkpoint("job123")
    assert checkpoint == {"stage_a": {"result": "data_a"}}

    # Save checkpoint for stage B
    await checkpoint_manager.save_stage(
        "job123", "stage_b", {"result": "data_b"}
    )

    # Load both checkpoints
    checkpoint = await checkpoint_manager.load_checkpoint("job123")
    assert checkpoint == {
        "stage_a": {"result": "data_a"},
        "stage_b": {"result": "data_b"},
    }


@pytest.mark.asyncio
async def test_checkpoint_clear(checkpoint_manager: CheckpointManager, temp_db: str):
    """Test clearing checkpoints."""
    # Create a job
    async with aiosqlite.connect(temp_db) as db:
        await db.execute(
            """INSERT INTO jobs (
                id, description, state, progress, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("job456", "Test job", "running", 0.0, "2024-01-01", "2024-01-01"),
        )
        await db.commit()

    # Save checkpoint
    await checkpoint_manager.save_stage("job456", "stage_a", {"result": "data"})

    # Clear checkpoint
    await checkpoint_manager.clear_checkpoint("job456")

    # Load should return empty dict
    checkpoint = await checkpoint_manager.load_checkpoint("job456")
    assert checkpoint == {}


@pytest.mark.asyncio
async def test_checkpoint_nonexistent_job(checkpoint_manager: CheckpointManager):
    """Test checkpoint operations on nonexistent job."""
    with pytest.raises(ValueError, match="Job fake_id not found"):
        await checkpoint_manager.save_stage("fake_id", "stage", {})

    with pytest.raises(ValueError, match="Job fake_id not found"):
        await checkpoint_manager.load_checkpoint("fake_id")

    with pytest.raises(ValueError, match="Job fake_id not found"):
        await checkpoint_manager.clear_checkpoint("fake_id")


# Tests for PlanningPipeline
@pytest.mark.asyncio
async def test_pipeline_execute_fresh(
    mock_client: MockLLMClient,
    checkpoint_manager: CheckpointManager,
    temp_db: str,
):
    """Test pipeline execution from scratch."""
    # Create job
    async with aiosqlite.connect(temp_db) as db:
        await db.execute(
            """INSERT INTO jobs (
                id, description, state, progress, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("job789", "Build web app", "running", 0.0, "2024-01-01", "2024-01-01"),
        )
        await db.commit()

    # Create pipeline
    stages = [MockStageA(), MockStageB()]
    pipeline = PlanningPipeline(stages, checkpoint_manager)

    # Execute
    result = await pipeline.execute(mock_client, "job789", "Build web app", resume=False)

    assert result.success is True
    assert "stage_a" in result.outputs
    assert "stage_b" in result.outputs
    assert result.outputs["stage_a"]["content"] == "Response A"
    assert result.outputs["stage_b"]["content"] == "Response B"
    assert result.failed_stage is None
    assert result.error is None


@pytest.mark.asyncio
async def test_pipeline_execute_resume(
    mock_client: MockLLMClient,
    checkpoint_manager: CheckpointManager,
    temp_db: str,
):
    """Test pipeline resume from checkpoint."""
    # Create job
    async with aiosqlite.connect(temp_db) as db:
        await db.execute(
            """INSERT INTO jobs (
                id, description, state, progress, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("job999", "Build API", "running", 0.0, "2024-01-01", "2024-01-01"),
        )
        await db.commit()

    # Save checkpoint for stage A
    await checkpoint_manager.save_stage(
        "job999", "stage_a", {"stage": "a", "content": "Cached result"}
    )

    # Create pipeline
    stages = [MockStageA(), MockStageB()]
    pipeline = PlanningPipeline(stages, checkpoint_manager)

    # Execute with resume=True
    result = await pipeline.execute(mock_client, "job999", "Build API", resume=True)

    assert result.success is True
    assert result.outputs["stage_a"]["content"] == "Cached result"  # From checkpoint
    assert result.outputs["stage_b"]["content"] == "Response A"  # From LLM (stage_1)
    assert mock_client.call_count == 1  # Only stage B called


@pytest.mark.asyncio
async def test_pipeline_execute_failure(
    mock_client: MockLLMClient,
    checkpoint_manager: CheckpointManager,
    temp_db: str,
):
    """Test pipeline handles stage failure."""
    # Create job
    async with aiosqlite.connect(temp_db) as db:
        await db.execute(
            """INSERT INTO jobs (
                id, description, state, progress, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("job111", "Build app", "running", 0.0, "2024-01-01", "2024-01-01"),
        )
        await db.commit()

    # Create pipeline with failing stage
    stages = [MockStageA(), FailingStage()]
    pipeline = PlanningPipeline(stages, checkpoint_manager)

    # Execute
    result = await pipeline.execute(mock_client, "job111", "Build app", resume=False)

    assert result.success is False
    assert result.failed_stage == "failing_stage"
    assert "Parse failed intentionally" in result.error
    assert "stage_a" in result.outputs  # First stage completed
    assert "failing_stage" not in result.outputs  # Failed stage not saved


@pytest.mark.asyncio
async def test_pipeline_get_progress():
    """Test progress calculation."""
    stages = [MockStageA(), MockStageB()]
    pipeline = PlanningPipeline(stages, None)

    # No stages completed
    assert pipeline.get_progress(set()) == 0.0

    # Stage A completed
    assert pipeline.get_progress({"stage_a"}) == 0.5

    # Both stages completed
    assert pipeline.get_progress({"stage_a", "stage_b"}) == 1.0


# Tests for schema migration
@pytest.mark.asyncio
async def test_schema_migration_v1_to_v2(tmp_path: Path):
    """Test schema migration from v1 to v2."""
    db_path = str(tmp_path / "migration.db")

    # Create v1 schema manually (without pipeline_state)
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                timeline TEXT,
                context TEXT,
                integration_points TEXT,
                state TEXT NOT NULL,
                progress REAL DEFAULT 0.0,
                current_phase TEXT,
                quality_score REAL,
                file_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await db.execute("CREATE TABLE schema_version (version INTEGER)")
        await db.execute("INSERT INTO schema_version (version) VALUES (1)")
        await db.commit()

    # Run init_db (should migrate to v2)
    await init_db(db_path)

    # Verify migration
    async with aiosqlite.connect(db_path) as db:
        # Check schema version
        cursor = await db.execute("SELECT version FROM schema_version")
        version = (await cursor.fetchone())[0]
        assert version == SCHEMA_VERSION

        # Check pipeline_state column exists
        cursor = await db.execute("PRAGMA table_info(jobs)")
        columns = await cursor.fetchall()
        column_names = [col[1] for col in columns]
        assert "pipeline_state" in column_names


@pytest.mark.asyncio
async def test_schema_new_database(tmp_path: Path):
    """Test schema initialization for new database."""
    db_path = str(tmp_path / "new.db")

    # Initialize database
    await init_db(db_path)

    # Verify schema
    async with aiosqlite.connect(db_path) as db:
        # Check schema version
        cursor = await db.execute("SELECT version FROM schema_version")
        version = (await cursor.fetchone())[0]
        assert version == SCHEMA_VERSION

        # Check pipeline_state column exists
        cursor = await db.execute("PRAGMA table_info(jobs)")
        columns = await cursor.fetchall()
        column_names = [col[1] for col in columns]
        assert "pipeline_state" in column_names


# Tests for get_git_sha helper
def test_get_git_sha():
    """Test git SHA retrieval."""
    sha = get_git_sha()
    # Should return 7-char SHA or None (if not in git repo)
    assert sha is None or (isinstance(sha, str) and len(sha) == 7)


# Integration test
@pytest.mark.asyncio
async def test_pipeline_full_integration(tmp_path: Path):
    """Integration test: full pipeline with checkpointing."""
    db_path = str(tmp_path / "integration.db")
    await init_db(db_path)

    # Create job
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT INTO jobs (
                id, description, state, progress, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("integration_job", "Build system", "running", 0.0, "2024-01-01", "2024-01-01"),
        )
        await db.commit()

    # Create pipeline
    checkpoint_mgr = CheckpointManager(db_path)
    stages = [MockStageA(), MockStageB()]
    pipeline = PlanningPipeline(stages, checkpoint_mgr)
    client = MockLLMClient(responses={"stage_1": "Result A", "stage_2": "Result B"})

    # Execute fresh
    result = await pipeline.execute(
        client, "integration_job", "Build system", resume=False
    )

    assert result.success is True
    assert len(result.outputs) == 2

    # Verify checkpoint was saved
    checkpoint = await checkpoint_mgr.load_checkpoint("integration_job")
    assert checkpoint == result.outputs

    # Resume (should skip all stages)
    client2 = MockLLMClient()  # Fresh client
    result2 = await pipeline.execute(
        client2, "integration_job", "Build system", resume=True
    )

    assert result2.success is True
    assert result2.outputs == checkpoint
    assert client2.call_count == 0  # No LLM calls made
