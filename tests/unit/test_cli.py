# tests/unit/test_cli.py
"""
CLI unit tests.

Tests each command via typer's CliRunner, using an in-memory job store
to avoid filesystem side effects.
"""

import pytest
from typer.testing import CliRunner
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from fitz_graveyard.cli import app
from fitz_graveyard.models.jobs import JobRecord, JobState

runner = CliRunner()

# 12-char hex IDs (matches generate_job_id format)
JOB_ID_1 = "abc123def456"
JOB_ID_2 = "789012abcdef"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_job(
    job_id=JOB_ID_1,
    description="build auth system",
    state=JobState.QUEUED,
    progress=0.0,
    quality_score=None,
    file_path=None,
    error=None,
    api_review=False,
    cost_estimate_json=None,
):
    return JobRecord(
        job_id=job_id,
        description=description,
        timeline=None,
        context=None,
        integration_points=[],
        state=state,
        progress=progress,
        current_phase=None,
        quality_score=quality_score,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        file_path=file_path,
        error=error,
        api_review=api_review,
        cost_estimate_json=cost_estimate_json,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHelp:
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        assert "Local-first AI architectural planning" in result.output

    def test_help_flag(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "plan" in result.output
        assert "run" in result.output
        assert "list" in result.output
        assert "serve" in result.output


class TestPlan:
    @patch("fitz_graveyard.cli._get_store")
    @patch("fitz_graveyard.cli.load_config", create=True)
    def test_plan_queues_job(self, mock_config, mock_store):
        """plan command calls create_plan and prints job ID."""
        mock_store_instance = AsyncMock()
        mock_store_instance.close = AsyncMock()
        mock_store.return_value = mock_store_instance

        with patch("fitz_graveyard.tools.create_plan.sanitize_description", return_value="test desc"):
            with patch("fitz_graveyard.tools.create_plan.generate_job_id", return_value="test12345678"):
                mock_store_instance.add = AsyncMock()
                result = runner.invoke(app, ["plan", "test desc"])

        assert result.exit_code == 0
        assert "test12345678" in result.output
        assert "fitz-graveyard run" in result.output


class TestList:
    @patch("fitz_graveyard.cli._get_store")
    def test_list_empty(self, mock_store):
        """list shows 'No plans' when empty."""
        mock_store_instance = AsyncMock()
        mock_store_instance.list_all = AsyncMock(return_value=[])
        mock_store_instance.close = AsyncMock()
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No plans" in result.output

    @patch("fitz_graveyard.cli._get_store")
    def test_list_with_jobs(self, mock_store):
        """list shows jobs in table format."""
        jobs = [
            _make_job(JOB_ID_1, "build auth", JobState.COMPLETE, 1.0, 0.82),
            _make_job(JOB_ID_2, "add caching", JobState.QUEUED),
        ]
        mock_store_instance = AsyncMock()
        mock_store_instance.list_all = AsyncMock(return_value=jobs)
        mock_store_instance.close = AsyncMock()
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert JOB_ID_1 in result.output
        assert JOB_ID_2 in result.output
        assert "complete" in result.output
        assert "queued" in result.output


class TestStatus:
    @patch("fitz_graveyard.cli._get_store")
    def test_status_queued(self, mock_store):
        """status shows job details."""
        job = _make_job()
        mock_store_instance = AsyncMock()
        mock_store_instance.get = AsyncMock(return_value=job)
        mock_store_instance.close = AsyncMock()
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["status", JOB_ID_1])
        assert result.exit_code == 0
        assert JOB_ID_1 in result.output
        assert "queued" in result.output
        assert "0%" in result.output

    @patch("fitz_graveyard.cli._get_store")
    def test_status_not_found(self, mock_store):
        """status exits with error for unknown job."""
        mock_store_instance = AsyncMock()
        mock_store_instance.get = AsyncMock(return_value=None)
        mock_store_instance.close = AsyncMock()
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["status", "nonexistent12"])
        assert result.exit_code == 1


class TestGet:
    @patch("fitz_graveyard.cli._get_store")
    @patch("builtins.open", new_callable=MagicMock)
    def test_get_complete_job(self, mock_open, mock_store):
        """get prints plan markdown."""
        plan_content = "# Plan for: build auth system\n\nSome plan content."
        mock_open.return_value.__enter__.return_value.read.return_value = plan_content

        job = _make_job(state=JobState.COMPLETE, quality_score=0.82, file_path="/tmp/plan.md")
        mock_store_instance = AsyncMock()
        mock_store_instance.get = AsyncMock(return_value=job)
        mock_store_instance.close = AsyncMock()
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["get", JOB_ID_1])
        assert result.exit_code == 0
        assert "build auth system" in result.output

    @patch("fitz_graveyard.cli._get_store")
    def test_get_incomplete_job_errors(self, mock_store):
        """get fails for non-complete jobs."""
        job = _make_job(state=JobState.RUNNING)
        mock_store_instance = AsyncMock()
        mock_store_instance.get = AsyncMock(return_value=job)
        mock_store_instance.close = AsyncMock()
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["get", JOB_ID_1])
        assert result.exit_code == 1


class TestRetry:
    @patch("fitz_graveyard.cli._get_store")
    def test_retry_failed_job(self, mock_store):
        """retry re-queues a failed job."""
        job = _make_job(state=JobState.FAILED, error="OOM")
        mock_store_instance = AsyncMock()
        mock_store_instance.get = AsyncMock(return_value=job)
        mock_store_instance.update = AsyncMock()
        mock_store_instance.close = AsyncMock()
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["retry", JOB_ID_1])
        assert result.exit_code == 0
        assert "re-queued" in result.output

    @patch("fitz_graveyard.cli._get_store")
    def test_retry_queued_job_errors(self, mock_store):
        """retry fails for non-retryable states."""
        job = _make_job(state=JobState.QUEUED)
        mock_store_instance = AsyncMock()
        mock_store_instance.get = AsyncMock(return_value=job)
        mock_store_instance.close = AsyncMock()
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["retry", JOB_ID_1])
        assert result.exit_code == 1


class TestConfirm:
    @patch("fitz_graveyard.cli._get_store")
    def test_confirm_awaiting_review(self, mock_store):
        """confirm approves API review."""
        job = _make_job(state=JobState.AWAITING_REVIEW, api_review=True)
        mock_store_instance = AsyncMock()
        mock_store_instance.get = AsyncMock(return_value=job)
        mock_store_instance.update = AsyncMock()
        mock_store_instance.close = AsyncMock()
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["confirm", JOB_ID_1])
        assert result.exit_code == 0
        assert "approved" in result.output

    @patch("fitz_graveyard.cli._get_store")
    def test_confirm_wrong_state_errors(self, mock_store):
        """confirm fails if job not awaiting review."""
        job = _make_job(state=JobState.QUEUED)
        mock_store_instance = AsyncMock()
        mock_store_instance.get = AsyncMock(return_value=job)
        mock_store_instance.close = AsyncMock()
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["confirm", JOB_ID_1])
        assert result.exit_code == 1


class TestCancel:
    @patch("fitz_graveyard.cli._get_store")
    def test_cancel_awaiting_review(self, mock_store):
        """cancel skips API review."""
        job = _make_job(state=JobState.AWAITING_REVIEW, api_review=True)
        mock_store_instance = AsyncMock()
        mock_store_instance.get = AsyncMock(return_value=job)
        mock_store_instance.update = AsyncMock()
        mock_store_instance.close = AsyncMock()
        mock_store.return_value = mock_store_instance

        result = runner.invoke(app, ["cancel", JOB_ID_1])
        assert result.exit_code == 0
        assert "skipped" in result.output
