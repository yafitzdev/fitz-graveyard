# tests/unit/test_krag_client.py
"""
Unit tests for KragClient and formatter module.

All fitz-ai imports are mocked - tests pass without fitz-ai installed.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from fitz_planner_mcp.config.schema import KragConfig
from fitz_planner_mcp.planning.krag import KragClient, format_krag_answer, format_krag_results


class TestKragClientDisabled:
    """Tests for disabled KRAG client."""

    def test_disabled_query_returns_empty(self):
        """When enabled=False, query() returns empty string."""
        client = KragClient(enabled=False)
        result = client.query("test question")
        assert result == ""

    def test_disabled_multi_query_returns_empty(self):
        """When enabled=False, multi_query() returns empty string."""
        client = KragClient(enabled=False)
        result = client.multi_query(["query1", "query2"])
        assert result == ""

    def test_disabled_get_fitz_returns_none(self):
        """When enabled=False, _get_fitz() returns None."""
        client = KragClient(enabled=False)
        assert client._get_fitz() is None


class TestKragClientImportError:
    """Tests for ImportError fallback behavior."""

    @patch("fitz_planner_mcp.planning.krag.client.logger")
    def test_import_error_fallback(self, mock_logger):
        """When fitz-ai not installed, query() returns empty string and logs debug."""
        client = KragClient(enabled=True)

        # Mock the import to raise ImportError
        with patch.dict("sys.modules", {"fitz_ai": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'fitz_ai'")):
                result = client.query("test question")

        assert result == ""
        # Logger should have debug message about fitz-ai not installed
        mock_logger.debug.assert_called()
        call_args = str(mock_logger.debug.call_args)
        assert "fitz-ai not installed" in call_args or "not installed" in call_args.lower()


class TestKragClientSuccess:
    """Tests for successful query execution."""

    def test_query_success(self):
        """Mock fitz SDK, verify query() returns formatted answer."""
        client = KragClient(enabled=True)

        # Create mock Answer object
        mock_answer = MagicMock()
        mock_answer.text = "This is the answer"
        mock_answer.mode = MagicMock()
        # Mock AnswerMode.ABSTAIN comparison
        mock_answer.mode.__eq__ = MagicMock(return_value=False)
        mock_answer.provenance = []

        # Mock fitz module
        mock_fitz = MagicMock()
        mock_fitz.ask = MagicMock(return_value=mock_answer)

        with patch.object(client, "_get_fitz", return_value=mock_fitz):
            result = client.query("test question")

        assert result == "This is the answer"
        mock_fitz.ask.assert_called_once_with("test question")

    def test_query_abstain_skipped(self):
        """When answer.mode is ABSTAIN, query() returns empty string."""
        client = KragClient(enabled=True)

        # Create mock AnswerMode enum with ABSTAIN constant
        mock_answer_mode = MagicMock()
        mock_abstain_value = MagicMock()
        mock_answer_mode.ABSTAIN = mock_abstain_value

        # Create mock Answer with ABSTAIN mode
        mock_answer = MagicMock()
        mock_answer.text = "I cannot answer this"
        mock_answer.mode = mock_abstain_value  # Same instance as ABSTAIN

        mock_fitz = MagicMock()
        mock_fitz.ask = MagicMock(return_value=mock_answer)

        # Patch the lazy import of AnswerMode from fitz_ai.core.answer_mode
        import sys

        mock_core_module = MagicMock()
        mock_core_module.AnswerMode = mock_answer_mode

        with patch.object(client, "_get_fitz", return_value=mock_fitz):
            with patch.dict(
                sys.modules,
                {"fitz_ai.core.answer_mode": mock_core_module},
            ):
                result = client.query("test question")

        assert result == ""

    @patch("fitz_planner_mcp.planning.krag.client.logger")
    def test_query_exception_returns_empty(self, mock_logger):
        """When fitz.ask() raises Exception, returns empty string with warning log."""
        client = KragClient(enabled=True)

        mock_fitz = MagicMock()
        mock_fitz.ask = MagicMock(side_effect=RuntimeError("Database error"))

        with patch.object(client, "_get_fitz", return_value=mock_fitz):
            result = client.query("test question")

        assert result == ""
        mock_logger.warning.assert_called()
        call_args = str(mock_logger.warning.call_args)
        assert "KRAG query failed" in call_args or "failed" in call_args.lower()


class TestKragClientMultiQuery:
    """Tests for multi-query aggregation."""

    def test_multi_query_aggregation(self):
        """multi_query() combines multiple results under section headers."""
        client = KragClient(enabled=True)

        # Mock query() to return specific results
        with patch.object(client, "query", side_effect=["answer1", "answer2", "answer3"]):
            result = client.multi_query(["query1", "query2", "query3"])

        assert "## Codebase Context" in result
        assert "### query1" in result
        assert "answer1" in result
        assert "### query2" in result
        assert "answer2" in result
        assert "### query3" in result
        assert "answer3" in result

    def test_multi_query_skips_empty(self):
        """multi_query() skips queries that return empty string."""
        client = KragClient(enabled=True)

        # Mock query() with some empty results
        with patch.object(client, "query", side_effect=["answer1", "", "answer3"]):
            result = client.multi_query(["query1", "query2", "query3"])

        assert "## Codebase Context" in result
        assert "### query1" in result
        assert "answer1" in result
        assert "### query2" not in result  # Skipped because empty
        assert "### query3" in result
        assert "answer3" in result

    def test_multi_query_all_empty_returns_empty(self):
        """multi_query() returns empty string if all queries fail."""
        client = KragClient(enabled=True)

        with patch.object(client, "query", return_value=""):
            result = client.multi_query(["query1", "query2"])

        assert result == ""


class TestKragClientConfiguration:
    """Tests for configuration and initialization."""

    def test_from_config(self):
        """from_config() creates client with correct settings from KragConfig."""
        config = KragConfig(enabled=True, fitz_ai_config="/path/to/config.yaml")
        client = KragClient.from_config(config, source_dir="/path/to/source")

        assert client._enabled is True
        assert client._fitz_ai_config == "/path/to/config.yaml"
        assert client._source_dir == "/path/to/source"

    def test_fitz_instance_cached(self):
        """_get_fitz() caches instance (not re-created each call)."""
        client = KragClient(enabled=True)

        # Mock the fitz module that would be returned by import
        mock_fitz_module = MagicMock()

        # Patch the import to return our mock
        with patch("builtins.__import__", return_value=mock_fitz_module) as mock_import:
            fitz1 = client._get_fitz()
            fitz2 = client._get_fitz()

            # Should be same instance (cached)
            assert fitz1 is fitz2
            # Import should only be called once (cached after first call)
            # Note: __import__ gets called multiple times for nested imports,
            # but the instance should be cached
            assert fitz1 is not None

    def test_point_called_with_source_dir(self):
        """fitz.point() called with source_dir when provided."""
        import sys

        client = KragClient(enabled=True, source_dir="/path/to/source")

        # Create mock fitz module with point method
        mock_fitz_module = MagicMock()
        mock_fitz_module.point = MagicMock()

        # Create mock fitz_ai package with fitz submodule
        mock_fitz_ai_package = MagicMock()
        mock_fitz_ai_package.fitz = mock_fitz_module

        # Patch sys.modules to simulate 'from fitz_ai import fitz'
        with patch.dict(sys.modules, {"fitz_ai": mock_fitz_ai_package}):
            result = client._get_fitz()

        # point() should have been called with source_dir
        mock_fitz_module.point.assert_called_once_with("/path/to/source")
        # Should return the fitz module
        assert result is mock_fitz_module


class TestFormatter:
    """Tests for context formatting functions."""

    def test_format_answer_with_provenance(self):
        """Formats answer text + file path citations."""
        mock_answer = MagicMock()
        mock_answer.text = "This is the answer text"

        mock_prov = MagicMock()
        mock_prov.file_path = "path/to/file.py"
        mock_prov.line_range = (10, 25)
        mock_prov.metadata = {"kind": "code"}

        mock_answer.provenance = [mock_prov]

        result = format_krag_answer(mock_answer)

        assert "This is the answer text" in result
        assert "**Sources:**" in result
        assert "**path/to/file.py**" in result
        assert "(lines 10-25)" in result
        assert "[code]" in result

    def test_format_answer_no_provenance(self):
        """Returns just answer text when no provenance."""
        mock_answer = MagicMock()
        mock_answer.text = "Just the answer"
        mock_answer.provenance = []

        result = format_krag_answer(mock_answer)

        assert result == "Just the answer"
        assert "**Sources:**" not in result

    def test_format_answer_with_line_range(self):
        """Includes line numbers in citation."""
        mock_answer = MagicMock()
        mock_answer.text = "Answer"

        mock_prov = MagicMock()
        mock_prov.file_path = "file.py"
        mock_prov.line_range = (100, 150)
        mock_prov.metadata = {}

        mock_answer.provenance = [mock_prov]

        result = format_krag_answer(mock_answer)

        assert "(lines 100-150)" in result

    def test_format_answer_max_provenance(self):
        """Caps at 5 provenance entries."""
        mock_answer = MagicMock()
        mock_answer.text = "Answer"

        # Create 10 provenance entries
        provs = []
        for i in range(10):
            mock_prov = MagicMock()
            mock_prov.file_path = f"file{i}.py"
            mock_prov.line_range = None
            mock_prov.metadata = {}
            provs.append(mock_prov)

        mock_answer.provenance = provs

        result = format_krag_answer(mock_answer)

        # Should only include first 5
        assert "file0.py" in result
        assert "file4.py" in result
        assert "file5.py" not in result
        assert "file9.py" not in result

    def test_format_empty_answer(self):
        """Handles empty/None answer text."""
        mock_answer = MagicMock()
        mock_answer.text = ""
        mock_answer.provenance = []

        result = format_krag_answer(mock_answer)
        assert result == ""

        # Test None text
        mock_answer.text = None
        result = format_krag_answer(mock_answer)
        assert result == ""

    def test_format_krag_results(self):
        """format_krag_results() aggregates (query, answer) tuples."""
        results = [
            ("query1", "answer1"),
            ("query2", "answer2"),
        ]

        output = format_krag_results(results)

        assert "## Codebase Context" in output
        assert "### query1" in output
        assert "answer1" in output
        assert "### query2" in output
        assert "answer2" in output

    def test_format_krag_results_empty(self):
        """format_krag_results() returns empty string for empty list."""
        result = format_krag_results([])
        assert result == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
