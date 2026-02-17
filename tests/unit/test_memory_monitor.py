# tests/unit/test_memory_monitor.py
"""Tests for MemoryMonitor RAM threshold detection."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from fitz_planner_mcp.llm import MemoryMonitor


class TestMemoryMonitorCheckOnce:
    """Test single RAM check."""

    def test_check_once_below_threshold(self):
        """Returns (percent, False) when RAM usage is below threshold."""
        monitor = MemoryMonitor(threshold_percent=80.0)

        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value = MagicMock(percent=50.0)

            current, exceeded = monitor.check_once()

            assert current == 50.0
            assert exceeded is False

    def test_check_once_above_threshold(self):
        """Returns (percent, True) when RAM usage exceeds threshold."""
        monitor = MemoryMonitor(threshold_percent=80.0)

        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value = MagicMock(percent=85.0)

            current, exceeded = monitor.check_once()

            assert current == 85.0
            assert exceeded is True

    def test_check_once_at_threshold(self):
        """Returns (percent, True) when RAM usage equals threshold."""
        monitor = MemoryMonitor(threshold_percent=80.0)

        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value = MagicMock(percent=80.0)

            current, exceeded = monitor.check_once()

            assert current == 80.0
            assert exceeded is True


class TestMemoryMonitorStartMonitoring:
    """Test continuous monitoring."""

    @pytest.mark.asyncio
    async def test_start_monitoring_detects_threshold(self):
        """Monitoring returns True when threshold is exceeded."""
        monitor = MemoryMonitor(threshold_percent=80.0)

        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value = MagicMock(percent=90.0)

            result = await monitor.start_monitoring(check_interval=0.01)

            assert result is True

    @pytest.mark.asyncio
    async def test_start_monitoring_stops_cleanly(self):
        """Monitoring returns False when stopped via stop()."""
        monitor = MemoryMonitor(threshold_percent=80.0)

        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value = MagicMock(percent=50.0)

            # Start monitoring in background
            task = asyncio.create_task(monitor.start_monitoring(check_interval=0.01))

            # Let it run a bit
            await asyncio.sleep(0.05)

            # Stop it
            monitor.stop()

            # Should return False (stopped, not exceeded)
            result = await task
            assert result is False

    @pytest.mark.asyncio
    async def test_start_monitoring_threshold_after_delay(self):
        """Monitoring detects threshold when it's exceeded mid-monitoring."""
        monitor = MemoryMonitor(threshold_percent=80.0)

        call_count = 0

        def mock_memory():
            nonlocal call_count
            call_count += 1
            # First 2 checks: below threshold
            if call_count <= 2:
                return MagicMock(percent=50.0)
            # After that: above threshold
            return MagicMock(percent=90.0)

        with patch("psutil.virtual_memory", side_effect=mock_memory):
            result = await monitor.start_monitoring(check_interval=0.01)

            assert result is True
            assert call_count >= 3  # At least 3 checks


class TestMemoryMonitorCustomThreshold:
    """Test custom threshold values."""

    def test_custom_threshold_90_percent(self):
        """Monitor with 90% threshold."""
        monitor = MemoryMonitor(threshold_percent=90.0)

        with patch("psutil.virtual_memory") as mock_mem:
            # 85% should NOT exceed 90% threshold
            mock_mem.return_value = MagicMock(percent=85.0)
            _, exceeded = monitor.check_once()
            assert exceeded is False

            # 95% should exceed 90% threshold
            mock_mem.return_value = MagicMock(percent=95.0)
            _, exceeded = monitor.check_once()
            assert exceeded is True

    def test_custom_threshold_50_percent(self):
        """Monitor with 50% threshold (very conservative)."""
        monitor = MemoryMonitor(threshold_percent=50.0)

        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value = MagicMock(percent=60.0)

            _, exceeded = monitor.check_once()
            assert exceeded is True
