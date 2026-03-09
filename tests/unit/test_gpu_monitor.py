# tests/unit/test_gpu_monitor.py
"""Tests for GPU temperature monitoring and thermal throttling."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from fitz_graveyard.llm.gpu_monitor import GPUTemperatureGuard


class TestGetGpuTemp:
    """Tests for nvidia-smi temperature query."""

    def test_returns_int_on_success(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "42\n"
            assert GPUTemperatureGuard.get_gpu_temp() == 42

    def test_returns_none_when_no_nvidia_smi(self):
        with patch("shutil.which", return_value=None):
            assert GPUTemperatureGuard.get_gpu_temp() is None

    def test_returns_none_on_bad_returncode(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
                assert GPUTemperatureGuard.get_gpu_temp() is None

    def test_returns_none_on_timeout(self):
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
                assert GPUTemperatureGuard.get_gpu_temp() is None

    def test_multi_gpu_returns_first(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "55\n62\n"
            assert GPUTemperatureGuard.get_gpu_temp() == 55


class TestPreflight:
    """Tests for pre-flight cooldown check."""

    @pytest.mark.asyncio
    async def test_no_delay_when_below_threshold(self):
        guard = GPUTemperatureGuard(threshold=80)
        with patch.object(guard, "get_gpu_temp", return_value=65):
            await guard.preflight()  # Should return immediately

    @pytest.mark.asyncio
    async def test_no_delay_when_nvidia_smi_unavailable(self):
        guard = GPUTemperatureGuard(threshold=80)
        with patch.object(guard, "get_gpu_temp", return_value=None):
            await guard.preflight()  # Should return immediately

    @pytest.mark.asyncio
    async def test_waits_then_resumes_on_cooldown(self):
        guard = GPUTemperatureGuard(threshold=80, cooldown_margin=10, check_interval=0.01)
        temps = iter([82, 78, 69])  # Above threshold → cooling → below cooldown target
        with patch.object(guard, "get_gpu_temp", side_effect=temps):
            await guard.preflight()
            # Should have waited, then resumed when temp hit 69 (< 70 cooldown target)

    @pytest.mark.asyncio
    async def test_proceeds_on_cooldown_timeout(self):
        guard = GPUTemperatureGuard(
            threshold=80, cooldown_margin=10, check_interval=0.01, cooldown_timeout=0,
        )
        with patch.object(guard, "get_gpu_temp", return_value=85):
            await guard.preflight()  # Should proceed after timeout


class TestMaybeThrottle:
    """Tests for mid-stream throttle check."""

    @pytest.mark.asyncio
    async def test_no_delay_below_threshold(self):
        guard = GPUTemperatureGuard(threshold=80, check_interval=0)
        with patch.object(guard, "get_gpu_temp", return_value=65):
            await guard.maybe_throttle()  # Should not sleep

    @pytest.mark.asyncio
    async def test_delays_above_threshold(self):
        guard = GPUTemperatureGuard(threshold=80, check_interval=0)
        guard._last_check = 0  # Force check
        with patch.object(guard, "get_gpu_temp", return_value=82):
            with patch(
                "fitz_graveyard.llm.gpu_monitor.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep:
                await guard.maybe_throttle()
                mock_sleep.assert_called_once()
                # 0.5 + (82-80) * 0.2 = 0.9
                assert abs(mock_sleep.call_args[0][0] - 0.9) < 0.01

    @pytest.mark.asyncio
    async def test_delay_caps_at_5s(self):
        guard = GPUTemperatureGuard(threshold=80, check_interval=0)
        guard._last_check = 0
        with patch.object(guard, "get_gpu_temp", return_value=110):
            with patch(
                "fitz_graveyard.llm.gpu_monitor.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep:
                await guard.maybe_throttle()
                assert mock_sleep.call_args[0][0] == 5.0

    @pytest.mark.asyncio
    async def test_rate_limited(self):
        guard = GPUTemperatureGuard(threshold=80, check_interval=60)
        # Simulate recent check
        import time
        guard._last_check = time.monotonic()
        with patch.object(guard, "get_gpu_temp") as mock_temp:
            await guard.maybe_throttle()
            mock_temp.assert_not_called()  # Should skip due to rate limiting

    @pytest.mark.asyncio
    async def test_no_delay_when_nvidia_smi_unavailable(self):
        guard = GPUTemperatureGuard(threshold=80, check_interval=0)
        guard._last_check = 0
        with patch.object(guard, "get_gpu_temp", return_value=None):
            await guard.maybe_throttle()  # Should not sleep


class TestConfig:
    """Tests for GPUConfig integration."""

    def test_gpu_config_defaults(self):
        from fitz_graveyard.config.schema import GPUConfig

        gpu = GPUConfig()
        assert gpu.temp_threshold == 73
        assert gpu.cooldown_margin == 10

    def test_gpu_config_on_root(self):
        from fitz_graveyard.config.schema import FitzPlannerConfig

        config = FitzPlannerConfig()
        assert config.gpu.temp_threshold == 73

    def test_gpu_config_extra_ignored(self):
        from fitz_graveyard.config.schema import FitzPlannerConfig

        config = FitzPlannerConfig(gpu={"temp_threshold": 75, "unknown_field": 99})
        assert config.gpu.temp_threshold == 75

    def test_factory_creates_guard(self):
        from fitz_graveyard.config.schema import FitzPlannerConfig
        from fitz_graveyard.llm.factory import _create_gpu_guard

        config = FitzPlannerConfig(gpu={"temp_threshold": 75})
        guard = _create_gpu_guard(config)
        assert guard is not None
        assert guard.threshold == 75
        assert guard.cooldown_target == 65

    def test_factory_disabled_when_zero(self):
        from fitz_graveyard.config.schema import FitzPlannerConfig
        from fitz_graveyard.llm.factory import _create_gpu_guard

        config = FitzPlannerConfig(gpu={"temp_threshold": 0})
        assert _create_gpu_guard(config) is None
