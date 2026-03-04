# tests/unit/test_llama_cpp_client.py
"""Unit tests for LlamaCppClient subprocess management and tier switching."""

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fitz_graveyard.config.schema import LlamaCppModelConfig
from fitz_graveyard.llm.llama_cpp import LlamaCppClient


def _make_client(**kwargs):
    defaults = dict(
        server_path="/usr/bin/llama-server",
        models_dir="/models",
        fast_model=LlamaCppModelConfig(
            path="fast.gguf", context_size=32768, gpu_layers=-1,
        ),
        smart_model=LlamaCppModelConfig(
            path="smart.gguf", context_size=8192, gpu_layers=-1,
        ),
        port=18080,
        timeout=30,
        startup_timeout=5,
    )
    defaults.update(kwargs)
    return LlamaCppClient(**defaults)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------
class TestProperties:
    def test_fast_model(self):
        client = _make_client()
        assert client.fast_model == "fast.gguf"

    def test_smart_model(self):
        client = _make_client()
        assert client.smart_model == "smart.gguf"

    def test_smart_defaults_to_fast(self):
        client = _make_client(smart_model=None)
        assert client.smart_model == "fast.gguf"

    def test_model_defaults_to_fast_path(self):
        client = _make_client()
        assert client.model == "fast.gguf"

    def test_base_url(self):
        client = _make_client(port=9999)
        assert client.base_url == "http://127.0.0.1:9999/v1"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_spawns_process(self):
        client = _make_client()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stderr = MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc) as popen, \
             patch.object(client, "_wait_for_ready", new_callable=AsyncMock):
            await client.start("fast")

        popen.assert_called_once()
        cmd = popen.call_args[0][0]
        assert "/usr/bin/llama-server" in cmd
        assert "-m" in cmd
        # Should use fast model path
        model_idx = cmd.index("-m")
        assert "fast.gguf" in cmd[model_idx + 1]
        assert client._active_tier == "fast"

    @pytest.mark.asyncio
    async def test_start_smart_tier(self):
        client = _make_client()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch.object(client, "_wait_for_ready", new_callable=AsyncMock):
            await client.start("smart")

        assert client._active_tier == "smart"
        assert client.model == "smart.gguf"

    @pytest.mark.asyncio
    async def test_stop_terminates_process(self):
        client = _make_client()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        client._process = mock_proc
        client._active_tier = "fast"

        await client.stop()

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once()
        assert client._process is None
        assert client._active_tier is None

    @pytest.mark.asyncio
    async def test_stop_kills_on_timeout(self):
        client = _make_client()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="", timeout=10),
            None,
        ]
        client._process = mock_proc
        client._active_tier = "fast"

        await client.stop()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_noop_when_not_started(self):
        client = _make_client()
        await client.stop()  # Should not raise


# ---------------------------------------------------------------------------
# Tier switching
# ---------------------------------------------------------------------------
class TestEnsureTier:
    @pytest.mark.asyncio
    async def test_no_switch_needed(self):
        client = _make_client()
        client._active_tier = "fast"
        with patch.object(client, "start", new_callable=AsyncMock) as start, \
             patch.object(client, "stop", new_callable=AsyncMock) as stop:
            await client._ensure_tier("fast.gguf")
        start.assert_not_called()
        stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_switches_fast_to_smart(self):
        client = _make_client()
        client._active_tier = "fast"
        with patch.object(client, "start", new_callable=AsyncMock) as start, \
             patch.object(client, "stop", new_callable=AsyncMock) as stop:
            await client._ensure_tier("smart.gguf")
        stop.assert_called_once()
        start.assert_called_once_with("smart")

    @pytest.mark.asyncio
    async def test_switches_smart_to_fast(self):
        client = _make_client()
        client._active_tier = "smart"
        with patch.object(client, "start", new_callable=AsyncMock) as start, \
             patch.object(client, "stop", new_callable=AsyncMock) as stop:
            await client._ensure_tier("fast.gguf")
        stop.assert_called_once()
        start.assert_called_once_with("fast")

    @pytest.mark.asyncio
    async def test_auto_starts_if_not_running(self):
        client = _make_client()
        assert client._active_tier is None
        with patch.object(client, "start", new_callable=AsyncMock) as start:
            await client._ensure_tier(None)
        start.assert_called_once_with("fast")

    @pytest.mark.asyncio
    async def test_none_model_keeps_current(self):
        client = _make_client()
        client._active_tier = "smart"
        with patch.object(client, "start", new_callable=AsyncMock) as start:
            await client._ensure_tier(None)
        start.assert_not_called()


# ---------------------------------------------------------------------------
# Health check and alive check
# ---------------------------------------------------------------------------
class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy_server(self):
        client = _make_client()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        client._process = mock_proc

        with patch("httpx.AsyncClient") as mock_http_cls:
            mock_http = AsyncMock()
            mock_http_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_http
            )
            mock_http_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_http.get = AsyncMock(return_value=mock_resp)

            result = await client.health_check()
        assert result is True

    def test_check_alive_raises_when_not_started(self):
        client = _make_client()
        with pytest.raises(RuntimeError, match="not started"):
            client._check_alive()

    def test_check_alive_raises_on_crash(self):
        client = _make_client()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b"segfault"
        client._process = mock_proc
        with pytest.raises(RuntimeError, match="crashed"):
            client._check_alive()


# ---------------------------------------------------------------------------
# Wait for ready
# ---------------------------------------------------------------------------
class TestWaitForReady:
    @pytest.mark.asyncio
    async def test_process_crash_during_wait(self):
        client = _make_client(startup_timeout=2)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # exited
        mock_proc.returncode = 1
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b"error: model not found"
        client._process = mock_proc

        with pytest.raises(RuntimeError, match="exited with code 1"):
            await client._wait_for_ready()


# ---------------------------------------------------------------------------
# Call metrics
# ---------------------------------------------------------------------------
class TestCallMetrics:
    def test_drain_returns_and_clears(self):
        client = _make_client()
        client._call_metrics = [
            {"elapsed_s": 1.0, "output_chars": 100, "model": "fast.gguf"}
        ]
        metrics = client.drain_call_metrics()
        assert len(metrics) == 1
        assert client._call_metrics == []

    def test_drain_empty(self):
        client = _make_client()
        assert client.drain_call_metrics() == []


# ---------------------------------------------------------------------------
# Tool result message
# ---------------------------------------------------------------------------
class TestToolResultMessage:
    def test_format(self):
        client = _make_client()
        msg = client.tool_result_message("call-123", "result text")
        assert msg == {
            "role": "tool",
            "tool_call_id": "call-123",
            "content": "result text",
        }
