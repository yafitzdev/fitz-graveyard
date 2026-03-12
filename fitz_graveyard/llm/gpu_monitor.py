# fitz_graveyard/llm/gpu_monitor.py
"""GPU temperature monitoring for thermal throttling during LLM generation."""

import asyncio
import logging
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)


class GPUTemperatureGuard:
    """
    Two-layer GPU thermal protection.

    Layer 1 — Pre-flight cooldown: called before each LLM generate() call.
    If the GPU is above ``threshold``, sleeps until it drops to
    ``threshold - cooldown_margin`` or the timeout expires.

    Layer 2 — Mid-stream throttle: called in the streaming loop between
    chunks.  When the GPU exceeds ``threshold``, inserts async sleeps
    that create TCP backpressure on llama-server, naturally slowing
    token generation.  Checks are rate-limited to avoid nvidia-smi spam.
    """

    def __init__(
        self,
        threshold: int = 73,
        cooldown_margin: int = 10,
        check_interval: float = 10.0,
        cooldown_timeout: int = 300,
    ):
        self.threshold = threshold
        self.cooldown_target = threshold - cooldown_margin
        self.check_interval = check_interval
        self.cooldown_timeout = cooldown_timeout
        self._last_check: float = 0.0

    # ------------------------------------------------------------------
    # Temperature query
    # ------------------------------------------------------------------

    @staticmethod
    def get_free_vram_mb() -> int | None:
        """Query free GPU VRAM via nvidia-smi.

        Returns:
            Free VRAM in MB, or None if nvidia-smi is unavailable.
        """
        smi = shutil.which("nvidia-smi")
        if not smi:
            return None
        try:
            result = subprocess.run(
                [
                    smi,
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            return int(result.stdout.strip().splitlines()[0])
        except (subprocess.TimeoutExpired, ValueError, IndexError, OSError):
            return None

    @staticmethod
    def get_gpu_temp() -> int | None:
        """Query GPU temperature via nvidia-smi.

        Returns:
            Temperature in °C, or None if nvidia-smi is unavailable.
        """
        smi = shutil.which("nvidia-smi")
        if not smi:
            return None
        try:
            result = subprocess.run(
                [
                    smi,
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            return int(result.stdout.strip().splitlines()[0])
        except (subprocess.TimeoutExpired, ValueError, IndexError, OSError):
            return None

    # ------------------------------------------------------------------
    # Layer 1: Pre-flight cooldown
    # ------------------------------------------------------------------

    async def preflight(self) -> None:
        """Wait if GPU is above threshold. Call before each LLM generate().

        Blocks until temperature drops to ``cooldown_target`` or the
        cooldown timeout is reached.  If the timeout expires the call
        proceeds anyway (driver thermal throttling will protect the GPU).

        Raises nothing — worst case it logs a warning and lets the call through.
        """
        temp = self.get_gpu_temp()
        if temp is None or temp < self.threshold:
            return

        logger.warning(
            f"GPU at {temp}°C (threshold {self.threshold}°C), "
            f"waiting for cooldown to {self.cooldown_target}°C..."
        )
        t0 = time.monotonic()

        while True:
            await asyncio.sleep(self.check_interval)
            temp = self.get_gpu_temp()
            elapsed = time.monotonic() - t0

            if temp is None or temp <= self.cooldown_target:
                logger.info(
                    f"GPU cooled to {temp}°C after {elapsed:.0f}s, resuming"
                )
                return

            if elapsed > self.cooldown_timeout:
                logger.warning(
                    f"GPU still at {temp}°C after {elapsed:.0f}s cooldown "
                    f"timeout, proceeding anyway"
                )
                return

            logger.info(
                f"GPU at {temp}°C, waiting... ({elapsed:.0f}s elapsed)"
            )

    # ------------------------------------------------------------------
    # Layer 2: Mid-stream throttle
    # ------------------------------------------------------------------

    async def maybe_throttle(self) -> None:
        """Insert a delay during streaming if GPU is too hot.

        Rate-limited: only queries nvidia-smi every ``check_interval``
        seconds. When the GPU exceeds the threshold, sleeps scale with
        how far above the threshold it is (0.5 s at threshold, up to 5 s
        at threshold+20 °C).  The sleep creates TCP backpressure that
        naturally slows token generation on the server side.
        """
        now = time.monotonic()
        if now - self._last_check < self.check_interval:
            return
        self._last_check = now

        temp = self.get_gpu_temp()
        if temp is None or temp < self.threshold:
            return

        overshoot = temp - self.threshold
        delay = min(0.5 + overshoot * 0.2, 5.0)
        logger.debug(
            f"GPU at {temp}°C (threshold {self.threshold}°C), "
            f"throttling stream — {delay:.1f}s pause"
        )
        await asyncio.sleep(delay)
