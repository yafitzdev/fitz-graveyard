# tests/unit/test_diagnostics.py
"""Tests for diagnostics: LLM call metrics, stage timings, and rendered output."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fitz_graveyard.llm.lm_studio import LMStudioClient
from fitz_graveyard.llm.client import OllamaClient
from fitz_graveyard.planning.pipeline.orchestrator import PipelineResult
from fitz_graveyard.planning.pipeline.output import PlanRenderer
from fitz_graveyard.planning.schemas.plan_output import PlanOutput
from fitz_graveyard.planning.schemas.context import ContextOutput
from fitz_graveyard.planning.schemas.architecture import ArchitectureOutput
from fitz_graveyard.planning.schemas.design import DesignOutput
from fitz_graveyard.planning.schemas.roadmap import RoadmapOutput
from fitz_graveyard.planning.schemas.risk import RiskOutput


def _minimal_plan(**kwargs) -> PlanOutput:
    """Build a PlanOutput with minimal valid data plus any overrides."""
    defaults = dict(
        context=ContextOutput(project_description="test"),
        architecture=ArchitectureOutput(recommended="A", reasoning="because"),
        design=DesignOutput(),
        roadmap=RoadmapOutput(),
        risk=RiskOutput(),
    )
    defaults.update(kwargs)
    return PlanOutput(**defaults)


# ---------------------------------------------------------------------------
# LMStudioClient call metrics
# ---------------------------------------------------------------------------

def _make_lm_client(**kwargs):
    with patch("fitz_graveyard.llm.lm_studio.AsyncOpenAI"):
        client = LMStudioClient(**kwargs)
    return client


async def _async_iter(items):
    for item in items:
        yield item


def _make_stream_chunks(texts):
    chunks = []
    for text in texts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = text
        chunks.append(chunk)
    return chunks


class TestLMStudioCallMetrics:
    @pytest.mark.asyncio
    async def test_generate_records_metrics(self):
        client = _make_lm_client(model="test-model")
        chunks = _make_stream_chunks(["Hello", " world"])
        client._client.chat.completions.create = AsyncMock(
            return_value=_async_iter(chunks)
        )

        await client.generate([{"role": "user", "content": "hi"}])

        metrics = client.drain_call_metrics()
        assert len(metrics) == 1
        assert metrics[0]["model"] == "test-model"
        assert metrics[0]["output_chars"] == 11
        assert metrics[0]["elapsed_s"] >= 0

    @pytest.mark.asyncio
    async def test_drain_clears_metrics(self):
        client = _make_lm_client(model="test-model")
        chunks = _make_stream_chunks(["ok"])
        client._client.chat.completions.create = AsyncMock(
            return_value=_async_iter(chunks)
        )

        await client.generate([{"role": "user", "content": "hi"}])
        first = client.drain_call_metrics()
        assert len(first) == 1

        second = client.drain_call_metrics()
        assert len(second) == 0

    @pytest.mark.asyncio
    async def test_multiple_generates_accumulate(self):
        client = _make_lm_client(model="test-model")

        for _ in range(3):
            chunks = _make_stream_chunks(["x"])
            client._client.chat.completions.create = AsyncMock(
                return_value=_async_iter(chunks)
            )
            await client.generate([{"role": "user", "content": "hi"}])

        metrics = client.drain_call_metrics()
        assert len(metrics) == 3


# ---------------------------------------------------------------------------
# OllamaClient call metrics
# ---------------------------------------------------------------------------

class TestOllamaCallMetrics:
    @pytest.mark.asyncio
    async def test_generate_records_metrics(self):
        client = OllamaClient(base_url="http://localhost:11434", model="test")

        async def fake_chat(**kwargs):
            async def gen():
                yield {"message": {"content": "Hello"}}
                yield {"message": {"content": " world"}}
            return gen()

        client.client.chat = AsyncMock(side_effect=fake_chat)

        await client.generate([{"role": "user", "content": "hi"}])

        metrics = client.drain_call_metrics()
        assert len(metrics) == 1
        assert metrics[0]["model"] == "test"
        assert metrics[0]["output_chars"] == 11
        assert metrics[0]["elapsed_s"] >= 0

    @pytest.mark.asyncio
    async def test_drain_clears_metrics(self):
        client = OllamaClient(base_url="http://localhost:11434", model="test")

        async def fake_chat(**kwargs):
            async def gen():
                yield {"message": {"content": "ok"}}
            return gen()

        client.client.chat = AsyncMock(side_effect=fake_chat)

        await client.generate([{"role": "user", "content": "hi"}])
        assert len(client.drain_call_metrics()) == 1
        assert len(client.drain_call_metrics()) == 0


# ---------------------------------------------------------------------------
# PipelineResult stage_timings
# ---------------------------------------------------------------------------

class TestPipelineResultTimings:
    def test_default_empty_dict(self):
        result = PipelineResult(success=True, outputs={})
        assert result.stage_timings == {}

    def test_preserves_stage_timings(self):
        timings = {"context": 10.5, "architecture_design": 20.3}
        result = PipelineResult(success=True, outputs={}, stage_timings=timings)
        assert result.stage_timings == timings


# ---------------------------------------------------------------------------
# PlanOutput diagnostics field
# ---------------------------------------------------------------------------

class TestPlanOutputDiagnostics:
    def test_default_empty_dict(self):
        plan = _minimal_plan()
        assert plan.diagnostics == {}

    def test_accepts_diagnostics(self):
        diag = {"provider": "lm_studio", "model": "test", "total_llm_calls": 5}
        plan = _minimal_plan(diagnostics=diag)
        assert plan.diagnostics["provider"] == "lm_studio"
        assert plan.diagnostics["total_llm_calls"] == 5


# ---------------------------------------------------------------------------
# Diagnostics rendering
# ---------------------------------------------------------------------------

class TestDiagnosticsRendering:
    def test_renders_diagnostics_table(self):
        plan = _minimal_plan(diagnostics={
            "provider": "lm_studio",
            "model": "qwen/qwen3-coder-30b",
            "agent_enabled": True,
            "total_llm_calls": 14,
            "total_generation_s": 342.1,
            "stage_timings_s": {
                "context": 89.2,
                "architecture_design": 156.3,
            },
        })
        renderer = PlanRenderer()
        md = renderer.render(plan)

        assert "## Diagnostics" in md
        assert "| Provider | lm_studio |" in md
        assert "| Model | qwen/qwen3-coder-30b |" in md
        assert "| Agent | enabled |" in md
        assert "| Total LLM calls | 14 |" in md
        assert "| Total generation time | 342.1s |" in md
        assert "| Stage: context | 89.2s |" in md
        assert "| Stage: architecture_design | 156.3s |" in md

    def test_no_diagnostics_section_when_empty(self):
        plan = _minimal_plan()
        renderer = PlanRenderer()
        md = renderer.render(plan)

        assert "## Diagnostics" not in md

    def test_agent_disabled_renders_correctly(self):
        plan = _minimal_plan(diagnostics={"agent_enabled": False})
        renderer = PlanRenderer()
        md = renderer.render(plan)

        assert "| Agent | disabled |" in md
