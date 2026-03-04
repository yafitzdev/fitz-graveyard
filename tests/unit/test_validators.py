# tests/unit/test_validators.py
"""Tests for post-extraction quality validators."""

import json
import pytest
from unittest.mock import AsyncMock

from fitz_graveyard.planning.pipeline.validators import (
    ensure_min_existing_files,
    ensure_min_adrs,
    ensure_phase_zero,
    ensure_concrete_verification,
    ensure_grounded_risks,
    _is_vague_verification,
)


# ---------------------------------------------------------------------------
# ensure_min_existing_files
# ---------------------------------------------------------------------------

class TestEnsureMinExistingFiles:
    def _make_summaries(self, paths: list[str]) -> str:
        return "\n".join(f"### {p}\nSome summary text." for p in paths)

    def test_backfilled_from_summaries(self):
        """< 6 files + summaries have more → backfilled."""
        merged = {"existing_files": [
            "src/a.py — does A",
            "src/b.py — does B",
            "src/c.py — does C",
            "src/d.py — does D",
        ]}
        prior = {"_raw_summaries": self._make_summaries([
            "src/a.py", "src/b.py", "src/c.py", "src/d.py",
            "src/e.py", "src/f.py", "src/g.py", "src/h.py",
        ])}
        result = ensure_min_existing_files(merged, prior)
        assert len(result["existing_files"]) >= 6
        paths = [e.split(" — ")[0].split(" - ")[0].strip() for e in result["existing_files"]]
        assert "src/e.py" in paths

    def test_backfills_even_when_enough(self):
        """Always backfills missing summarized files, even if >= 6."""
        files = [f"src/{c}.py — does {c}" for c in "abcdefg"]
        merged = {"existing_files": files[:]}
        prior = {"_raw_summaries": self._make_summaries(["src/z.py"])}
        result = ensure_min_existing_files(merged, prior)
        paths = [e.split(" — ")[0].split(" - ")[0].strip() for e in result["existing_files"]]
        assert "src/z.py" in paths

    def test_noop_without_summaries(self):
        """No raw summaries → unchanged."""
        merged = {"existing_files": ["src/a.py — A"]}
        result = ensure_min_existing_files(merged, {})
        assert len(result["existing_files"]) == 1

    def test_max_cap(self):
        """Never exceeds MAX_EXISTING_FILES (25)."""
        merged = {"existing_files": ["src/a.py — A"]}
        paths = [f"src/{i}.py" for i in range(30)]
        prior = {"_raw_summaries": self._make_summaries(paths)}
        result = ensure_min_existing_files(merged, prior)
        assert len(result["existing_files"]) <= 25


# ---------------------------------------------------------------------------
# ensure_min_adrs
# ---------------------------------------------------------------------------

class TestEnsureMinAdrs:
    @pytest.mark.asyncio
    async def test_adrs_added_when_below_minimum(self):
        """1 ADR → LLM adds more → now >= 2."""
        merged = {
            "adrs": [{"title": "ADR: Existing", "context": "c", "decision": "d", "rationale": "r",
                       "consequences": [], "alternatives_considered": []}],
            "recommended": "Approach A",
            "reasoning": "Some reasoning",
            "components": [{"name": "Comp1"}],
        }
        client = AsyncMock()
        client.generate.return_value = json.dumps([
            {"title": "ADR: New One", "context": "ctx", "decision": "dec",
             "rationale": "rat", "consequences": ["c1"], "alternatives_considered": ["a1"]},
        ])
        result = await ensure_min_adrs(merged, client, {}, "reasoning text")
        assert len(result["adrs"]) >= 2
        assert client.generate.called

    @pytest.mark.asyncio
    async def test_adrs_noop_when_enough(self):
        """>=2 ADRs → unchanged, no LLM call."""
        adrs = [
            {"title": f"ADR: {i}", "context": "c", "decision": "d", "rationale": "r",
             "consequences": [], "alternatives_considered": []}
            for i in range(3)
        ]
        merged = {"adrs": adrs[:]}
        client = AsyncMock()
        result = await ensure_min_adrs(merged, client, {}, "reasoning")
        assert len(result["adrs"]) == 3
        assert not client.generate.called

    @pytest.mark.asyncio
    async def test_adrs_llm_failure_no_crash(self):
        """LLM failure → original ADRs kept, no crash."""
        merged = {"adrs": [{"title": "ADR: Solo"}], "recommended": "A", "reasoning": "r", "components": []}
        client = AsyncMock()
        client.generate.side_effect = RuntimeError("LLM down")
        result = await ensure_min_adrs(merged, client, {}, "reasoning")
        assert len(result["adrs"]) == 1


# ---------------------------------------------------------------------------
# ensure_phase_zero
# ---------------------------------------------------------------------------

class TestEnsurePhaseZero:
    def _make_summaries(self, paths: list[str]) -> str:
        return "\n".join(f"### {p}\nSummary." for p in paths)

    def test_phase_zero_injected(self):
        """No Phase 0 → injected with discovered files."""
        merged = {
            "phases": [
                {"number": 1, "name": "Build", "dependencies": [], "deliverables": ["x"]},
                {"number": 2, "name": "Test", "dependencies": [1], "deliverables": ["y"]},
            ],
            "critical_path": [1, 2],
            "total_phases": 2,
        }
        prior = {"_raw_summaries": self._make_summaries(["src/main.py", "src/config.py", "src/utils.py"])}
        result = ensure_phase_zero(merged, prior)
        assert result["phases"][0]["number"] == 0
        assert result["phases"][0]["name"] == "Read Before Writing"
        assert len(result["phases"][0]["deliverables"]) == 3

    def test_phase_zero_noop_when_exists(self):
        """Phase 0 already exists → unchanged."""
        merged = {
            "phases": [
                {"number": 0, "name": "Existing Phase 0", "dependencies": []},
                {"number": 1, "name": "Build", "dependencies": [0]},
            ],
        }
        prior = {"_raw_summaries": "### src/a.py\nStuff."}
        result = ensure_phase_zero(merged, prior)
        assert result["phases"][0]["name"] == "Existing Phase 0"

    def test_phase_numbers_bumped(self):
        """Existing phases renumbered after Phase 0 injection."""
        merged = {
            "phases": [
                {"number": 1, "name": "Build", "dependencies": []},
                {"number": 2, "name": "Test", "dependencies": [1]},
            ],
            "total_phases": 2,
        }
        prior = {"_raw_summaries": "### src/a.py\nStuff."}
        result = ensure_phase_zero(merged, prior)
        numbers = [p["number"] for p in result["phases"]]
        assert numbers == [0, 2, 3]
        # Phase that was 2 (now 3) should depend on 2 (was 1)
        assert result["phases"][2]["dependencies"] == [2]
        assert result["total_phases"] == 3

    def test_noop_without_summaries(self):
        """No raw summaries → no Phase 0."""
        merged = {"phases": [{"number": 1, "name": "Build", "dependencies": []}]}
        result = ensure_phase_zero(merged, {})
        assert len(result["phases"]) == 1


# ---------------------------------------------------------------------------
# ensure_concrete_verification
# ---------------------------------------------------------------------------

class TestEnsureConcreteVerification:
    @pytest.mark.asyncio
    async def test_vague_verification_replaced(self):
        """"run tests" → concrete command via LLM."""
        merged = {
            "phases": [
                {"number": 1, "name": "Build", "verification_command": "run tests",
                 "deliverables": ["src/widget.py"]},
            ],
        }
        client = AsyncMock()
        client.generate.return_value = json.dumps({"1": "python -m pytest tests/unit/test_widget.py -v"})
        result = await ensure_concrete_verification(merged, client, "reasoning")
        assert "test_widget" in result["phases"][0]["verification_command"]

    @pytest.mark.asyncio
    async def test_concrete_verification_kept(self):
        """Already has path → unchanged, no LLM call."""
        merged = {
            "phases": [
                {"number": 1, "name": "Build",
                 "verification_command": "python -m pytest tests/unit/test_foo.py -v",
                 "deliverables": []},
            ],
        }
        client = AsyncMock()
        result = await ensure_concrete_verification(merged, client, "reasoning")
        assert not client.generate.called
        assert "test_foo" in result["phases"][0]["verification_command"]

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        """LLM fails → template fallback applied."""
        merged = {
            "phases": [
                {"number": 1, "name": "Build", "verification_command": "verify",
                 "deliverables": ["src/widget.py"]},
            ],
        }
        client = AsyncMock()
        client.generate.side_effect = RuntimeError("LLM down")
        result = await ensure_concrete_verification(merged, client, "reasoning")
        cmd = result["phases"][0]["verification_command"]
        assert "pytest" in cmd


class TestIsVagueVerification:
    @pytest.mark.parametrize("cmd", [
        "run tests", "Run Tests", "verify", "check that it works",
        "test it", "manual", "pytest", "python", "# run tests",
    ])
    def test_vague_detected(self, cmd):
        assert _is_vague_verification(cmd) is True

    @pytest.mark.parametrize("cmd", [
        "python -m pytest tests/unit/test_foo.py -v",
        "curl localhost:8080/health",
        "python -c 'from mod import func; func()'",
    ])
    def test_concrete_detected(self, cmd):
        assert _is_vague_verification(cmd) is False


# ---------------------------------------------------------------------------
# ensure_grounded_risks
# ---------------------------------------------------------------------------

class TestEnsureGroundedRisks:
    def test_ungrounded_risks_removed(self):
        """Risk mentions fake file → removed."""
        merged = {
            "risks": [
                {"description": "Bug in src/real.py parser", "mitigation": "test"},
                {"description": "Bug in src/fake_nonexistent.py handler", "mitigation": "test"},
                {"description": "Timeout in src/real.py calls", "mitigation": "retry"},
            ],
        }
        prior = {
            "_gathered_context": "src/real.py is the main parser",
            "_raw_summaries": "### src/real.py\nParser module.",
        }
        result = ensure_grounded_risks(merged, prior)
        descs = [r["description"] for r in result["risks"]]
        assert "Bug in src/fake_nonexistent.py handler" not in descs
        assert len(result["risks"]) == 2

    def test_grounded_risks_kept(self):
        """Risk mentions real file from context → kept."""
        merged = {
            "risks": [
                {"description": "Race condition in src/worker.py", "mitigation": "lock"},
            ],
        }
        prior = {
            "_gathered_context": "src/worker.py handles background jobs",
            "_raw_summaries": "### src/worker.py\nWorker.",
        }
        result = ensure_grounded_risks(merged, prior)
        assert len(result["risks"]) == 1

    def test_minimum_risks_preserved(self):
        """Would drop below 2 → kept."""
        merged = {
            "risks": [
                {"description": "Bug in src/fake1.py", "mitigation": "a"},
                {"description": "Bug in src/fake2.py", "mitigation": "b"},
            ],
        }
        prior = {"_gathered_context": "nothing relevant here"}
        result = ensure_grounded_risks(merged, prior)
        assert len(result["risks"]) >= 2

    def test_conceptual_risks_kept(self):
        """Risks without file references → kept (conceptual risk)."""
        merged = {
            "risks": [
                {"description": "Token limit exceeded during generation", "mitigation": "truncate"},
            ],
        }
        prior = {"_gathered_context": "some context"}
        result = ensure_grounded_risks(merged, prior)
        assert len(result["risks"]) == 1

    def test_noop_without_context(self):
        """No context available → all kept."""
        merged = {
            "risks": [
                {"description": "Bug in src/fake.py", "mitigation": "test"},
            ],
        }
        result = ensure_grounded_risks(merged, {})
        assert len(result["risks"]) == 1
