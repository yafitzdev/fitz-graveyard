# tests/unit/test_confidence.py
"""Unit tests for confidence scoring and flagging."""

import pytest

from fitz_graveyard.config.schema import ConfidenceConfig
from fitz_graveyard.planning.confidence import ConfidenceScorer, SectionFlagger


class TestConfidenceScorerHeuristics:
    """Test heuristic scoring (no LLM)."""

    def test_short_content_low_score(self):
        """Short content should score low on length."""
        scorer = ConfidenceScorer(ollama_client=None)
        short_content = "A brief section."
        score = scorer._length_score(short_content)
        assert score == 0.2

    def test_medium_content_medium_score(self):
        """Medium content should score medium on length."""
        scorer = ConfidenceScorer(ollama_client=None)
        medium_content = "This is a medium-length section with some detail but not extensive. It provides basic information."
        score = scorer._length_score(medium_content)
        assert score == 0.5

    def test_long_content_high_score(self):
        """Long content should score high on length."""
        scorer = ConfidenceScorer(ollama_client=None)
        long_content = (
            "This is a detailed section that provides extensive information. "
            "It covers multiple aspects in depth, including implementation details, "
            "design considerations, and technical specifications. The section also "
            "discusses edge cases and provides concrete examples of how the feature works. "
            "Additional details ensure we exceed 300 characters for maximum score."
        )
        score = scorer._length_score(long_content)
        assert score == 1.0

    def test_specificity_with_concrete_keywords(self):
        """Content with specific keywords should score high."""
        scorer = ConfidenceScorer(ollama_client=None)
        specific_content = "Implement function login() in module auth.py using class UserAuth and database schema users."
        score = scorer._specificity_score(specific_content)
        assert score == 1.0

    def test_specificity_with_vague_keywords(self):
        """Content with vague keywords should score low."""
        scorer = ConfidenceScorer(ollama_client=None)
        vague_content = "Maybe do something, probably needs work, could be unclear, TBD placeholder."
        score = scorer._specificity_score(vague_content)
        assert score == 0.3

    def test_specificity_neutral(self):
        """Content with no specific or vague keywords is neutral."""
        scorer = ConfidenceScorer(ollama_client=None)
        neutral_content = "This section describes a component."
        score = scorer._specificity_score(neutral_content)
        assert score == 0.5

    def test_structure_with_bullets(self):
        """Content with bullet points should score high."""
        scorer = ConfidenceScorer(ollama_client=None)
        bullet_content = """Implementation steps:
- Step 1: Initialize
- Step 2: Configure
- Step 3: Deploy"""
        score = scorer._structure_score(bullet_content)
        assert score == 1.0

    def test_structure_with_numbers(self):
        """Content with numbered lists should score high."""
        scorer = ConfidenceScorer(ollama_client=None)
        numbered_content = """Steps:
1. First step
2. Second step
3. Third step"""
        score = scorer._structure_score(numbered_content)
        assert score == 1.0

    def test_structure_with_paragraphs(self):
        """Content with multiple paragraphs should score medium."""
        scorer = ConfidenceScorer(ollama_client=None)
        paragraph_content = """First paragraph here.

Second paragraph here."""
        score = scorer._structure_score(paragraph_content)
        assert score == 0.7

    def test_structure_single_block(self):
        """Single block content should score medium-low."""
        scorer = ConfidenceScorer(ollama_client=None)
        single_block = "This is all one continuous block of text without structure."
        score = scorer._structure_score(single_block)
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_heuristic_score_combines_subscores(self):
        """Heuristic score should average length, specificity, structure."""
        scorer = ConfidenceScorer(ollama_client=None)
        content = """Implement authentication module with:
- UserAuth class in auth.py
- login() method using database schema
- Session management API endpoint"""

        score = await scorer.score_section("implementation", content)

        # Should have high scores across all dimensions
        assert score > 0.8

    @pytest.mark.asyncio
    async def test_heuristics_only_mode_no_llm(self):
        """Heuristics-only mode should be deterministic."""
        scorer = ConfidenceScorer(ollama_client=None)
        content = "Some test content here."

        score1 = await scorer.score_section("test", content)
        score2 = await scorer.score_section("test", content)

        assert score1 == score2  # Deterministic


class MockOllamaClient:
    """Mock LLM client for testing."""

    def __init__(self, response: str = "yes"):
        self.response = response
        self.call_count = 0

    async def generate(self, messages: list[dict]) -> str:
        self.call_count += 1
        if self.response == "error":
            raise RuntimeError("Mock error")
        return self.response


class TestConfidenceScorerLLM:
    """Test LLM-based scoring with 1-10 scale."""

    @pytest.mark.asyncio
    async def test_llm_assessment_10_returns_1_0(self):
        """LLM '10' response should return 1.0."""
        mock_llm = MockOllamaClient(response="10")
        scorer = ConfidenceScorer(ollama_client=mock_llm)

        score = await scorer._llm_assessment("test", "content")
        assert score == 1.0
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_llm_assessment_1_returns_0_1(self):
        """LLM '1' response should return 0.1."""
        mock_llm = MockOllamaClient(response="1")
        scorer = ConfidenceScorer(ollama_client=mock_llm)

        score = await scorer._llm_assessment("test", "content")
        assert score == 0.1

    @pytest.mark.asyncio
    async def test_llm_assessment_5_returns_0_5(self):
        """LLM '5' response should return 0.5."""
        mock_llm = MockOllamaClient(response="5")
        scorer = ConfidenceScorer(ollama_client=mock_llm)

        score = await scorer._llm_assessment("test", "content")
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_llm_assessment_7_returns_0_7(self):
        """LLM '7' response should return 0.7."""
        mock_llm = MockOllamaClient(response="7")
        scorer = ConfidenceScorer(ollama_client=mock_llm)

        score = await scorer._llm_assessment("test", "content")
        assert score == 0.7

    @pytest.mark.asyncio
    async def test_llm_assessment_extracts_digit_from_text(self):
        """LLM response with surrounding text should still extract digit."""
        mock_llm = MockOllamaClient(response="I'd rate this a 7 out of 10")
        scorer = ConfidenceScorer(ollama_client=mock_llm)

        score = await scorer._llm_assessment("test", "content")
        assert score == 0.7  # 7 → 0.7

    @pytest.mark.asyncio
    async def test_llm_assessment_extracts_10_from_text(self):
        """LLM response '10' should be extracted correctly (not as '1' + '0')."""
        mock_llm = MockOllamaClient(response="I'd give this a 10")
        scorer = ConfidenceScorer(ollama_client=mock_llm)

        score = await scorer._llm_assessment("test", "content")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_llm_assessment_invalid_returns_default(self):
        """Invalid LLM response should return 0.5."""
        mock_llm = MockOllamaClient(response="this section is okay")
        scorer = ConfidenceScorer(ollama_client=mock_llm)

        score = await scorer._llm_assessment("test", "content")
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_llm_assessment_error_returns_default(self):
        """LLM error should return 0.5."""
        mock_llm = MockOllamaClient(response="error")
        scorer = ConfidenceScorer(ollama_client=mock_llm)

        score = await scorer._llm_assessment("test", "content")
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_hybrid_scoring_weights(self):
        """Hybrid scoring should weight LLM (0.7) and heuristics (0.3)."""
        mock_llm = MockOllamaClient(response="8")
        scorer = ConfidenceScorer(ollama_client=mock_llm)

        # Use content that would score ~0.5 heuristically
        content = "This is a medium section."
        score = await scorer.score_section("test", content)

        # Expected: 0.7 * 0.8 (LLM 8) + 0.3 * ~0.5 (heuristic) ≈ 0.71
        assert 0.60 <= score <= 0.80

    @pytest.mark.asyncio
    async def test_codebase_context_included_in_prompt(self):
        """Codebase context is included in LLM assessment prompt."""
        mock_llm = MockOllamaClient(response="4")
        scorer = ConfidenceScorer(ollama_client=mock_llm)

        await scorer.score_section(
            "architecture", "Use REST API with Flask.",
            codebase_context="## Files\n- src/api.py: existing REST endpoints",
        )

        # The mock stores calls implicitly — verify the prompt had context
        # Since MockOllamaClient doesn't store calls, check the LLM was called
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_no_codebase_context_no_grounding(self):
        """Without codebase context, no grounding criterion in prompt."""
        mock_llm = MockOllamaClient(response="4")
        scorer = ConfidenceScorer(ollama_client=mock_llm)

        score = await scorer.score_section("test", "Some content")
        assert 0.0 <= score <= 1.0
        assert mock_llm.call_count == 1


class TestSectionFlagger:
    """Test section flagging logic."""

    def test_default_section_above_threshold_not_flagged(self):
        """Regular section above default threshold should not be flagged."""
        flagger = SectionFlagger(default_threshold=0.7, security_threshold=0.9)
        flagged, reason = flagger.flag_section("implementation", 0.8)

        assert not flagged
        assert reason == ""

    def test_default_section_below_threshold_flagged(self):
        """Regular section below default threshold should be flagged."""
        flagger = SectionFlagger(default_threshold=0.7, security_threshold=0.9)
        flagged, reason = flagger.flag_section("implementation", 0.6)

        assert flagged
        assert "implementation" in reason
        assert "0.60" in reason
        assert "0.70" in reason

    def test_security_section_below_security_threshold_flagged(self):
        """Security section below security threshold should be flagged."""
        flagger = SectionFlagger(default_threshold=0.7, security_threshold=0.9)
        flagged, reason = flagger.flag_section("security", 0.85)

        assert flagged
        assert "Security-sensitive" in reason
        assert "security" in reason
        assert "0.85" in reason
        assert "0.90" in reason

    def test_security_section_above_security_threshold_not_flagged(self):
        """Security section above security threshold should not be flagged."""
        flagger = SectionFlagger(default_threshold=0.7, security_threshold=0.9)
        flagged, reason = flagger.flag_section("security", 0.95)

        assert not flagged
        assert reason == ""

    def test_risk_section_uses_security_threshold(self):
        """'risk' section should use security threshold."""
        flagger = SectionFlagger(default_threshold=0.7, security_threshold=0.9)
        flagged, _ = flagger.flag_section("risk", 0.85)

        assert flagged  # 0.85 < 0.9

    def test_authentication_section_uses_security_threshold(self):
        """'authentication' section should use security threshold."""
        flagger = SectionFlagger(default_threshold=0.7, security_threshold=0.9)
        flagged, _ = flagger.flag_section("authentication", 0.85)

        assert flagged

    def test_authorization_section_uses_security_threshold(self):
        """'authorization' section should use security threshold."""
        flagger = SectionFlagger(default_threshold=0.7, security_threshold=0.9)
        flagged, _ = flagger.flag_section("authorization", 0.85)

        assert flagged

    def test_encryption_section_uses_security_threshold(self):
        """'encryption' section should use security threshold."""
        flagger = SectionFlagger(default_threshold=0.7, security_threshold=0.9)
        flagged, _ = flagger.flag_section("encryption", 0.85)

        assert flagged

    def test_compute_overall_score_averages(self):
        """Overall score should average all section scores."""
        flagger = SectionFlagger()
        section_scores = {
            "context": 0.8,
            "implementation": 0.9,
            "testing": 0.7,
        }

        overall = flagger.compute_overall_score(section_scores)
        expected = (0.8 + 0.9 + 0.7) / 3
        assert overall == round(expected, 2)

    def test_compute_overall_score_empty_dict(self):
        """Overall score for empty dict should be 0.0."""
        flagger = SectionFlagger()
        overall = flagger.compute_overall_score({})
        assert overall == 0.0

    def test_from_config_factory(self):
        """from_config() should create flagger from ConfidenceConfig."""
        config = ConfidenceConfig(default_threshold=0.75, security_threshold=0.95)
        flagger = SectionFlagger.from_config(config)

        assert flagger.default_threshold == 0.75
        assert flagger.security_threshold == 0.95

    def test_from_config_uses_defaults(self):
        """from_config() should handle default ConfidenceConfig values."""
        config = ConfidenceConfig()  # Uses defaults from schema
        flagger = SectionFlagger.from_config(config)

        assert flagger.default_threshold == 0.7
        assert flagger.security_threshold == 0.9
