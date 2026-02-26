# fitz_graveyard/planning/confidence/scorer.py
"""
Confidence scorer using hybrid LLM + heuristic approach.

Scores plan section quality on 0.0-1.0 scale to identify sections needing review.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Scores plan section quality using hybrid LLM self-assessment + heuristics.

    Scoring formula:
    - Hybrid mode (has LLM): 0.7 * LLM_score + 0.3 * heuristic_score
    - Heuristics-only mode (no LLM): heuristic_score

    Heuristic scoring checks:
    - Length (sufficient detail)
    - Specificity keywords (concrete terms)
    - Vague keywords (ambiguous terms)
    - Structure (bullet points, numbers)
    """

    # Keywords that indicate specificity/concreteness
    SPECIFICITY_KEYWORDS = {
        "implementation",
        "function",
        "class",
        "method",
        "module",
        "file",
        "database",
        "api",
        "endpoint",
        "schema",
        "test",
        "step",
        "algorithm",
        "protocol",
        "interface",
    }

    # Keywords that indicate vagueness/ambiguity
    VAGUE_KEYWORDS = {
        "maybe",
        "possibly",
        "probably",
        "should",
        "could",
        "might",
        "unclear",
        "unknown",
        "tbd",
        "todo",
        "placeholder",
        "example",
        "etc",
    }

    def __init__(self, ollama_client: Any | None = None):
        """
        Initialize confidence scorer.

        Args:
            ollama_client: Optional OllamaClient for LLM self-assessment.
                          If None, uses heuristics-only mode.
        """
        self.ollama_client = ollama_client

    async def score_section(
        self, section_name: str, content: str, codebase_context: str = "",
    ) -> float:
        """
        Score a plan section's quality.

        Args:
            section_name: Name of the section (e.g., "implementation", "security")
            content: Section content text
            codebase_context: Optional codebase context for grounding assessment

        Returns:
            Confidence score (0.0-1.0), higher is better quality
        """
        heuristic_score = self._heuristic_score(content)

        if self.ollama_client is None:
            # Heuristics-only mode
            return heuristic_score

        # Hybrid mode: combine LLM and heuristics
        llm_score = await self._llm_assessment(section_name, content, codebase_context)
        hybrid_score = 0.7 * llm_score + 0.3 * heuristic_score
        return round(hybrid_score, 2)

    async def _llm_assessment(
        self, section_name: str, content: str, codebase_context: str = "",
    ) -> float:
        """
        Ask LLM to rate section quality on a 1-5 scale.

        When codebase_context is provided, the LLM also checks whether the section
        references real files/APIs from the codebase vs hallucinated ones.

        Returns:
            Mapped score: 1→0.2, 2→0.4, 3→0.6, 4→0.8, 5→1.0. Default 0.5 on error.
        """
        context_block = ""
        grounding_criterion = ""
        if codebase_context:
            context_block = f"\nCodebase context (ground truth):\n{codebase_context}\n"
            grounding_criterion = (
                "\nAlso check: does the section reference real files, APIs, and patterns "
                "from the codebase context? Hallucinated references should lower the score."
            )

        prompt = f"""Rate the quality of this "{section_name}" section on a 1-5 scale:
1 = Missing or incoherent
2 = Vague, generic, lacks specifics
3 = Adequate but could be more concrete
4 = Good — specific and actionable
5 = Excellent — concrete, thorough, production-ready
{grounding_criterion}
{context_block}
Section content:
{content}

Reply with ONLY a single digit (1-5)."""

        SCALE = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.ollama_client.generate(messages)
            match = re.search(r'[1-5]', response.strip())
            if match:
                digit = int(match.group())
                return SCALE[digit]
            logger.warning(f"LLM response not 1-5: {response[:50]}. Defaulting to 0.5")
            return 0.5
        except Exception as e:
            logger.warning(f"LLM assessment failed: {e}. Defaulting to 0.5")
            return 0.5

    def _heuristic_score(self, content: str) -> float:
        """
        Compute heuristic quality score based on content analysis.

        Returns:
            Average of length_score, specificity_score, structure_score
        """
        length_score = self._length_score(content)
        specificity_score = self._specificity_score(content)
        structure_score = self._structure_score(content)

        avg_score = (length_score + specificity_score + structure_score) / 3.0
        return round(avg_score, 2)

    def _length_score(self, content: str) -> float:
        """Score based on content length (sufficient detail)."""
        char_count = len(content.strip())

        if char_count < 50:
            return 0.2  # Too short
        elif char_count < 150:
            return 0.5  # Minimal
        elif char_count < 300:
            return 0.8  # Good
        else:
            return 1.0  # Detailed

    def _specificity_score(self, content: str) -> float:
        """Score based on specificity keywords vs vague keywords."""
        content_lower = content.lower()

        specificity_count = sum(
            1 for keyword in self.SPECIFICITY_KEYWORDS if keyword in content_lower
        )
        vague_count = sum(
            1 for keyword in self.VAGUE_KEYWORDS if keyword in content_lower
        )

        # Penalize vague keywords, reward specific keywords
        if specificity_count == 0 and vague_count > 0:
            return 0.3  # Vague, no specifics
        elif specificity_count == 0:
            return 0.5  # Neutral
        elif vague_count > specificity_count:
            return 0.6  # More vague than specific
        else:
            return 1.0  # Specific and concrete

    def _structure_score(self, content: str) -> float:
        """Score based on structure (bullet points, numbers, etc.)."""
        has_bullets = bool(re.search(r"^\s*[-*•]", content, re.MULTILINE))
        has_numbers = bool(re.search(r"^\s*\d+\.", content, re.MULTILINE))
        has_newlines = "\n" in content

        if has_bullets or has_numbers:
            return 1.0  # Structured list
        elif has_newlines:
            return 0.7  # Multiple paragraphs
        else:
            return 0.5  # Single block
