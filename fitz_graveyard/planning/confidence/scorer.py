# fitz_graveyard/planning/confidence/scorer.py
"""
Confidence scorer using hybrid LLM + heuristic approach.

Scores plan section quality on 0.0-1.0 scale to identify sections needing review.
Uses section-specific criteria for more accurate scoring.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Section-specific scoring criteria (appended to LLM prompt)
_SECTION_CRITERIA: dict[str, str] = {
    "context": (
        "- Does it identify specific, testable requirements (not vague goals)?\n"
        "- Does it list real existing files from the codebase with accurate descriptions?\n"
        "- Are needed artifacts genuinely NEW files, or does it re-list existing ones?\n"
        "- If the codebase already implements this task, does the plan acknowledge that?\n"
        "- Are assumptions explicit with impact assessment?"
    ),
    "architecture_design": (
        "- Does it consider whether the task is already implemented before proposing new code?\n"
        "- Are the explored approaches genuinely different, not just variations?\n"
        "- Does the recommended approach match the actual codebase patterns?\n"
        "- Do ADRs record real decisions (not obvious choices)?\n"
        "- Do artifacts reference real interfaces and field names from the codebase?"
    ),
    "architecture": (
        "- Does it consider whether the task is already implemented before proposing new code?\n"
        "- Are the explored approaches genuinely different, not just variations?\n"
        "- Does the scope statement honestly characterize the engineering effort?\n"
        "- Does the recommended approach match the actual codebase patterns?"
    ),
    "design": (
        "- Do ADRs record real decisions with concrete tradeoffs (not obvious choices)?\n"
        "- Do component interfaces use real function signatures from the codebase?\n"
        "- Do artifacts reference real field names and patterns from existing code?\n"
        "- Is the data model grounded in existing schemas?"
    ),
    "roadmap": (
        "- Do phases have concrete deliverables (not just descriptions)?\n"
        "- Are effort estimates realistic for the actual scope?\n"
        "- Do verification commands actually test what the phase delivers?\n"
        "- If the task is already implemented, do phases focus on verification not building?"
    ),
    "roadmap_risk": (
        "- Do roadmap phases have concrete deliverables with verification commands?\n"
        "- Are effort estimates realistic for the actual scope?\n"
        "- Do risks cite specific technical causes (not generic categories)?\n"
        "- Do risk mitigations reference real phases?"
    ),
    "risk": (
        "- Do risks cite specific technical causes (not generic 'might fail')?\n"
        "- Are impact/likelihood ratings justified, not all 'medium/medium'?\n"
        "- Do mitigations name concrete actions (not 'add error handling')?\n"
        "- Do affected phases actually exist in the roadmap?"
    ),
}


class ConfidenceScorer:
    """
    Scores plan section quality using hybrid LLM self-assessment + heuristics.

    Scoring formula:
    - Hybrid mode (has LLM): 0.7 * LLM_score + 0.3 * heuristic_score
    - Heuristics-only mode (no LLM): heuristic_score

    LLM uses a 1-10 scale for finer granularity, with section-specific criteria.
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
            section_name: Name of the section (e.g., "context", "architecture_design")
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
        Ask LLM to rate section quality on a 1-10 scale.

        Uses section-specific criteria for more accurate scoring.
        When codebase_context is provided, the LLM also checks whether the section
        references real files/APIs from the codebase vs hallucinated ones.

        Returns:
            Score mapped to 0.0-1.0 (1→0.1, 5→0.5, 10→1.0). Default 0.5 on error.
        """
        context_block = ""
        grounding_criterion = ""
        if codebase_context:
            context_block = f"\nCodebase context (ground truth):\n{codebase_context}\n"
            grounding_criterion = (
                "\nGROUNDING CHECK: Does the section reference real files, APIs, and "
                "patterns from the codebase context? Hallucinated references or proposing "
                "to build something that already exists should lower the score significantly."
            )

        # Get section-specific criteria
        criteria = _SECTION_CRITERIA.get(section_name, "")
        criteria_block = ""
        if criteria:
            criteria_block = f"\nSection-specific criteria:\n{criteria}\n"

        prompt = (
            f'Rate the quality of this "{section_name}" section on a 1-10 scale:\n'
            "1-2 = Missing, incoherent, or fundamentally wrong\n"
            "3-4 = Vague, generic, or ignores existing codebase\n"
            "5-6 = Adequate but could be more concrete or grounded\n"
            "7-8 = Good — specific, actionable, and grounded in codebase\n"
            "9-10 = Excellent — production-ready, correctly leverages existing code\n"
            f"{criteria_block}"
            f"{grounding_criterion}"
            f"{context_block}"
            f"\nSection content:\n{content}\n\n"
            "Reply with ONLY a number from 1 to 10."
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.ollama_client.generate(messages)
            match = re.search(r'\b(10|[1-9])\b', response.strip())
            if match:
                digit = int(match.group())
                return digit / 10.0
            logger.warning(f"LLM response not 1-10: {response[:50]}. Defaulting to 0.5")
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
