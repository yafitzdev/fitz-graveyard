# fitz_graveyard/planning/confidence/flagging.py
"""
Section flagger for low-confidence plan sections.

Flags sections below quality thresholds for human review.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fitz_graveyard.config.schema import ConfidenceConfig

# Security-sensitive sections require higher confidence threshold
SECURITY_SECTIONS = {
    "risk",
    "security",
    "authentication",
    "authorization",
    "encryption",
}


class SectionFlagger:
    """
    Flags plan sections below confidence thresholds.

    Uses different thresholds for regular vs security-sensitive sections.
    """

    def __init__(self, default_threshold: float = 0.7, security_threshold: float = 0.9):
        """
        Initialize section flagger.

        Args:
            default_threshold: Threshold for regular sections (default: 0.7)
            security_threshold: Higher threshold for security sections (default: 0.9)
        """
        self.default_threshold = default_threshold
        self.security_threshold = security_threshold

    def flag_section(self, section_name: str, score: float) -> tuple[bool, str]:
        """
        Determine if section should be flagged for review.

        Args:
            section_name: Name of the section
            score: Confidence score (0.0-1.0)

        Returns:
            (is_flagged, reason): Whether flagged and why
        """
        section_lower = section_name.lower()
        is_security = any(sec_keyword in section_lower for sec_keyword in SECURITY_SECTIONS)

        threshold = self.security_threshold if is_security else self.default_threshold

        if score < threshold:
            reason = self._get_flag_reason(section_name, score, threshold, is_security)
            return True, reason
        else:
            return False, ""

    def _get_flag_reason(
        self, section_name: str, score: float, threshold: float, is_security: bool
    ) -> str:
        """Generate human-readable flag reason."""
        section_type = "Security-sensitive" if is_security else "Standard"
        return (
            f"{section_type} section '{section_name}' scored {score:.2f}, "
            f"below threshold {threshold:.2f}"
        )

    def compute_overall_score(self, section_scores: dict[str, float]) -> float:
        """
        Compute overall plan quality score.

        Args:
            section_scores: Mapping of section_name -> confidence_score

        Returns:
            Average confidence score across all sections (0.0-1.0)
        """
        if not section_scores:
            return 0.0

        total = sum(section_scores.values())
        count = len(section_scores)
        avg = total / count
        return round(avg, 2)

    @classmethod
    def from_config(cls, config: "ConfidenceConfig") -> "SectionFlagger":
        """
        Create SectionFlagger from ConfidenceConfig.

        Args:
            config: ConfidenceConfig with threshold values

        Returns:
            Configured SectionFlagger instance
        """
        return cls(
            default_threshold=config.default_threshold,
            security_threshold=config.security_threshold,
        )
