# fitz_graveyard/planning/krag/__init__.py
"""
KRAG (Knowledge-Retrieval Augmented Generation) client module.

Wraps fitz-ai SDK with graceful fallback, context formatting, and multi-query support.
"""

from .client import KragClient
from .formatter import format_krag_answer, format_krag_results

__all__ = ["KragClient", "format_krag_answer", "format_krag_results"]
