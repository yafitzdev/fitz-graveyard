# fitz_planner_mcp/llm/__init__.py
"""LLM integration module with Ollama client, memory monitoring, and retry logic."""

from .memory import MemoryMonitor
from .retry import ollama_retry

__all__ = ["MemoryMonitor", "ollama_retry"]

# OllamaClient will be added in Task 2
try:
    from .client import OllamaClient

    __all__.append("OllamaClient")
except ImportError:
    pass
