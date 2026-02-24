# fitz_graveyard/planning/agent/__init__.py
"""Local LLM context-gathering agent using Ollama tool calls."""

from fitz_graveyard.planning.agent.gatherer import AgentContextGatherer

__all__ = ["AgentContextGatherer"]
