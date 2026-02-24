# fitz_graveyard/llm/types.py
"""Normalized LLM response types shared by all client implementations."""

from dataclasses import dataclass, field


@dataclass
class AgentToolCall:
    """A normalized tool call from any LLM provider."""

    id: str  # "" for Ollama (no ids), UUID/call_id for OpenAI
    name: str
    arguments: dict  # always a dict (LMStudio client parses JSON string)


@dataclass
class AgentMessage:
    """Normalized agent response from any LLM provider."""

    content: str | None
    tool_calls: list[AgentToolCall] | None
    assistant_dict: dict  # provider-specific dict ready to append to messages list
