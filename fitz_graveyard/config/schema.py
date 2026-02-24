# fitz_graveyard/config/schema.py
"""
Pydantic configuration models for fitz-graveyard.

All models use extra="ignore" to allow unknown YAML keys without crashing.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OllamaConfig(BaseModel):
    """Ollama server configuration."""

    model_config = ConfigDict(extra="ignore")

    base_url: str = Field(
        default="http://localhost:11434", description="Ollama API base URL"
    )
    model: str = Field(
        default="qwen2.5-coder-next:80b-instruct",
        description="Ollama model to use for planning",
    )
    fallback_model: str | None = Field(
        default="qwen2.5-coder-next:32b-instruct",
        description="Fallback model on OOM errors (None to disable)",
    )
    timeout: int = Field(
        default=300, description="Request timeout in seconds (generous for model loading)"
    )
    memory_threshold: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="RAM usage % threshold to abort generation",
    )


class AgentConfig(BaseModel):
    """Local LLM agent configuration for codebase context gathering."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(
        default=True, description="Enable local agent for codebase context gathering"
    )
    agent_model: str | None = Field(
        default=None, description="Model for agent tool calls (None = use ollama.model)"
    )
    max_iterations: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Maximum tool-call iterations before stopping",
    )
    max_file_bytes: int = Field(
        default=50_000, description="Maximum bytes to read per file"
    )
    source_dir: str | None = Field(
        default=None, description="Default source directory (overridden by create_plan parameter)"
    )


class OutputConfig(BaseModel):
    """Output and file path configuration."""

    model_config = ConfigDict(extra="ignore")

    plans_dir: str = Field(
        default=".fitz-graveyard/plans",
        description="Directory for generated plans (relative to project root)",
    )
    verbosity: Literal["quiet", "normal", "verbose"] = Field(
        default="normal", description="Logging verbosity level"
    )


class ConfidenceConfig(BaseModel):
    """Confidence threshold configuration."""

    model_config = ConfigDict(extra="ignore")

    default_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Default confidence threshold for accepting LLM outputs",
    )
    security_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Higher threshold for security-sensitive operations",
    )


class AnthropicConfig(BaseModel):
    """Anthropic API configuration for optional API review."""

    model_config = ConfigDict(extra="ignore")

    api_key: str | None = Field(
        default=None,
        description="Anthropic API key (None = API review unavailable)",
    )
    model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Model to use for API review",
    )
    max_review_tokens: int = Field(
        default=2048,
        ge=1,
        le=8192,
        description="Maximum tokens for review responses",
    )


class LMStudioConfig(BaseModel):
    """LM Studio server configuration."""

    model_config = ConfigDict(extra="ignore")

    base_url: str = Field(
        default="http://localhost:1234/v1", description="LM Studio API base URL"
    )
    model: str = Field(default="local-model", description="LM Studio model to use for planning")
    fallback_model: str | None = Field(
        default=None, description="Fallback model (None = no fallback)"
    )
    timeout: int = Field(
        default=300, description="Request timeout in seconds"
    )


class FitzPlannerConfig(BaseModel):
    """Root configuration for fitz-graveyard."""

    model_config = ConfigDict(extra="ignore")

    provider: Literal["ollama", "lm_studio"] = Field(
        default="ollama", description="LLM provider to use"
    )
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    lm_studio: LMStudioConfig = Field(default_factory=LMStudioConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
