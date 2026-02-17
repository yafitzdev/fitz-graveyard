# fitz_planner_mcp/config/schema.py
"""
Pydantic configuration models for fitz-planner-mcp.

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


class KragConfig(BaseModel):
    """KRAG (Knowledge-Retrieval Augmented Generation) configuration."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(
        default=True, description="Enable KRAG for enhanced planning quality"
    )
    fitz_ai_config: str | None = Field(
        default=None, description="Path to fitz-ai config file (uses defaults if None)"
    )


class OutputConfig(BaseModel):
    """Output and file path configuration."""

    model_config = ConfigDict(extra="ignore")

    plans_dir: str = Field(
        default=".fitz-planner/plans",
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


class FitzPlannerConfig(BaseModel):
    """Root configuration for fitz-planner-mcp."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(
        default="qwen2.5-coder-next:80b-instruct",
        description="Ollama model to use for planning",
    )
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    krag: KragConfig = Field(default_factory=KragConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
