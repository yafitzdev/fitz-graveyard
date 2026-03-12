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
    max_file_bytes: int = Field(
        default=50_000, description="Maximum bytes to read per file"
    )
    source_dir: str | None = Field(
        default=None,
        description=(
            "Source directory for codebase context gathering. "
            "Resolution order: create_plan(source_dir=) parameter > this config value > cwd at runtime."
        ),
    )
    max_seed_files: int = Field(
        default=30,
        description=(
            "Maximum files included as full source in the planning prompt (seed set). "
            "Remaining files are available via read_file tool during reasoning. "
            "Lower values force the LLM to actively explore via tool calls."
        ),
    )


class OutputConfig(BaseModel):
    """Output and file path configuration."""

    model_config = ConfigDict(extra="ignore")

    plans_dir: str = Field(
        default=".fitz-graveyard/plans",
        description="Directory for plans. Relative paths are resolved against the project directory.",
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
    fast_model: str | None = Field(
        default=None, description="Model for fast/screening tasks (None = use model)"
    )
    smart_model: str | None = Field(
        default=None, description="Model for reasoning tasks (None = use model)"
    )
    fallback_model: str | None = Field(
        default=None, description="Fallback model (None = no fallback)"
    )
    timeout: int = Field(
        default=300, description="Request timeout in seconds"
    )
    context_length: int = Field(
        default=65536,
        description="Context length to use when auto-loading the model via lms CLI",
    )


class LlamaCppModelConfig(BaseModel):
    """Settings for a single GGUF model used by llama-server."""

    model_config = ConfigDict(extra="ignore")

    path: str = Field(default="", description="GGUF filename (relative to models_dir)")
    context_size: int = Field(default=8192, description="Context window size")
    gpu_layers: int = Field(default=-1, description="GPU layers to offload (-1 = all)")
    flash_attention: bool = Field(default=False, description="Enable flash attention")
    cache_type_k: str | None = Field(default=None, description="KV cache type for keys (e.g. q4_0)")
    cache_type_v: str | None = Field(default=None, description="KV cache type for values (e.g. q4_0)")


class LlamaCppConfig(BaseModel):
    """llama.cpp server configuration with fast/smart model tiers."""

    model_config = ConfigDict(extra="ignore")

    server_path: str = Field(
        default="", description="Path to llama-server binary"
    )
    models_dir: str = Field(
        default="", description="Directory containing GGUF model files"
    )
    fast_model: LlamaCppModelConfig = Field(
        default_factory=LlamaCppModelConfig,
        description="Small model for screening (fast YES/NO calls)",
    )
    mid_model: LlamaCppModelConfig | None = Field(
        default=None,
        description="Mid-tier model for summarization (None = use fast_model)",
    )
    smart_model: LlamaCppModelConfig | None = Field(
        default=None,
        description="Large model for reasoning (None = use fast_model for everything)",
    )
    port: int = Field(default=8012, description="llama-server port")
    timeout: int = Field(default=300, description="Request timeout in seconds")
    startup_timeout: int = Field(
        default=120, description="Max seconds to wait for server startup"
    )


class GPUConfig(BaseModel):
    """GPU thermal protection configuration."""

    model_config = ConfigDict(extra="ignore")

    temp_threshold: int = Field(
        default=73,
        ge=0,
        le=95,
        description=(
            "Pause/throttle LLM calls when GPU temperature exceeds this (°C). "
            "0 to disable GPU temperature monitoring."
        ),
    )
    cooldown_margin: int = Field(
        default=10,
        ge=5,
        le=30,
        description="Resume after pre-flight pause when temp drops this many °C below threshold",
    )


class FitzPlannerConfig(BaseModel):
    """Root configuration for fitz-graveyard."""

    model_config = ConfigDict(extra="ignore")

    provider: Literal["ollama", "lm_studio", "llama_cpp"] = Field(
        default="ollama", description="LLM provider to use"
    )
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    lm_studio: LMStudioConfig = Field(default_factory=LMStudioConfig)
    llama_cpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
