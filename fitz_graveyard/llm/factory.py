# fitz_graveyard/llm/factory.py
"""Factory for creating the configured LLM client."""

from fitz_graveyard.config.schema import FitzPlannerConfig

from .client import OllamaClient
from .gpu_monitor import GPUTemperatureGuard
from .llama_cpp import LlamaCppClient
from .lm_studio import LMStudioClient


def _create_gpu_guard(config: FitzPlannerConfig) -> GPUTemperatureGuard | None:
    """Create a GPU temperature guard from config, or None if disabled."""
    if config.gpu.temp_threshold <= 0:
        return None
    return GPUTemperatureGuard(
        threshold=config.gpu.temp_threshold,
        cooldown_margin=config.gpu.cooldown_margin,
    )


def create_llm_client(
    config: FitzPlannerConfig,
) -> OllamaClient | LMStudioClient | LlamaCppClient:
    """
    Create the appropriate LLM client based on config.provider.

    Args:
        config: Root FitzPlannerConfig

    Returns:
        Client instance for the configured provider.

    Raises:
        ValueError: If llama_cpp config is missing required fields.
    """
    gpu_guard = _create_gpu_guard(config)

    if config.provider == "lm_studio":
        return LMStudioClient(
            base_url=config.lm_studio.base_url,
            model=config.lm_studio.model,
            fallback_model=config.lm_studio.fallback_model,
            timeout=config.lm_studio.timeout,
            context_length=config.lm_studio.context_length,
            gpu_guard=gpu_guard,
        )
    if config.provider == "llama_cpp":
        cfg = config.llama_cpp
        if not cfg.server_path:
            raise ValueError(
                "llama_cpp.server_path is required when provider=llama_cpp"
            )
        if not cfg.fast_model.path:
            raise ValueError(
                "llama_cpp.fast_model.path is required when provider=llama_cpp"
            )
        if not cfg.models_dir:
            raise ValueError(
                "llama_cpp.models_dir is required when provider=llama_cpp"
            )
        return LlamaCppClient(
            server_path=cfg.server_path,
            models_dir=cfg.models_dir,
            fast_model=cfg.fast_model,
            mid_model=cfg.mid_model,
            smart_model=cfg.smart_model,
            port=cfg.port,
            timeout=cfg.timeout,
            startup_timeout=cfg.startup_timeout,
            gpu_guard=gpu_guard,
        )
    return OllamaClient(
        base_url=config.ollama.base_url,
        model=config.ollama.model,
        fallback_model=config.ollama.fallback_model,
        timeout=config.ollama.timeout,
    )
