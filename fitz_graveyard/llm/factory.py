# fitz_graveyard/llm/factory.py
"""Factory for creating the configured LLM client."""

from fitz_graveyard.config.schema import FitzPlannerConfig

from .client import OllamaClient
from .lm_studio import LMStudioClient


def create_llm_client(config: FitzPlannerConfig) -> OllamaClient | LMStudioClient:
    """
    Create the appropriate LLM client based on config.provider.

    Args:
        config: Root FitzPlannerConfig

    Returns:
        OllamaClient for provider="ollama", LMStudioClient for provider="lm_studio"
    """
    if config.provider == "lm_studio":
        return LMStudioClient(
            base_url=config.lm_studio.base_url,
            model=config.lm_studio.model,
            fallback_model=config.lm_studio.fallback_model,
            timeout=config.lm_studio.timeout,
        )
    return OllamaClient(
        base_url=config.ollama.base_url,
        model=config.ollama.model,
        fallback_model=config.ollama.fallback_model,
        timeout=config.ollama.timeout,
    )
