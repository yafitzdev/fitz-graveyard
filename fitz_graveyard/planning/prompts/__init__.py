# fitz_graveyard/planning/prompts/__init__.py
"""Prompt loading utilities for planning stages."""

from pathlib import Path


def load_prompt(name: str) -> str:
    """Load a prompt template by name.

    Args:
        name: Prompt filename without .txt extension
              (e.g., 'context', 'architecture', 'architecture_format')

    Returns:
        Prompt content as string

    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    prompt_path = Path(__file__).parent / f"{name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8")


__all__ = ["load_prompt"]
