# fitz_graveyard/planning/clarification.py
"""
Clarification question generation for planning jobs.

Asks the LLM to identify what's underspecified before planning begins.
"""

import logging
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import SYSTEM_PROMPT, extract_json
from fitz_graveyard.planning.prompts import load_prompt

logger = logging.getLogger(__name__)


async def get_clarifying_questions(client: Any, description: str) -> list[str]:
    """
    Generate clarifying questions for a planning description.

    Calls the LLM to identify 2-3 questions that would most sharpen the plan.
    Returns empty list if description is already specific or if LLM call fails.

    Args:
        client: OllamaClient or LMStudioClient instance
        description: Raw planning description from user

    Returns:
        List of 2-3 question strings, or empty list
    """
    try:
        prompt_template = load_prompt("clarification")
        prompt = prompt_template.format(description=description)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        raw = await client.generate(messages=messages)
        questions = extract_json(raw)

        if not isinstance(questions, list):
            return []

        return [q for q in questions if isinstance(q, str) and q.strip()][:3]

    except Exception as e:
        logger.warning(f"Clarification question generation failed (skipping): {e}")
        return []
