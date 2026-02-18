# fitz_planner_mcp/planning/krag/formatter.py
"""
Context formatting functions for fitz-ai Answer objects.

Converts Answer objects into structured markdown for LLM consumption, including
file path citations and provenance information.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fitz_ai.core import Answer


def format_krag_answer(answer: "Answer") -> str:
    """
    Format a fitz Answer object into structured markdown.

    Args:
        answer: fitz Answer object with .text, .provenance, .mode attributes

    Returns:
        Formatted markdown string with answer text and provenance citations.

    Format:
        {answer.text}

        **Sources:**
        - **path/to/file.py** (lines 10-25) [kind: code]
        - **path/to/doc.md**

    Max 5 provenance entries to keep context concise.
    """
    if not answer or not hasattr(answer, "text"):
        return ""

    text = answer.text or ""
    if not text.strip():
        return ""

    # Start with answer text
    result = text.strip()

    # Add provenance if available
    if hasattr(answer, "provenance") and answer.provenance:
        provenance_lines = []

        # Cap at 5 entries to avoid overwhelming the LLM
        for prov in answer.provenance[:5]:
            if not hasattr(prov, "file_path"):
                continue

            file_path = prov.file_path

            # Build citation line
            citation = f"- **{file_path}**"

            # Add line range if available
            if hasattr(prov, "line_range") and prov.line_range:
                start, end = prov.line_range
                citation += f" (lines {start}-{end})"

            # Add kind tag if available
            if hasattr(prov, "metadata") and isinstance(prov.metadata, dict):
                kind = prov.metadata.get("kind")
                if kind:
                    citation += f" [{kind}]"

            provenance_lines.append(citation)

        if provenance_lines:
            result += "\n\n**Sources:**\n" + "\n".join(provenance_lines)

    return result


def format_krag_results(results: list[tuple[str, str]]) -> str:
    """
    Format multiple (query, answer) tuples into aggregated markdown.

    Args:
        results: List of (query, formatted_answer) tuples

    Returns:
        Markdown with section headers for each query/answer pair.

    Format:
        ## Codebase Context

        ### Query 1
        answer 1

        ### Query 2
        answer 2

    Returns empty string if no results provided.
    """
    if not results:
        return ""

    sections = [f"### {query}\n\n{answer}\n" for query, answer in results]
    return "## Codebase Context\n\n" + "\n".join(sections)
