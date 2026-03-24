# fitz_graveyard/planning/pipeline/decomposition_validator.py
"""
Post-decomposition validation: detect gaps in the decision list and
inject synthetic decisions to fill them.

Runs after DecisionDecompositionStage, before DecisionResolutionStage.
Pure Python — no LLM calls.

Two checks:
1. Return type bridge: if the call graph shows a type mismatch between
   entry point and leaf (e.g., answer() -> Answer but chat_stream() ->
   Iterator[str]), ensure a decision addresses bridging them.
2. Non-happy-path: if the call graph contains gating/validation methods
   (decide, validate, check, gate, reject, abort), ensure a decision
   addresses what happens on rejection.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Method names that indicate gating/validation behavior
_GATE_PATTERNS = re.compile(
    r"\b(decide|validate|check|gate|reject|abort|abstain|guard|filter|block)\b",
    re.IGNORECASE,
)


def _extract_return_types(class_detail: str) -> list[tuple[str, str]]:
    """Extract (method_name, return_type) pairs from class_detail string.

    class_detail looks like: 'FitzKragEngine [answer -> Answer, generate -> str]'
    Returns: [('answer', 'Answer'), ('generate', 'str')]
    """
    bracket_match = re.search(r"\[(.+)\]", class_detail)
    if not bracket_match:
        return []
    inner = bracket_match.group(1)
    pairs = []
    for segment in inner.split(","):
        segment = segment.strip()
        arrow = segment.split("->")
        if len(arrow) == 2:
            method = arrow[0].strip()
            ret_type = arrow[1].strip()
            pairs.append((method, ret_type))
    return pairs


def _decisions_mention(decisions: list[dict], *terms: str) -> bool:
    """Check if any decision's question mentions ALL of the given terms."""
    for d in decisions:
        q = d.get("question", "").lower()
        if all(t.lower() in q for t in terms):
            return True
    return False


def _decisions_mention_any(decisions: list[dict], terms: list[str]) -> bool:
    """Check if any decision's question mentions ANY of the given terms."""
    for d in decisions:
        q = d.get("question", "").lower()
        if any(t.lower() in q for t in terms):
            return True
    return False


def validate_and_augment(
    decisions: list[dict],
    call_graph,
) -> list[dict]:
    """Validate decomposition completeness and inject synthetic decisions.

    Args:
        decisions: List of AtomicDecision dicts from decomposition stage.
        call_graph: CallGraph object with nodes and edges.

    Returns:
        Augmented decision list (original + any synthetic decisions).
    """
    if not call_graph or not call_graph.nodes:
        return decisions

    injected: list[dict] = []
    existing_ids = {d["id"] for d in decisions}

    # --- Check 1: Return type bridge ---
    # Find entry point return type and leaf return types
    entry_types: list[tuple[str, str, str]] = []  # (file, method, type)
    leaf_types: list[tuple[str, str, str]] = []

    # Entry points are depth-0 nodes
    callee_files = {tgt for _, tgt in call_graph.edges}
    caller_files = {src for src, _ in call_graph.edges}

    for node in call_graph.nodes:
        pairs = _extract_return_types(node.class_detail)
        if not pairs:
            continue
        if node.depth == 0 or node.file_path not in callee_files:
            # Entry point (not called by anything in graph)
            for method, ret_type in pairs:
                entry_types.append((node.file_path, method, ret_type))
        if node.file_path not in caller_files:
            # Leaf (doesn't call anything in graph)
            for method, ret_type in pairs:
                leaf_types.append((node.file_path, method, ret_type))

    # Check if any entry return type differs from any leaf return type
    entry_type_names = {t for _, _, t in entry_types}
    leaf_type_names = {t for _, _, t in leaf_types}

    mismatched = entry_type_names - leaf_type_names
    if mismatched and leaf_type_names - entry_type_names:
        # There's a type mismatch — check if any decision addresses bridging
        bridge_terms = ["bridge", "wrapper", "collector", "adapter", "convert",
                        "accumulate", "combine", "assemble", "return type",
                        "back to", "final answer", "complete answer"]
        # Also check for explicit type name mentions
        all_type_names = list(entry_type_names | leaf_type_names)

        has_bridge = (
            _decisions_mention_any(decisions, bridge_terms)
            or _decisions_mention_any(decisions, all_type_names)
        )

        if not has_bridge:
            # Find the last interface decision to depend on
            interface_ids = [
                d["id"] for d in decisions
                if d.get("category") in ("interface", "pattern")
            ]
            depends = interface_ids[-1:] if interface_ids else []

            entry_repr = ", ".join(
                f"{m}() -> {t}" for f, m, t in entry_types[:3]
            )
            leaf_repr = ", ".join(
                f"{m}() -> {t}" for f, m, t in leaf_types[:3]
            )

            synth_id = "d_bridge"
            while synth_id in existing_ids:
                synth_id += "_"

            injected.append({
                "id": synth_id,
                "question": (
                    f"The entry point returns {entry_repr} but the underlying "
                    f"implementation will produce {leaf_repr}. What new class or "
                    f"wrapper bridges these types so callers still receive the "
                    f"expected return type while the new behavior is available?"
                ),
                "relevant_files": [
                    entry_types[0][0] if entry_types else "",
                    leaf_types[0][0] if leaf_types else "",
                ],
                "depends_on": depends,
                "category": "interface",
                "_synthetic": True,
            })
            logger.info(
                f"Decomposition validator: injected bridge decision "
                f"({entry_type_names} ↔ {leaf_type_names})"
            )

    # --- Check 2: Non-happy-path coverage ---
    # Scan call graph nodes for gating/validation methods
    gate_files: list[tuple[str, str]] = []  # (file, method_hint)
    for node in call_graph.nodes:
        detail = node.class_detail.lower()
        matches = _GATE_PATTERNS.findall(detail)
        if matches:
            gate_files.append((node.file_path, matches[0]))

    if gate_files:
        # Check if any decision mentions rejection/error/abort paths
        rejection_terms = [
            "reject", "abort", "error", "fail", "refuse", "deny",
            "abstain", "invalid", "not allowed", "blocked",
            "non-happy", "unhappy", "edge case", "what happens when",
        ]
        has_rejection = _decisions_mention_any(decisions, rejection_terms)

        if not has_rejection:
            # Find the integration decisions to depend on
            integration_ids = [
                d["id"] for d in decisions
                if d.get("category") == "integration"
            ]
            # Fall back to any late decision
            depends = integration_ids[-1:] if integration_ids else [
                decisions[-1]["id"]
            ] if decisions else []

            gate_desc = ", ".join(
                f"{f.rsplit('/', 1)[-1]}:{m}" for f, m in gate_files[:3]
            )
            synth_id = "d_rejection"
            while synth_id in existing_ids:
                synth_id += "_"

            injected.append({
                "id": synth_id,
                "question": (
                    f"The call graph contains gating/validation logic ({gate_desc}). "
                    f"What happens when these gates reject or abort the request? "
                    f"What does the caller receive? How does this interact with "
                    f"the proposed changes?"
                ),
                "relevant_files": [f for f, _ in gate_files[:3]],
                "depends_on": depends,
                "category": "integration",
                "_synthetic": True,
            })
            logger.info(
                f"Decomposition validator: injected rejection decision "
                f"(gates: {gate_desc})"
            )

    if injected:
        logger.info(
            f"Decomposition validator: augmented {len(decisions)} decisions "
            f"with {len(injected)} synthetic decisions"
        )
        # Clean up relevant_files (remove empty strings)
        for d in injected:
            d["relevant_files"] = [f for f in d["relevant_files"] if f]

    return decisions + injected
